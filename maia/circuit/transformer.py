"""Node-level circuit systems for real transformers (attention heads + MLP blocks).

* ``LMAdapter``  causal LMs from Hugging Face (GPT-2, Qwen2.5, Llama, ...) on MIB tasks (IOI, ...)
* ``ViTAdapter`` Hugging Face ViT image classifiers (vision transformers), task = logit difference
                 between two classes on paired clean / counterfactual image sets

Ablation modes: "patching" (activation from the counterfactual input, MIB default), "mean", "zero".
Metric: logit(correct) - logit(incorrect) at the answer position, averaged over examples.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from maia.circuit.base import CircuitSystem

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
class LMAdapter:
    def __init__(self, loaded_lm):
        self.lm = loaded_lm
        self.n_layers, self.n_heads, self.head_dim = loaded_lm.n_layers, loaded_lm.n_heads, loaded_lm.head_dim
        self.device = next(loaded_lm.model.parameters()).device

    def head_module(self, l):
        return self.lm.attn_out_proj(l)

    def mlp_module(self, l):
        b = self.lm.block(l)
        return b.mlp

    def embed_module(self):
        return self.lm.model.get_input_embeddings()

    def encode(self, prompts: List[str]):
        tok = self.lm.tokenizer
        tok.padding_side = "right"
        enc = tok(prompts, return_tensors="pt", padding=True).to(self.device)
        last = enc.attention_mask.sum(1) - 1
        return {"enc": enc, "last": last}

    def logits(self, inp) -> torch.Tensor:
        out = self.lm.model(**inp["enc"]).logits
        return out[torch.arange(out.shape[0], device=out.device), inp["last"]]


class ViTAdapter:
    def __init__(self, model_id: str = "google/vit-base-patch16-224", device: str = "cuda"):
        from transformers import AutoImageProcessor, ViTForImageClassification

        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = ViTForImageClassification.from_pretrained(model_id).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        c = self.model.config
        self.n_layers, self.n_heads = c.num_hidden_layers, c.num_attention_heads
        self.head_dim = c.hidden_size // c.num_attention_heads
        self.device = device
        self.id2label = c.id2label

    def head_module(self, l):
        return self.model.vit.encoder.layer[l].attention.output.dense

    def mlp_module(self, l):
        return self.model.vit.encoder.layer[l].output.dense

    def embed_module(self):
        return self.model.vit.embeddings

    def encode(self, images):
        return {"pixel_values": self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device)}

    def logits(self, inp):
        return self.model(pixel_values=inp["pixel_values"]).logits

    def class_index(self, name: str) -> int:
        name = name.lower()
        for i, l in self.id2label.items():
            if name in [s.strip().lower() for s in l.split(",")]:
                return int(i)
        raise KeyError(name)


# ---------------------------------------------------------------------------
@dataclass
class PairBatch:
    clean: dict
    corrupt: dict
    correct: torch.Tensor
    incorrect: torch.Tensor


class TransformerCircuitSystem(CircuitSystem):
    def __init__(self, adapter, clean: Sequence, corrupt: Sequence, correct: Sequence[int], incorrect: Sequence[int],
                 batch_size: int = 16, mode: str = "patching", task_description: str = ""):
        self.ad, self.mode, self.task_description = adapter, mode, task_description
        self.nodes = [f"L{l}.H{h}" for l in range(adapter.n_layers) for h in range(adapter.n_heads)] + \
                     [f"L{l}.MLP" for l in range(adapter.n_layers)]
        dev = adapter.device
        self.batches: List[PairBatch] = []
        for s in range(0, len(clean), batch_size):
            sl = slice(s, s + batch_size)
            self.batches.append(PairBatch(adapter.encode(list(clean[sl])), adapter.encode(list(corrupt[sl])),
                                          torch.tensor(list(correct[sl]), device=dev), torch.tensor(list(incorrect[sl]), device=dev)))
        self._corrupt_cache = [self._record(b.corrupt) for b in self.batches]
        self._clean_cache = [self._record(b.clean) for b in self.batches]
        self._means = self._compute_means()
        self._metric_cache: Dict[tuple, float] = {}

    # -- recording ------------------------------------------------------------------------
    @torch.no_grad()
    def _record(self, inp) -> Dict[str, torch.Tensor]:
        rec, hooks = {}, []
        for l in range(self.ad.n_layers):
            hooks.append(self.ad.head_module(l).register_forward_pre_hook(
                lambda m, i, l=l: rec.__setitem__(f"H{l}", i[0].detach().clone())))
            hooks.append(self.ad.mlp_module(l).register_forward_hook(
                lambda m, i, o, l=l: rec.__setitem__(f"M{l}", (o[0] if isinstance(o, tuple) else o).detach().clone())))
        try:
            self.ad.logits(inp)
        finally:
            for h in hooks:
                h.remove()
        return rec

    def _compute_means(self):
        means = {}
        for k in self._clean_cache[0]:
            means[k] = torch.cat([c[k].flatten(0, 1) for c in self._clean_cache]).mean(0)
        return means

    # -- patched forward -------------------------------------------------------------------
    def _parse(self, nodes: Iterable[str]):
        heads: Dict[int, List[int]] = {}
        mlps: set = set()
        for n in nodes:
            l, c = n.split(".")
            l = int(l[1:])
            if c == "MLP":
                mlps.add(l)
            else:
                heads.setdefault(l, []).append(int(c[1:]))
        return heads, mlps

    def _source(self, key, bi, mode, like):
        if mode == "zero":
            return torch.zeros_like(like)
        if mode == "mean":
            return self._means[key].expand_as(like)
        return self._corrupt_cache[bi][key]

    def _run(self, bi: int, ablate: Iterable[str], interventions: Dict[str, float], mode: str, grad: bool = False):
        b = self.batches[bi]
        heads, mlps = self._parse(ablate)
        iv_heads: Dict[int, Dict[int, float]] = {}
        iv_mlps: Dict[int, float] = {}
        for n, v in (interventions or {}).items():  # value = multiplicative scale on the node's output
            l, c = n.split(".")
            li = int(l[1:])
            if c == "MLP":
                iv_mlps[li] = float(v)
            else:
                iv_heads.setdefault(li, {})[int(c[1:])] = float(v)
        hd = self.ad.head_dim
        hooks = []
        for l in range(self.ad.n_layers):
            if l in heads or l in iv_heads:
                def pre(m, inp, l=l):
                    x = inp[0].clone()
                    if l in heads:
                        src = self._source(f"H{l}", bi, mode, x)
                        for h in heads[l]:
                            x[..., h * hd:(h + 1) * hd] = src[..., h * hd:(h + 1) * hd]
                    for h, s in iv_heads.get(l, {}).items():
                        x[..., h * hd:(h + 1) * hd] *= s
                    return (x,) + tuple(inp[1:])
                hooks.append(self.ad.head_module(l).register_forward_pre_hook(pre))
            if l in mlps or l in iv_mlps:
                def post(m, inp, out, l=l):
                    o = out[0] if isinstance(out, tuple) else out
                    o = self._source(f"M{l}", bi, mode, o).clone() if l in mlps else o
                    if l in iv_mlps:
                        o = o * iv_mlps[l]
                    return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
                hooks.append(self.ad.mlp_module(l).register_forward_hook(post))
        try:
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                lg = self.ad.logits(b.clean)
        finally:
            for h in hooks:
                h.remove()
        idx = torch.arange(lg.shape[0], device=lg.device)
        return lg[idx, b.correct] - lg[idx, b.incorrect]

    def metric(self, ablate: Iterable[str] = (), interventions: Dict[str, float] | None = None, mode: str | None = None) -> float:
        mode = mode or self.mode
        ab = tuple(sorted(set(ablate)))
        key = (ab, tuple(sorted((interventions or {}).items())), mode)
        if key in self._metric_cache:
            return self._metric_cache[key]
        vals = torch.cat([self._run(i, ab, interventions or {}, mode) for i in range(len(self.batches))])
        v = float(vals.float().mean())
        self._metric_cache[key] = v
        return v

    def per_example_metric(self, ablate: Iterable[str] = (), mode: str | None = None) -> np.ndarray:
        return torch.cat([self._run(i, list(ablate), {}, mode or self.mode) for i in range(len(self.batches))]).float().cpu().numpy()

    def _last(self, bi):
        inp = self.batches[bi].clean
        return inp["last"] if "last" in inp else None

    def node_activations(self, node: str) -> np.ndarray:
        l, c = node.split(".")
        l = int(l[1:])
        out = []
        for bi, rec in enumerate(self._clean_cache):
            last = self._last(bi)
            if c == "MLP":
                x = rec[f"M{l}"]
            else:
                h = int(c[1:])
                x = rec[f"H{l}"][..., h * self.ad.head_dim:(h + 1) * self.ad.head_dim]
            x = x[torch.arange(x.shape[0]), last] if last is not None else x[:, 0]
            out.append(x.norm(dim=-1).float().cpu())
        return torch.cat(out).numpy()

    def target_values(self) -> np.ndarray:
        return self.per_example_metric()

    def attribution_scores(self) -> Dict[str, float]:
        """Edge-free node-level EAP: |Σ grad · (corrupt - clean)| averaged over examples."""
        scores = {n: 0.0 for n in self.nodes}
        n_ex = 0
        hd = self.ad.head_dim
        for bi, b in enumerate(self.batches):
            acts, hooks = {}, []
            emb = self.ad.embed_module()
            hooks.append(emb.register_forward_hook(lambda m, i, o: (o[0] if isinstance(o, tuple) else o).requires_grad_(True)))
            for l in range(self.ad.n_layers):
                def pre(m, inp, l=l):
                    inp[0].retain_grad()
                    acts[f"H{l}"] = inp[0]
                def post(m, inp, out, l=l):
                    o = out[0] if isinstance(out, tuple) else out
                    o.retain_grad()
                    acts[f"M{l}"] = o
                hooks.append(self.ad.head_module(l).register_forward_pre_hook(pre))
                hooks.append(self.ad.mlp_module(l).register_forward_hook(post))
            try:
                with torch.enable_grad():
                    lg = self.ad.logits(b.clean)
                    idx = torch.arange(lg.shape[0], device=lg.device)
                    (lg[idx, b.correct] - lg[idx, b.incorrect]).sum().backward()
            finally:
                for h in hooks:
                    h.remove()
            cor = self._corrupt_cache[bi]
            for l in range(self.ad.n_layers):
                gH = acts[f"H{l}"].grad
                dH = cor[f"H{l}"] - acts[f"H{l}"].detach()
                prod = (gH * dH).float()
                for h in range(self.ad.n_heads):
                    scores[f"L{l}.H{h}"] += float(prod[..., h * hd:(h + 1) * hd].sum(dim=(-1, -2)).abs().sum())
                gM = acts[f"M{l}"].grad
                dM = cor[f"M{l}"] - acts[f"M{l}"].detach()
                scores[f"L{l}.MLP"] += float((gM * dM).float().sum(dim=(-1, -2)).abs().sum())
            n_ex += len(b.correct)
            self.ad_zero_grad()
        return {k: v / n_ex for k, v in scores.items()}

    def ad_zero_grad(self):
        m = getattr(self.ad, "model", None) or self.ad.lm.model
        m.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# MIB task loaders
# ---------------------------------------------------------------------------
def load_mib_ioi(tokenizer, split: str = "test", n: int = 200, counterfactual: str = "s2_io_flip_counterfactual",
                 seed: int = 0, cache_dir: str | None = None):
    """MIB IOI (Mueller et al. 2025; hf: mib-bench/ioi), filtered like the official circuit track:
    the indirect object, subject and random name must be single-token-length-matched."""
    from datasets import load_dataset

    ds = load_dataset("mib-bench/ioi", split=split, cache_dir=cache_dir).shuffle(seed=seed)
    L = lambda s: len(tokenizer(s, add_special_tokens=False).input_ids)  # noqa: E731
    clean, corrupt, cor, inc = [], [], [], []
    for row in ds:
        md = row["metadata"]
        io, s = f" {md['indirect_object']}", f" {md['subject']}"
        if L(io) != L(s) or (md.get("random_c") and L(io) != L(f" {md['random_c']}")):
            continue
        cf = row[counterfactual]["prompt"]
        if L(row["prompt"]) != L(cf):
            continue
        clean.append(row["prompt"])
        corrupt.append(cf)
        cor.append(tokenizer(io, add_special_tokens=False).input_ids[0])
        inc.append(tokenizer(s, add_special_tokens=False).input_ids[0])
        if len(clean) >= n:
            break
    return clean, corrupt, cor, inc
