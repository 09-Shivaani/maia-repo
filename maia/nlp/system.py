"""System classes for language models.

LLMNeuronSystem   one MLP neuron (input to the down-projection) or one attention head of a causal LM
LLMBehaviorSystem the model's *output behaviour* on prompts (sentiment of continuations or the
                  log-probability of a target continuation); the language analogue of the paper's
                  ClassifierSystem used for bias identification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from maia.utils import get_device


@dataclass
class LoadedLM:
    name: str
    model: torch.nn.Module
    tokenizer: object
    n_layers: int
    n_heads: int
    head_dim: int
    d_mlp: int
    arch: str  # "gpt2" | "llama" (llama/qwen/mistral/gemma style)

    def mlp_out_proj(self, layer: int):
        if self.arch == "gpt2":
            return self.model.transformer.h[layer].mlp.c_proj
        return self.model.model.layers[layer].mlp.down_proj

    def attn_out_proj(self, layer: int):
        if self.arch == "gpt2":
            return self.model.transformer.h[layer].attn.c_proj
        return self.model.model.layers[layer].self_attn.o_proj

    def block(self, layer: int):
        return self.model.transformer.h[layer] if self.arch == "gpt2" else self.model.model.layers[layer]


def load_lm(model_name: str, device: str = "cpu", dtype: str = "float32") -> LoadedLM:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    from maia.utils import hf_from_pretrained

    model = hf_from_pretrained(AutoModelForCausalLM, model_name, getattr(torch, dtype)).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    c = model.config
    arch = "gpt2" if c.model_type == "gpt2" else "llama"
    n_heads = getattr(c, "n_head", None) or c.num_attention_heads
    hidden = getattr(c, "n_embd", None) or c.hidden_size
    head_dim = getattr(c, "head_dim", None) or hidden // n_heads
    d_mlp = getattr(c, "n_inner", None) or getattr(c, "intermediate_size", None) or 4 * hidden
    n_layers = getattr(c, "n_layer", None) or c.num_hidden_layers
    return LoadedLM(model_name, model, tok, n_layers, n_heads, head_dim, d_mlp, arch)


def highlight(tokens: List[str], acts: List[float], rel: float = 0.5) -> str:
    """Render text with strongly activating tokens marked as «token»(value)."""
    m = max(acts) if acts else 0.0
    out = []
    for t, a in zip(tokens, acts):
        if m > 0 and a >= rel * m:
            lead = t[: len(t) - len(t.lstrip())]
            out.append(f"{lead}«{t.strip()}»({a:.2f})")
        else:
            out.append(t)
    return "".join(out)


class LLMNeuronSystem:
    def __init__(self, model_name: str, layer: int, neuron_num: int, kind: str = "mlp", device: str | int = "auto",
                 loaded: LoadedLM | None = None, max_length: int = 128, batch_size: int = 16):
        self.device = get_device(device)
        self.lm = loaded or load_lm(model_name, self.device)
        self.model_name, self.layer, self.neuron_num, self.kind = model_name, int(layer), int(neuron_num), kind
        self.max_length, self.batch_size = max_length, batch_size
        if kind == "mlp" and not 0 <= self.neuron_num < self.lm.d_mlp:
            raise ValueError(f"neuron {neuron_num} out of range (d_mlp={self.lm.d_mlp})")
        if kind == "attn_head" and not 0 <= self.neuron_num < self.lm.n_heads:
            raise ValueError(f"head {neuron_num} out of range (n_heads={self.lm.n_heads})")

    def _token_acts(self, texts: List[str]) -> List[Tuple[List[str], List[float]]]:
        tok = self.lm.tokenizer
        mod = self.lm.mlp_out_proj(self.layer) if self.kind == "mlp" else self.lm.attn_out_proj(self.layer)
        store = {}
        h = mod.register_forward_pre_hook(lambda m, inp: store.__setitem__("x", inp[0]))
        out = []
        try:
            for s in range(0, len(texts), self.batch_size):
                batch = texts[s : s + self.batch_size]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
                with torch.no_grad():
                    self.lm.model(**enc)
                x = store["x"].float()  # B,T,D_in
                if self.kind == "mlp":
                    a = x[..., self.neuron_num]
                else:
                    hd = self.lm.head_dim
                    a = x[..., self.neuron_num * hd : (self.neuron_num + 1) * hd].norm(dim=-1)
                for i in range(len(batch)):
                    ids = enc.input_ids[i][enc.attention_mask[i].bool()]
                    vals = a[i][enc.attention_mask[i].bool()].cpu().tolist()
                    toks = [tok.decode([t]) for t in ids.tolist()]
                    out.append((toks, vals))
        finally:
            h.remove()
        return out

    def neuron(self, text_list) -> Tuple[List[float], List[str]]:
        """Max activation over tokens for every text, and the text with activating tokens highlighted."""
        if isinstance(text_list, str):
            text_list = [text_list]
        res = self._token_acts(list(text_list))
        acts = [round(max(v), 3) if v else 0.0 for _, v in res]
        return acts, [highlight(t, v) for t, v in res]

    def token_activations(self, text: str) -> List[Tuple[str, float]]:
        toks, vals = self._token_acts([text])[0]
        return list(zip(toks, [round(v, 3) for v in vals]))


class LLMBehaviorSystem:
    """Scores a model's behaviour on prompts.

    metric="sentiment": greedy continuation of each prompt, scored with VADER compound sentiment
                        (the metric used by BOLD, Dhamala et al. 2021) -> value in [-1, 1].
    metric="logprob":   mean log-probability of ``target`` as continuation of each prompt.
    """

    def __init__(self, model_name: str, metric: str = "sentiment", target: str | None = None, device: str | int = "auto",
                 loaded: LoadedLM | None = None, max_new_tokens: int = 30, batch_size: int = 16, description: str = ""):
        self.device = get_device(device)
        self.lm = loaded or load_lm(model_name, self.device)
        self.model_name, self.metric, self.target = model_name, metric, target
        self.max_new_tokens, self.batch_size, self.description = max_new_tokens, batch_size, description
        self.layer, self.neuron_num = "output", 0
        if metric == "sentiment":
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._vader = SentimentIntensityAnalyzer()

    @torch.no_grad()
    def generate(self, prompts: List[str]) -> List[str]:
        tok = self.lm.tokenizer
        tok.padding_side = "left"
        outs = []
        for s in range(0, len(prompts), self.batch_size):
            batch = prompts[s : s + self.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True).to(self.device)
            g = self.lm.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(g[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
        return outs

    @torch.no_grad()
    def _logprob(self, prompts: List[str]) -> List[float]:
        tok = self.lm.tokenizer
        vals = []
        for p in prompts:
            ids_p = tok(p, return_tensors="pt").input_ids
            ids_t = tok(self.target, add_special_tokens=False, return_tensors="pt").input_ids
            ids = torch.cat([ids_p, ids_t], 1).to(self.device)
            logits = self.lm.model(ids).logits.log_softmax(-1)
            n = ids_t.shape[1]
            lp = logits[0, -n - 1 : -1].gather(1, ids[0, -n:, None]).mean()
            vals.append(float(lp))
        return vals

    def neuron(self, text_list) -> Tuple[List[float], List[str]]:
        if isinstance(text_list, str):
            text_list = [text_list]
        if self.metric == "logprob":
            v = self._logprob(text_list)
            return [round(x, 3) for x in v], [f"{p} ⟶ {self.target}" for p in text_list]
        conts = self.generate(list(text_list))
        scores = [round(self._vader.polarity_scores(c)["compound"], 3) for c in conts]
        return scores, [f"{p}⟶{c}" for p, c in zip(text_list, conts)]
