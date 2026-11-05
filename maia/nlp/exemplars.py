"""Top-activating text exemplars for every MLP neuron / head of a layer, over a real text corpus
(WikiText-103 by default; scripts/download_datasets.py --text-corpus)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch

from maia.nlp.system import LoadedLM

log = logging.getLogger(__name__)


def load_corpus(path: str | Path, n: int | None = None, min_chars: int = 80) -> List[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["text"].strip()
            if len(t) >= min_chars:
                out.append(t)
            if n and len(out) >= n:
                break
    if not out:
        raise FileNotFoundError(f"empty corpus {path}; run scripts/download_datasets.py --text-corpus")
    return out


@torch.no_grad()
def compute_text_exemplars(lm: LoadedLM, layer: int, texts: List[str], kind: str = "mlp", k: int = 15,
                           max_length: int = 64, batch_size: int = 32, device: str = "cuda",
                           out_path: str | Path | None = None):
    tok = lm.tokenizer
    tok.padding_side = "right"
    mod = lm.mlp_out_proj(layer) if kind == "mlp" else lm.attn_out_proj(layer)
    store = {}
    h = mod.register_forward_pre_hook(lambda m, inp: store.__setitem__("x", inp[0]))
    top_a = top_i = None
    try:
        for s in range(0, len(texts), batch_size):
            enc = tok(texts[s : s + batch_size], return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length).to(device)
            lm.model(**enc)
            x = store["x"].float()
            if kind != "mlp":
                B, T, _ = x.shape
                x = x.view(B, T, lm.n_heads, lm.head_dim).norm(dim=-1)
            x = x.masked_fill(~enc.attention_mask.bool()[..., None], -float("inf"))
            a = x.amax(1).T  # U,B
            idx = torch.arange(s, s + a.shape[1], device=a.device).expand_as(a)
            if top_a is None:
                ca, ci = a, idx
            else:
                ca, ci = torch.cat([top_a, a], 1), torch.cat([top_i, idx], 1)
            top_a, sel = ca.topk(min(k, ca.shape[1]), 1)
            top_i = ci.gather(1, sel)
            if (s // batch_size) % 200 == 0:
                log.info("text exemplars layer %d: %d/%d", layer, s, len(texts))
    finally:
        h.remove()
    res = {"acts": top_a.cpu().numpy(), "idx": top_i.cpu().numpy()}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **res)
    return res
