"""Predictive evaluation protocol (paper Sec. 4.1, Fig. 3).

For each neuron:
 (a) a Prompt Generator LM turns every candidate label into 7 positive and 7 neutral prompts;
 (b) all prompts from all labeling methods form one Prompt Dataset (42 prompts for 3 methods);
 (c) for each method, a Prompt Selector LM picks from the dataset the 7 prompts most likely to
     activate the neuron and the 7 least likely, *if the label were accurate*;
 (d) the prompts are rendered (text-to-image for vision, used as-is for text) and the mean
     activation of the positive and neutral sets is recorded.
A predictive label gives high positive and low neutral activations.

The protocol is modality-agnostic: pass ``render`` (prompt list -> inputs) and ``activate``
(inputs -> list of floats).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from maia.backbone import Backbone
from maia.utils import bootstrap_ci, read_json, write_json

log = logging.getLogger(__name__)

GEN_PROMPT = """A neuron inside a {modality} model has been described with this label:
"{label}"

Write {n} {item_kind} that should MAXIMALLY activate the neuron if the label is accurate (concrete, varied
{item_kind} that clearly contain the concept), and {n} neutral {item_kind} that are unrelated to the label
and should produce baseline activation.

Return ONLY JSON: {{"positive": [...{n} strings...], "neutral": [...{n} strings...]}}"""

SELECT_PROMPT = """A neuron inside a {modality} model is described by this label:
"{label}"

Below is a numbered list of {item_kind}. Assuming the label is accurate, choose the {n} {item_kind} that are
MOST likely to strongly activate the neuron, and the {n} that are LEAST likely to activate it.

{listing}

Return ONLY JSON with the numbers: {{"most": [{n} integers], "least": [{n} integers]}}"""


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in LM output: {text[:200]}")
    return json.loads(m.group(0))


def generate_prompts(gen: Backbone, label: str, n: int = 7, modality: str = "vision",
                     retries: int = 3) -> Dict[str, List[str]]:
    item_kind = "image prompts (one-sentence descriptions of images)" if modality == "vision" else "short text passages (1-2 sentences)"
    for _ in range(retries):
        try:
            d = _json(gen.complete(GEN_PROMPT.format(label=label, n=n, modality=modality, item_kind=item_kind), temperature=0.7))
            pos, neu = [str(x) for x in d["positive"]][:n], [str(x) for x in d["neutral"]][:n]
            if len(pos) == n and len(neu) == n:
                return {"positive": pos, "neutral": neu}
        except Exception as e:  # malformed output -> retry
            log.warning("prompt generation failed: %s", e)
    raise RuntimeError(f"could not generate prompts for label '{label}'")


def select_prompts(sel: Backbone, label: str, dataset: List[str], n: int = 7, modality: str = "vision",
                   retries: int = 3) -> Dict[str, List[str]]:
    item_kind = "image prompts" if modality == "vision" else "text passages"
    listing = "\n".join(f"{i}. {p}" for i, p in enumerate(dataset))
    for _ in range(retries):
        try:
            d = _json(sel.complete(SELECT_PROMPT.format(label=label, n=n, listing=listing, modality=modality,
                                                        item_kind=item_kind), temperature=0.0))
            most = [dataset[int(i)] for i in d["most"]][:n]
            least = [dataset[int(i)] for i in d["least"]][:n]
            if most and least:
                return {"positive": most, "neutral": least}
        except Exception as e:
            log.warning("prompt selection failed: %s", e)
    raise RuntimeError(f"could not select prompts for label '{label}'")


class RenderCache:
    """Caches rendered inputs + activations per prompt so shared prompts are generated once."""

    def __init__(self, render: Callable, activate: Callable, cache_file: str | Path | None = None):
        self.render, self.activate = render, activate
        self.cache_file = Path(cache_file) if cache_file else None
        self.acts: Dict[str, float] = read_json(self.cache_file) if self.cache_file and self.cache_file.exists() else {}

    def __call__(self, prompts: List[str]) -> List[float]:
        todo = [p for p in dict.fromkeys(prompts) if p not in self.acts]
        if todo:
            inputs = self.render(todo)
            for p, a in zip(todo, self.activate(inputs)):
                self.acts[p] = float(a)
            if self.cache_file:
                write_json(self.acts, self.cache_file)
        return [self.acts[p] for p in prompts]


def evaluate_neuron(labels: Dict[str, str], gen: Backbone, sel: Backbone, render: Callable, activate: Callable,
                    n: int = 7, modality: str = "vision", out_dir: str | Path | None = None) -> Dict[str, dict]:
    """labels: {method_name: label}. Returns {method: {"pos": mean, "neu": mean, ...}}."""
    out_dir = Path(out_dir) if out_dir else None
    prompts_file = out_dir / "prompts.json" if out_dir else None
    if prompts_file and prompts_file.exists():
        gen_prompts = read_json(prompts_file)
    else:
        gen_prompts = {m: generate_prompts(gen, l, n, modality) for m, l in labels.items() if l}
        if prompts_file:
            write_json(gen_prompts, prompts_file)
    dataset = list(dict.fromkeys(p for d in gen_prompts.values() for k in ("positive", "neutral") for p in d[k]))
    cache = RenderCache(render, activate, out_dir / "activations.json" if out_dir else None)
    results = {}
    for m, l in labels.items():
        if not l:
            continue
        s = select_prompts(sel, l, dataset, n, modality)
        pa, na = cache(s["positive"]), cache(s["neutral"])
        results[m] = {"label": l, "pos": float(np.mean(pa)), "neu": float(np.mean(na)),
                      "pos_prompts": s["positive"], "neu_prompts": s["neutral"], "pos_acts": pa, "neu_acts": na}
    if out_dir:
        write_json(results, out_dir / "predictive.json")
    return results


def aggregate(per_neuron: List[Dict[str, dict]], n_boot: int = 1000, normalizer: List[float] | None = None) -> Dict[str, dict]:
    """Average positive / neutral activation per method across neurons with bootstrap 95% CIs.
    ``normalizer`` (one value per neuron, e.g. the 95th-percentile activation; App. C1) divides activations."""
    methods = sorted({m for d in per_neuron for m in d})
    out = {}
    for m in methods:
        pos, neu = [], []
        for i, d in enumerate(per_neuron):
            if m in d:
                z = normalizer[i] if normalizer else 1.0
                pos.append(d[m]["pos"] / z)
                neu.append(d[m]["neu"] / z)
        out[m] = {"pos": bootstrap_ci(pos, n_boot), "neu": bootstrap_ci(neu, n_boot), "n": len(pos)}
    return out
