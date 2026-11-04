"""Human-study helpers.

* 2AFC (paper Table 1): judges see a synthetic neuron's ground-truth label and two descriptions and pick
  the better match. ``make_2afc_hits`` writes an MTurk-ready CSV; ``analyze_2afc`` computes the proportion
  of trials won by method A with a 95% CI. ``llm_2afc`` is an automatic proxy using an LM judge.
* Human utility (paper Table A7): participants see a description and four sets of 4 images (one set of
  top-activating images, three random sets) and choose the activating set (chance 25%).
"""
from __future__ import annotations

import csv
import itertools
import random
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from maia.backbone import Backbone
from maia.utils import bootstrap_ci


def make_2afc_hits(gt: Dict[str, str], descs: Dict[str, Dict[str, str]], out_csv: str | Path,
                   pairs: List[tuple] | None = None, seed: int = 0) -> int:
    """gt: {neuron_id: gt_label}; descs: {method: {neuron_id: description}}."""
    rng = random.Random(seed)
    methods = list(descs)
    pairs = pairs or list(itertools.combinations(methods, 2))
    rows = []
    for a, b in pairs:
        for nid, g in gt.items():
            if nid not in descs[a] or nid not in descs[b]:
                continue
            flip = rng.random() < 0.5
            d1, d2 = (descs[b][nid], descs[a][nid]) if flip else (descs[a][nid], descs[b][nid])
            m1, m2 = (b, a) if flip else (a, b)
            rows.append({"neuron": nid, "ground_truth": g, "description_1": d1, "description_2": d2,
                         "method_1": m1, "method_2": m2})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def analyze_2afc(results_csv: str | Path, a: str, b: str, choice_col: str = "Answer.choice") -> tuple:
    """Proportion of judgements preferring method a over b (choice_col holds 1 or 2)."""
    wins = []
    with open(results_csv) as f:
        for r in csv.DictReader(f):
            ms = {r["Input.method_1"] if "Input.method_1" in r else r["method_1"],
                  r["Input.method_2"] if "Input.method_2" in r else r["method_2"]}
            if ms != {a, b}:
                continue
            m1 = r.get("Input.method_1", r.get("method_1"))
            m2 = r.get("Input.method_2", r.get("method_2"))
            chosen = m1 if str(r[choice_col]).strip() == "1" else m2
            wins.append(1.0 if chosen == a else 0.0)
    return bootstrap_ci(wins)


JUDGE_PROMPT = """Ground-truth concept of a neuron: "{gt}"

Description 1: "{d1}"
Description 2: "{d2}"

Which description better matches the ground-truth concept? Answer with only "1" or "2"."""


def llm_2afc(judge: Backbone, gt: Dict[str, str], da: Dict[str, str], db: Dict[str, str], seed: int = 0) -> tuple:
    """Automatic 2AFC proxy: returns (mean, lo, hi) preference for A over B."""
    rng = random.Random(seed)
    wins = []
    for nid, g in gt.items():
        if nid not in da or nid not in db:
            continue
        flip = rng.random() < 0.5
        d1, d2 = (db[nid], da[nid]) if flip else (da[nid], db[nid])
        out = judge.complete(JUDGE_PROMPT.format(gt=g, d1=d1, d2=d2), temperature=0.0, max_tokens=2048)
        m = re.search(r"[12]", out)
        if not m:
            continue
        pick_a = (m.group(0) == "1") != flip
        wins.append(1.0 if pick_a else 0.0)
    return bootstrap_ci(wins)


def make_utility_hits(descriptions: Dict[str, Dict[str, str]], top_images: Dict[str, List[str]],
                      random_pool: List[str], out_csv: str | Path, set_size: int = 4, n_distractor_sets: int = 3,
                      seed: int = 0) -> int:
    """descriptions: {method: {neuron: text}}; top_images: {neuron: [image urls/paths]}."""
    rng = random.Random(seed)
    rows = []
    for method, d in descriptions.items():
        for nid, text in d.items():
            sets = [rng.sample(top_images[nid], set_size)] + [rng.sample(random_pool, set_size) for _ in range(n_distractor_sets)]
            order = list(range(len(sets)))
            rng.shuffle(order)
            row = {"method": method, "neuron": nid, "description": text, "correct_set": order.index(0) + 1}
            for pos, si in enumerate(order, 1):
                for j, img in enumerate(sets[si]):
                    row[f"set{pos}_img{j + 1}"] = img
            rows.append(row)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return len(rows)
