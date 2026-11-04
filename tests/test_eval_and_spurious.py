import json

import numpy as np

from maia.backbone import ScriptedBackbone
from maia.eval.predictive import aggregate, evaluate_neuron
from maia import spurious as sp


def test_predictive_protocol():
    pos = [f"dog {i}" for i in range(7)]
    neu = [f"car {i}" for i in range(7)]
    gen = ScriptedBackbone([json.dumps({"positive": pos, "neutral": neu}),
                            json.dumps({"positive": [f"cat {i}" for i in range(7)], "neutral": neu})])
    # dataset order: dog 0-6, car 0-6, cat 0-6
    sel = ScriptedBackbone([json.dumps({"most": list(range(7)), "least": list(range(7, 14))}),
                            json.dumps({"most": list(range(14, 21)), "least": list(range(7, 14))})])
    act = lambda xs: [5.0 if x.startswith("dog") else 0.0 for x in xs]  # noqa: E731
    r = evaluate_neuron({"good": "dogs", "bad": "cats"}, gen, sel, render=lambda x: x, activate=act, modality="language")
    assert r["good"]["pos"] == 5.0 and r["bad"]["pos"] == 0.0 and r["good"]["neu"] == 0.0
    agg = aggregate([r, r])
    assert agg["good"]["pos"][0] == 5.0


def test_l1_select_hits_target_sparsity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 60))
    y = (X[:, :4] @ np.array([2, -2, 1.5, 1]) > 0).astype(int) + 2 * (X[:, 4] > 0)
    units, _, C = sp.l1_select(X, y, 10)
    assert 8 <= len(units) <= 12
    assert sp.fit_unregularized(X, y, X, y, units) > 0.6
