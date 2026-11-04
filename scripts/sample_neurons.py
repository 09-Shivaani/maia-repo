#!/usr/bin/env python
"""Sample the evaluated units: 100 per model with the paper's per-layer counts (Table A3), and the 25%
subset used for human labels and ablations (App. C2/C3).  Writes outputs/neurons.json."""
import random

from _common import ROOT, parser, setup

from maia.utils import get_device, write_json
from maia.vision.models import TABLE_A3_UNITS, load_vision_model


def n_units(lm, layer):
    mod = lm.resolve_layer(layer)
    for attr in ("out_channels", "num_features", "out_features"):
        if hasattr(mod, attr):
            return getattr(mod, attr)
    # resolve by a dummy forward
    import torch

    from maia.vision.models import forward_model

    store = {}
    h = mod.register_forward_hook(lambda m, i, o: store.__setitem__("o", o[0] if isinstance(o, tuple) else o))
    with torch.no_grad():
        forward_model(lm, torch.zeros(1, 3, lm.resolution, lm.resolution, device=next(lm.model.parameters()).device))
    h.remove()
    o = store["o"]
    return o.shape[1] if o.ndim == 4 else o.shape[-1]


def main():
    p = parser(__doc__)
    p.add_argument("--models", nargs="+", default=["resnet152", "dino", "clip-rn50"])
    p.add_argument("--human-frac", type=float, default=0.25)
    args = p.parse_args()
    cfg = setup(args)
    rng = random.Random(cfg["seed"])
    out = {}
    for m in args.models:
        lm = load_vision_model(m, get_device(cfg["device"]))
        units = []
        for layer, k in TABLE_A3_UNITS[lm.name].items():
            n = n_units(lm, layer)
            for u in rng.sample(range(n), k):
                units.append({"model": lm.name, "layer": layer, "unit": u})
        sub = []
        for layer in TABLE_A3_UNITS[lm.name]:
            lay = [x for x in units if x["layer"] == layer]
            sub += rng.sample(lay, max(1, round(args.human_frac * len(lay))))
        for x in units:
            x["human_subset"] = x in sub
        out[lm.name] = units
        print(f"{lm.name}: {len(units)} units, {len(sub)} in the 25% subset")
    write_json(out, ROOT / cfg["paths"]["outputs"] / "neurons.json")


if __name__ == "__main__":
    main()
