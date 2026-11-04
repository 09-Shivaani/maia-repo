#!/usr/bin/env python
"""Run MAIA on the sampled neurons (paper Sec. 4.1) or on one neuron.

python scripts/run_neuron_descriptions.py --model resnet152                       # all sampled units
python scripts/run_neuron_descriptions.py --model resnet152 --layer layer4 --unit 122
python scripts/run_neuron_descriptions.py --model resnet152 --subset human --ablation real_only
"""
import logging

from _common import ROOT, parser, setup

from maia.tasks import ABLATIONS, run_neuron_description
from maia.utils import read_json

log = logging.getLogger("run")


def main():
    p = parser(__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--layer")
    p.add_argument("--unit", type=int)
    p.add_argument("--subset", choices=["all", "human"], default="all")
    p.add_argument("--ablation", choices=list(ABLATIONS), default="full")
    p.add_argument("--t2i", choices=["default", "dalle3", "sdxl"], default="default",
                   help="text2image backbone (paper Fig. 7 compares SD-v1.5 with DALL-E 3)")
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    cfg = setup(args)
    t2i_cfg = None
    if args.t2i == "dalle3":
        t2i_cfg = {"backend": "dalle3", "model": "dall-e-3"}
    elif args.t2i == "sdxl":
        t2i_cfg = dict(cfg["evaluation"]["text2image"])
    tag = args.tag or (args.ablation if args.t2i == "default" else f"{args.ablation}_{args.t2i}")
    if args.layer is not None:
        units = [{"model": args.model, "layer": args.layer, "unit": args.unit}]
    else:
        units = read_json(ROOT / cfg["paths"]["outputs"] / "neurons.json")[args.model]
        if args.subset == "human":
            units = [u for u in units if u["human_subset"]]
    for u in units:
        out_dir = ROOT / cfg["paths"]["outputs"] / "descriptions" / tag / u["model"] / f"{u['layer']}_{u['unit']}"
        if (out_dir / "result.json").exists():
            continue
        try:
            r = run_neuron_description(cfg, u["model"], u["layer"], u["unit"], out_dir, args.ablation, t2i_cfg)
            log.info("%s %s %d -> %s", u["model"], u["layer"], u["unit"], r["label"])
        except Exception:
            log.exception("failed on %s", u)


if __name__ == "__main__":
    main()
