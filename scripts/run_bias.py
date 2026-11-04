#!/usr/bin/env python
"""Bias identification in ResNet-152 ImageNet output classes (paper Sec. 5.2, Fig. 10).

python scripts/run_bias.py --classes "Labrador retriever" "beer glass" suit flagpole tench
"""
import logging

from _common import ROOT, parser, setup

from maia.tasks import run_bias_detection

log = logging.getLogger("bias")


def main():
    p = parser(__doc__)
    p.add_argument("--classes", nargs="+", default=["Labrador retriever", "beer glass", "suit", "flagpole", "tench"])
    p.add_argument("--model", default="resnet152")
    args = p.parse_args()
    cfg = setup(args)
    for c in args.classes:
        d = ROOT / cfg["paths"]["outputs"] / "bias" / c.replace(" ", "_")
        r = run_bias_detection(cfg, c, d, args.model)
        print(f"[{c}] {r['bias']}")


if __name__ == "__main__":
    main()
