#!/usr/bin/env python
"""Precompute top-15 dataset exemplars for all units of the evaluated layers (paper Sec. 3.2).

python scripts/compute_exemplars.py --models resnet152 dino clip-rn50
"""
from _common import ROOT, parser, setup

from maia.utils import get_device
from maia.vision.exemplars import cache_path, compute_layer_exemplars, list_images
from maia.vision.models import TABLE_A3_UNITS, load_vision_model


def main():
    p = parser(__doc__)
    p.add_argument("--models", nargs="+", default=["resnet152", "dino", "clip-rn50"])
    p.add_argument("--layers", nargs="*", default=None, help="default: the paper's layers (Table A3)")
    p.add_argument("--image-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    cfg = setup(args)
    device = get_device(cfg["device"])
    paths = list_images(args.image_dir or ROOT / cfg["paths"]["cc3m_images"], args.limit)
    print(f"{len(paths)} images")
    for m in args.models:
        lm = load_vision_model(m, device)
        for layer in args.layers or list(TABLE_A3_UNITS[lm.name]):
            out = cache_path(ROOT / cfg["paths"]["exemplar_cache"], lm.name, layer)
            if out.exists():
                print(f"skip {out}")
                continue
            compute_layer_exemplars(lm, layer, paths, k=cfg["tools"]["exemplars_k"], batch_size=args.batch_size,
                                    num_workers=args.workers, device=device, out_path=out)
            print(f"saved {out}")


if __name__ == "__main__":
    main()
