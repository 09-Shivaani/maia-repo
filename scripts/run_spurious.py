#!/usr/bin/env python
"""Spurious-feature removal on Spawrious O2O-Easy (paper Sec. 5.1, Table 2).

Steps (each cached under outputs/spurious/):
  train     ResNet-18, one epoch                      -> resnet18_spawrious.pth
  features  512-d final-layer features (val / test / balanced pool)
  maia      run MAIA on the l1 top-50 units           -> per-unit SELECTIVE / SPURIOUS
  table     every row of Table 2                      -> tables/table2_spurious.json

python scripts/run_spurious.py --steps train features maia table
"""
import logging
import random

import numpy as np

from _common import ROOT, parser, setup

from maia import spurious as sp
from maia.utils import get_device, read_json, set_seed, write_json

log = logging.getLogger("spurious")


def layer4_max(lm, paths, device, bs=128):
    import torch
    from PIL import Image

    from maia.vision.system import pil_batch_to_tensor, preprocess_pil

    store, out = {}, []
    h = lm.model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("o", o))
    with torch.no_grad():
        for s in range(0, len(paths), bs):
            imgs = [preprocess_pil(Image.open(p).convert("RGB"), 224) for p in paths[s:s + bs]]
            lm.model(pil_batch_to_tensor(imgs, *lm.normalize, device))
            out.append(store["o"].flatten(2).amax(2).cpu().numpy())
    h.remove()
    return np.concatenate(out)


def main():
    p = parser(__doc__)
    p.add_argument("--steps", nargs="+", default=["train", "features", "maia", "table"])
    p.add_argument("--milan-units", default=None, help="json list of units chosen from MILAN captions (App. F2)")
    p.add_argument("--vlm-units", default=None, help="json list of units chosen from VLM exemplar captions")
    args = p.parse_args()
    cfg = setup(args)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    out = ROOT / cfg["paths"]["outputs"] / "spurious"
    out.mkdir(parents=True, exist_ok=True)
    pool, test = sp.build_o2o_easy(ROOT / cfg["paths"]["spawrious_root"])
    train, val = sp.split_train_val(pool, seed=cfg["seed"])
    test_eval, bal_pool = sp.split_test_balanced_pool(test, seed=cfg["seed"])
    log.info("train %d  val %d  test %d (+%d held out for balanced sets)", len(train), len(val), len(test_eval), len(bal_pool))
    ckpt = out / "resnet18_spawrious.pth"

    if "train" in args.steps and not ckpt.exists():
        sp.train_resnet18(train, device, out=ckpt)

    from maia.vision.models import load_vision_model

    lm = load_vision_model("resnet18-spawrious", device, checkpoint=str(ckpt), num_classes=4)
    feat_file = out / "features.npz"
    if "features" in args.steps and not feat_file.exists():
        Xv, yv, _ = sp.extract_features(lm.model, val, device)
        Xt, yt, Lt = sp.extract_features(lm.model, test_eval, device)
        Xp, yp, _ = sp.extract_features(lm.model, bal_pool, device)
        np.savez_compressed(feat_file, Xv=Xv, yv=yv, Xt=Xt, yt=yt, Lt=Lt, Xp=Xp, yp=yp)
    F = np.load(feat_file)
    Xv, yv, Xt, yt, Xp, yp = F["Xv"], F["yv"], F["Xt"], F["yt"], F["Xp"], F["yp"]

    top50, _, C50 = sp.l1_select(Xv, yv, 50)
    top50 = [int(u) for u in top50]
    write_json(top50, out / "l1_top50_units.json")

    if "maia" in args.steps:
        from maia.tasks import run_spurious_unit

        # exemplar pool = the unbalanced validation images only
        vdir = out / "val_images"
        if not vdir.exists():
            vdir.mkdir()
            for i, s in enumerate(val):
                (vdir / f"{i:05d}.png").symlink_to(s.path)
        # A unit counts as "active" (red-outlined region) when its max activation exceeds its 90th
        # percentile on the validation images.
        mx_file = out / "val_layer4_max.npy"
        if not mx_file.exists():
            np.save(mx_file, layer4_max(lm, [s.path for s in val], device))
        floors = np.percentile(np.load(mx_file), 90, axis=0)
        for u in top50:
            d = out / "maia" / f"unit_{u}"
            if (d / "result.json").exists():
                continue
            r = run_spurious_unit(cfg, lm, u, d, str(vdir), activation_floor=float(floors[u]))
            log.info("unit %d -> %s", u, r["label"])

    if "table" in args.steps:
        rows = []
        acc_orig = float((F["Lt"].argmax(1) == yt).mean())
        rows.append(("All", "Original Model", 512, False, acc_orig))
        acc50, _ = sp.l1_eval(Xv, yv, Xt, yt, C50)
        rows.append(("l1 Top 50", "All", len(top50), False, acc50))
        rng = random.Random(cfg["seed"])
        maia_units = [u for u in top50 if (out / "maia" / f"unit_{u}" / "result.json").exists()
                      and read_json(out / "maia" / f"unit_{u}" / "result.json")["label"] == "SELECTIVE"]
        k = len(maia_units) or 22
        rnd = [sp.fit_unregularized(Xv, yv, Xt, yt, rng.sample(top50, k)) for _ in range(100)]
        rows.append(("l1 Top 50", "Random", k, False, f"{np.mean(rnd):.3f} ± {np.std(rnd):.3f}"))
        topk, _, Ck = sp.l1_select(Xv, yv, k)
        acck, _ = sp.l1_eval(Xv, yv, Xt, yt, Ck)
        rows.append(("l1 Top 50", f"l1 Top {k}", len(topk), False, acck))
        for name, f in (("MILAN", args.milan_units), ("MILAN (VLM)", args.vlm_units)):
            if f:
                units = read_json(f)
                rows.append(("l1 Top 50", name, len(units), False, sp.fit_unregularized(Xv, yv, Xt, yt, units)))
        if maia_units:
            rows.append(("l1 Top 50", "MAIA", len(maia_units), False, sp.fit_unregularized(Xv, yv, Xt, yt, maia_units)))
        # balanced references (ten balanced sets of 320)
        vs = sp.balanced_sets(val, bal_pool, n_sets=10, size=320, seed=cfg["seed"])
        vidx = {s.path: i for i, s in enumerate(val)}
        pidx = {s.path: i for i, s in enumerate(bal_pool)}
        accs_t, nz_t, accs_k = [], [], []
        for S in vs:
            Xb = np.stack([Xv[vidx[s.path]] if s.path in vidx else Xp[pidx[s.path]] for s in S])
            yb = np.array([s.label for s in S])
            a, nz, _ = sp.l1_tune_balanced(Xb, yb, Xt, yt)
            accs_t.append(a)
            nz_t.append(nz)
            _, _, Cb = sp.l1_select(Xb, yb, k)
            accs_k.append(sp.l1_eval(Xb, yb, Xt, yt, Cb)[0])
        rows.append(("All", "l1 Hyper. Tuning", int(np.median(nz_t)), True, float(np.mean(accs_t))))
        rows.append(("All", f"l1 Top {k}", k, True, float(np.mean(accs_k))))
        table = [{"subset": a, "method": b, "units": c, "balanced": d, "test_acc": e} for a, b, c, d, e in rows]
        write_json(table, ROOT / cfg["paths"]["outputs"] / "tables" / "table2_spurious.json")
        for r in table:
            acc = r["test_acc"] if isinstance(r["test_acc"], str) else f"{r['test_acc']:.3f}"
            print(f"{r['subset']:>10s} | {r['method']:>18s} | {r['units']:>4} | {'✓' if r['balanced'] else '✗'} | {acc}")


if __name__ == "__main__":
    main()
