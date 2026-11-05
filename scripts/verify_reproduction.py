#!/usr/bin/env python
"""Compare reproduced numbers with the paper. Checks relative orderings (the claims), and reports the
paper's numbers next to ours. Our absolute values will differ (open backbones, CC3M instead of ImageNet
exemplars, SDXL instead of DALL-E 3 unless configured), so the checks are on directions.

Paper reference values
  Fig.4 / Table A3 (avg "+" activation): ResNet-152 MILAN 2.99, CLIP-Dissect 3.37, MAIA 3.50, Human 4.15
                                         DINO 1.03 / 0.44 / 1.93 / 1.97;  CLIP-RN50 2.79 / 3.36 / 3.41 / 3.29
  Table 1 (2AFC): MAIA vs MILAN 0.73, MAIA vs Human 0.53, Human vs MILAN 0.83
  Table 2: original 0.731, l1 top-50 0.779, random-22 0.705, l1 top-22 0.757, MAIA 0.837, balanced l1 0.830/0.865
  Table A5 ablation (ResNet-152 "+"): real-only 3.53, SD-v1.5 3.56, DALL-E 3 4.64
"""
import json

from _common import ROOT, parser, setup

PAPER = {
    "fig4_pos": {"resnet152": {"milan": 2.99, "clip_dissect": 3.37, "maia": 3.50, "human": 4.15},
                 "dino": {"milan": 1.03, "clip_dissect": 0.44, "maia": 1.93, "human": 1.97},
                 "clip-rn50": {"milan": 2.79, "clip_dissect": 3.36, "maia": 3.41, "human": 3.29}},
    "table1": {"MAIA vs MILAN": 0.73, "MAIA vs Human": 0.53, "Human vs MILAN": 0.83},
    "table2": {"Original Model": 0.731, "l1 Top 50 All": 0.779, "MAIA": 0.837},
}


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main():
    p = parser(__doc__)
    p.add_argument("--maia", default="full")
    p.add_argument("--baseline", default="vlm_exemplars")
    args = p.parse_args()
    cfg = setup(args)
    tdir = ROOT / cfg["paths"]["outputs"] / "tables"
    n_pass = n_total = 0
    for f in sorted(tdir.glob("predictive_*_all_*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))["overall"]
        if args.maia in s and args.baseline in s:
            n_total += 1
            n_pass += check(f"{f.stem}: MAIA > {args.baseline}", s[args.maia]["pos"][0] > s[args.baseline]["pos"][0],
                            f"{s[args.maia]['pos'][0]:.2f} vs {s[args.baseline]['pos'][0]:.2f}")
        if args.maia in s and "human" in s:
            n_total += 1
            n_pass += check(f"{f.stem}: MAIA ≈ human (within 25%)",
                            abs(s[args.maia]["pos"][0] - s["human"]["pos"][0]) <= 0.25 * abs(s["human"]["pos"][0]),
                            f"{s[args.maia]['pos'][0]:.2f} vs {s['human']['pos'][0]:.2f}")
    t2 = tdir / "table2_spurious.json"
    if t2.exists():
        rows = {r["method"]: r["test_acc"] for r in json.loads(t2.read_text(encoding="utf-8"))}
        if "MAIA" in rows:
            for ref in ("Original Model", "All"):
                if ref in rows and not isinstance(rows[ref], str):
                    n_total += 1
                    n_pass += check(f"Table 2: MAIA > {ref}", rows["MAIA"] > rows[ref], f"{rows['MAIA']:.3f} vs {rows[ref]:.3f}")
    for f in (tdir / "table1_llm_2afc.json",):
        if f.exists():
            for k, v in json.loads(f.read_text(encoding="utf-8")).items():
                if "maia" in k.split(" vs ")[0]:
                    n_total += 1
                    n_pass += check(f"2AFC {k} > 0.5", v[0] > 0.5, f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")
    cs = tdir / "circuit_synthetic.json"
    if cs.exists():
        rows = json.loads(cs.read_text(encoding="utf-8"))
        import numpy as np

        mf = [r["maia"]["f1"] for r in rows if "maia" in r]
        cf = [r["baselines_at_true_size"]["correlation"]["f1"] for r in rows]
        if mf:
            n_total += 1
            n_pass += check("Circuit: MAIA F1 > correlation F1", np.mean(mf) > np.mean(cf), f"{np.mean(mf):.2f} vs {np.mean(cf):.2f}")
    print(f"\n{n_pass}/{n_total} checks passed. Paper reference values: {json.dumps(PAPER)}")


if __name__ == "__main__":
    main()
