#!/usr/bin/env python
"""Synthetic vision neurons (paper Sec. 4.2): run MAIA and the exemplar-labeling baseline on all 85
synthetic neurons, then build 2AFC tasks (Table 1).

python scripts/run_synthetic_neurons.py --method maia
python scripts/run_synthetic_neurons.py --method vlm_exemplars
python scripts/run_synthetic_neurons.py --make-2afc --methods synthetic_maia synthetic_vlm_exemplars synthetic_human
python scripts/run_synthetic_neurons.py --llm-2afc  --methods synthetic_maia synthetic_vlm_exemplars
"""
import itertools
import logging

from _common import ROOT, parser, setup

from maia.backbone import backbone_from_config
from maia.baselines import vlm_exemplar_label
from maia.eval.human import llm_2afc, make_2afc_hits
from maia.tasks import _shared, run_synthetic_neuron
from maia.utils import get_device, read_json, write_json
from maia.vision.synthetic import GroundedSAM, SyntheticNeuronSystem, all_synthetic_neurons

log = logging.getLogger("synthetic")


def nid(spec):
    return spec.label.replace(" ", "_").replace("|", "_given_")


def main():
    p = parser(__doc__)
    p.add_argument("--method", choices=["maia", "vlm_exemplars"])
    p.add_argument("--make-2afc", action="store_true")
    p.add_argument("--llm-2afc", action="store_true")
    p.add_argument("--methods", nargs="*", default=[])
    args = p.parse_args()
    cfg = setup(args)
    out = ROOT / cfg["paths"]["outputs"]
    specs = all_synthetic_neurons()
    if args.method:
        device = get_device(cfg["device"])
        for s in specs:
            d = out / "descriptions" / f"synthetic_{args.method}" / "synthetic" / nid(s)
            if (d / "result.json").exists():
                continue
            if args.method == "maia":
                r = run_synthetic_neuron(cfg, s, d)
            else:
                seg = _shared("grounded_sam", lambda: GroundedSAM(device))
                sysn = SyntheticNeuronSystem(s, seg, device, exemplar_images=str(ROOT / cfg["paths"]["cc3m_images"]))
                _, masks = sysn.dataset_exemplars(15)
                r = {**vlm_exemplar_label(backbone_from_config(cfg, "describer"), masks), "gt": s.gt_description}
                write_json(r, d / "result.json")
            log.info("%s -> %s", s.label, r.get("label"))
    if args.make_2afc or args.llm_2afc:
        gt = {nid(s): s.gt_description for s in specs}
        descs = {}
        for m in args.methods:
            descs[m] = {}
            for s in specs:
                f = out / "descriptions" / m / "synthetic" / nid(s) / "result.json"
                if f.exists():
                    r = read_json(f)
                    descs[m][nid(s)] = r.get("description") or r.get("label")
        if args.make_2afc:
            n = make_2afc_hits(gt, descs, out / "human" / "2afc_hits.csv")
            print(f"wrote {n} 2AFC comparisons to {out / 'human' / '2afc_hits.csv'} (10 judges each on MTurk)")
        if args.llm_2afc:
            judge = backbone_from_config(cfg, "judge")
            res = {f"{a} vs {b}": llm_2afc(judge, gt, descs[a], descs[b]) for a, b in itertools.combinations(args.methods, 2)}
            write_json(res, out / "tables" / "table1_llm_2afc.json")
            for k, v in res.items():
                print(f"{k}: {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")


if __name__ == "__main__":
    main()
