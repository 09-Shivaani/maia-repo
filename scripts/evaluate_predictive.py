#!/usr/bin/env python
"""Predictive evaluation (paper Sec. 4.1, Fig. 3/4, Table A3/A4/A5) of neuron labels.

python scripts/evaluate_predictive.py --model resnet152 --methods full vlm_exemplars clip_dissect human
python scripts/evaluate_predictive.py --synthetic --methods synthetic_maia synthetic_vlm_exemplars synthetic_human
Labels are read from outputs/descriptions/<method>/<model>/<layer>_<unit>/result.json.
"""
import logging
from pathlib import Path

from _common import ROOT, parser, setup

from maia.backbone import backbone_from_config
from maia.eval.predictive import aggregate, evaluate_neuron
from maia.utils import get_device, read_json, write_json
from maia.vision.generation import build_text2image

log = logging.getLogger("predictive")


def main():
    p = parser(__doc__)
    p.add_argument("--model")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--subset", choices=["all", "human"], default="all")
    p.add_argument("--normalize", action="store_true", help="divide by 95th-percentile exemplar activation (App. C1)")
    args = p.parse_args()
    cfg = setup(args)
    device = get_device(cfg["device"])
    out_root = ROOT / cfg["paths"]["outputs"]
    gen = backbone_from_config(cfg, "prompt_generator")
    sel = backbone_from_config(cfg, "prompt_selector")
    t2i = build_text2image(cfg["evaluation"]["text2image"], device)

    if args.synthetic:
        from maia.tasks import _shared
        from maia.vision.synthetic import GroundedSAM, SyntheticNeuronSystem, all_synthetic_neurons

        seg = _shared("grounded_sam", lambda: GroundedSAM(device))
        items = [(s.label.replace(" ", "_").replace("|", "_given_"), SyntheticNeuronSystem(s, seg, device)) for s in all_synthetic_neurons()]
        model_name = "synthetic"
    else:
        from maia.vision.models import load_vision_model
        from maia.vision.system import VisionNeuronSystem

        lm = load_vision_model(args.model, device)
        units = read_json(out_root / "neurons.json")[lm.name]
        if args.subset == "human":
            units = [u for u in units if u["human_subset"]]
        items = [(f"{u['layer']}_{u['unit']}", VisionNeuronSystem(u["unit"], u["layer"], lm.name, device, loaded=lm)) for u in units]
        model_name = lm.name

    per_neuron, by_layer = [], {}
    for nid, system in items:
        labels = {}
        for m in args.methods:
            f = out_root / "descriptions" / m / model_name / nid / "result.json"
            if f.exists():
                labels[m] = read_json(f).get("label", "")
        if len(labels) < 2:
            continue
        odir = out_root / "predictive" / model_name / nid
        try:
            res = evaluate_neuron(labels, gen, sel, render=t2i, activate=lambda imgs: system.neuron(imgs)[0],
                                  n=cfg["evaluation"]["n_prompts"], modality="vision", out_dir=odir)
        except Exception:
            log.exception("eval failed for %s", nid)
            continue
        per_neuron.append(res)
        by_layer.setdefault(nid.rsplit("_", 1)[0], []).append(res)
        log.info("%s: %s", nid, {m: (round(r['pos'], 2), round(r['neu'], 2)) for m, r in res.items()})
    summary = {"overall": aggregate(per_neuron, cfg["evaluation"]["n_bootstrap"]),
               "by_layer": {l: aggregate(v, cfg["evaluation"]["n_bootstrap"]) for l, v in by_layer.items()},
               "n_neurons": len(per_neuron)}
    tag = "_".join(args.methods)
    write_json(summary, out_root / "tables" / f"predictive_{model_name}_{args.subset}_{tag}.json")
    for m, s in summary["overall"].items():
        print(f"{m:>20s}  +{s['pos'][0]:.2f} [{s['pos'][1]:.2f},{s['pos'][2]:.2f}]   -{s['neu'][0]:.2f}  (n={s['n']})")


if __name__ == "__main__":
    main()
