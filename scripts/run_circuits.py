#!/usr/bin/env python
"""MAIA-Circuit experiments (extension 2).

  synthetic  planted-circuit MLPs (xor, and_or, parity3) x seeds: recovery (P/R/F1/IoU) and faithfulness,
             MAIA-Circuit vs correlation / attribution / greedy-pruning baselines
  mib        MIB circuit-localization IOI task (Mueller et al. 2025) on GPT-2 small or Qwen2.5-0.5B,
             node level (heads + MLPs): CPR / CMD of baseline rankings and faithfulness of MAIA's circuit
  vit        ViT-B/16 (HF) class-vs-class logit difference on two real image folders
             (e.g. Spawrious labrador vs corgi), same metrics as mib

python scripts/run_circuits.py synthetic --seeds 0 1 2
python scripts/run_circuits.py mib --lm gpt2 --n 200
python scripts/run_circuits.py vit --clean-dir data/spawrious/spawrious224/0/dirt/labrador \
       --corrupt-dir data/spawrious/spawrious224/0/snow/corgi --classes "Labrador retriever" "Pembroke"
"""
import logging

from _common import ROOT, parser, setup

from maia.circuit import baselines as B
from maia.circuit.base import circuit_overlap, cpr_cmd
from maia.tasks import run_circuit_agent
from maia.utils import get_device, write_json

log = logging.getLogger("circuits")


def evaluate_set(system, nodes, truth=None):
    r = {"size": len(nodes), "faithfulness": system.faithfulness(nodes), "necessity": system.necessity(nodes)}
    if truth is not None:
        r.update(circuit_overlap(nodes, truth))
    return r


def baselines_for(system, truth=None, k=None):
    out = {}
    for name, ranking in (("correlation", B.correlation_ranking(system)), ("attribution", B.attribution_ranking(system))):
        nodes = ranking[:k] if k else B.smallest_faithful_prefix(system, ranking, 0.9)
        out[name] = evaluate_set(system, nodes, truth)
        out[name]["cpr_cmd"] = cpr_cmd(system, ranking)
    if len(system.nodes) <= 200:
        out["greedy_pruning"] = evaluate_set(system, B.greedy_pruning(system, order=B.attribution_ranking(system)[::-1]), truth)
    return out


def main():
    p = parser(__doc__)
    p.add_argument("cmd", choices=["synthetic", "mib", "vit"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--tasks", nargs="+", default=["xor", "and_or", "parity3"])
    p.add_argument("--lm", default="gpt2")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--clean-dir")
    p.add_argument("--corrupt-dir")
    p.add_argument("--classes", nargs=2)
    p.add_argument("--no-agent", action="store_true")
    args = p.parse_args()
    cfg = setup(args)
    out = ROOT / cfg["paths"]["outputs"] / "circuits"
    device = get_device(cfg["device"])
    results = []

    if args.cmd == "synthetic":
        from maia.circuit.synthetic import SyntheticCircuitSystem, build_planted_net

        for seed in args.seeds:
            net = build_planted_net(seed=seed)
            for task in args.tasks:
                sysc = SyntheticCircuitSystem(net, task, seed=seed)
                row = {"seed": seed, "task": task, "truth": sysc.ground_truth, "decoys": sysc.decoys,
                       "baselines": baselines_for(sysc, sysc.ground_truth)}
                row["baselines_at_true_size"] = baselines_for(sysc, sysc.ground_truth, k=len(sysc.ground_truth))
                if not args.no_agent:
                    a = run_circuit_agent(cfg, sysc, out / "synthetic" / f"seed{seed}_{task}")
                    row["maia"] = {**evaluate_set(sysc, a["circuit"], sysc.ground_truth), "metric_calls": a["metric_calls"],
                                   "circuit": a["circuit"], "explanation": a["explanation"]}
                results.append(row)
                log.info("seed %d %s: %s", seed, task, {k: round(v.get("f1", 0), 2) for k, v in row["baselines"].items()}
                         | ({"maia": round(row["maia"]["f1"], 2)} if "maia" in row else {}))
        write_json(results, ROOT / cfg["paths"]["outputs"] / "tables" / "circuit_synthetic.json")
        return

    from maia.circuit.transformer import LMAdapter, TransformerCircuitSystem, ViTAdapter, load_mib_ioi

    if args.cmd == "mib":
        from maia.nlp.system import load_lm

        lm = load_lm(args.lm, device)
        clean, corrupt, cor, inc = load_mib_ioi(lm.tokenizer, n=args.n)
        sysc = TransformerCircuitSystem(LMAdapter(lm), clean, corrupt, cor, inc, mode="patching",
                                        task_description=f"{args.lm} on MIB indirect-object identification (IOI): metric = "
                                        "logit(indirect object) - logit(subject) at the last position; ablation = activation "
                                        "patching from the counterfactual prompt with the names swapped")
        tag = f"mib_ioi_{args.lm.replace('/', '_')}"
    else:
        from maia.vision.exemplars import list_images
        from PIL import Image

        ad = ViTAdapter(device=device)
        ca, cb = ad.class_index(args.classes[0]), ad.class_index(args.classes[1])
        clean = [Image.open(p).convert("RGB") for p in list_images(args.clean_dir, args.n)]
        corrupt = [Image.open(p).convert("RGB") for p in list_images(args.corrupt_dir, len(clean))]
        n = min(len(clean), len(corrupt))
        sysc = TransformerCircuitSystem(ad, clean[:n], corrupt[:n], [ca] * n, [cb] * n, batch_size=16, mode="patching",
                                        task_description=f"ViT-B/16 classifying '{args.classes[0]}' vs '{args.classes[1]}': metric = "
                                        "logit difference on images of the first class; ablation = patching from images of the second class")
        tag = f"vit_{args.classes[0]}_vs_{args.classes[1]}".replace(" ", "_")

    row = {"task": tag, "full": sysc.full_metric(), "empty": sysc.empty_metric(), "baselines": baselines_for(sysc)}
    if not args.no_agent:
        a = run_circuit_agent(cfg, sysc, out / tag)
        row["maia"] = {**evaluate_set(sysc, a["circuit"]), "metric_calls": a["metric_calls"], "circuit": a["circuit"],
                       "explanation": a["explanation"]}
        k = len(a["circuit"])
        row["baselines_at_maia_size"] = baselines_for(sysc, k=k) if k else {}
    write_json(row, ROOT / cfg["paths"]["outputs"] / "tables" / f"circuit_{tag}.json")
    print({k: (round(v["faithfulness"], 3), v["size"]) for k, v in row["baselines"].items()},
          row.get("maia", {}).get("faithfulness"))


if __name__ == "__main__":
    main()
