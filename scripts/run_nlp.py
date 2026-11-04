#!/usr/bin/env python
"""MAIA-NLP experiments (extension 1).

  sample       sample MLP neurons of the target LM                      -> outputs/nlp/neurons_<model>.json
  maia         MAIA-NLP descriptions of the sampled neurons (--toolset for ablations)
  exemplars    one-shot exemplar-labeling baseline (MILAN-style, text LM)
  synthetic    MAIA-NLP / exemplar baseline on synthetic WordNet neurons (ground truth)
  predictive   predictive evaluation of labels (paper Fig. 3 protocol, text modality)
  2afc         LM-judge 2AFC on synthetic neurons vs ground truth
  bias         MAIA-NLP bias discovery on the LM's behaviour + scoring against BOLD group gaps

python scripts/run_nlp.py sample --model Qwen/Qwen2.5-0.5B --n 50
python scripts/run_nlp.py maia --model Qwen/Qwen2.5-0.5B
python scripts/run_nlp.py predictive --model Qwen/Qwen2.5-0.5B --methods nlp_full nlp_exemplars
python scripts/run_nlp.py bias --model Qwen/Qwen2.5-0.5B
"""
import logging
import random

from _common import ROOT, parser, setup

from maia.backbone import backbone_from_config
from maia.utils import get_device, read_json, write_json

log = logging.getLogger("nlp")


def _slug(m):
    return m.replace("/", "_")


def main():
    p = parser(__doc__)
    p.add_argument("cmd", choices=["sample", "maia", "exemplars", "synthetic", "predictive", "2afc", "bias"])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--kind", default="mlp", choices=["mlp", "attn_head"])
    p.add_argument("--toolset", default="full")
    p.add_argument("--method", default="maia", choices=["maia", "exemplars"])
    p.add_argument("--methods", nargs="*", default=["nlp_full", "nlp_exemplars"])
    args = p.parse_args()
    cfg = setup(args)
    out = ROOT / cfg["paths"]["outputs"]
    device = get_device(cfg["device"])
    from maia.tasks import _nlp_tools, _shared, run_nlp_bias, run_nlp_neuron, run_nlp_synthetic

    if args.cmd == "sample":
        from maia.nlp.system import load_lm

        lm = load_lm(args.model, "cpu")
        rng = random.Random(cfg["seed"])
        n_units = lm.d_mlp if args.kind == "mlp" else lm.n_heads
        units = [{"layer": rng.randrange(lm.n_layers), "unit": rng.randrange(n_units), "kind": args.kind} for _ in range(args.n)]
        write_json(units, out / "nlp" / f"neurons_{_slug(args.model)}.json")
        print(f"sampled {len(units)} units")
        return

    if args.cmd in ("maia", "exemplars"):
        units = read_json(out / "nlp" / f"neurons_{_slug(args.model)}.json")
        tag = f"nlp_{args.toolset}" if args.cmd == "maia" else "nlp_exemplars"
        for u in units:
            d = out / "descriptions" / tag / _slug(args.model) / f"L{u['layer']}_{u['unit']}"
            if (d / "result.json").exists():
                continue
            if args.cmd == "maia":
                r = run_nlp_neuron(cfg, args.model, u["layer"], u["unit"], d, u["kind"], args.toolset)
            else:
                from maia.nlp.evaluation import exemplar_label
                from maia.nlp.system import LLMNeuronSystem, load_lm

                lm = _shared(("lm", args.model), lambda: load_lm(args.model, device))
                s = LLMNeuronSystem(args.model, u["layer"], u["unit"], u["kind"], device, loaded=lm)
                tools = _nlp_tools(cfg, "exemplars_only")
                _, marked = tools.dataset_exemplars(s)
                r = {**u, **exemplar_label(backbone_from_config(cfg, "text_agent"), marked)}
                write_json(r, d / "result.json")
            log.info("%s -> %s", u, r["label"])
        return

    if args.cmd == "synthetic":
        from maia.nlp.evaluation import exemplar_label
        from maia.nlp.synthetic import SyntheticTextNeuronSystem, all_text_synthetic

        tag = f"nlp_synthetic_{args.method}"
        for s in all_text_synthetic():
            d = out / "descriptions" / tag / "synthetic_text" / s.label.replace(" ", "_").replace("|", "_given_")
            if (d / "result.json").exists():
                continue
            if args.method == "maia":
                r = run_nlp_synthetic(cfg, s, d, args.toolset)
            else:
                tools = _nlp_tools(cfg, "exemplars_only")
                _, marked = tools.dataset_exemplars(SyntheticTextNeuronSystem(s))
                r = {**exemplar_label(backbone_from_config(cfg, "text_agent"), marked), "gt": s.gt_description}
                write_json(r, d / "result.json")
            log.info("%s -> %s", s.label, r["label"])
        return

    if args.cmd == "predictive":
        from maia.eval.predictive import aggregate, evaluate_neuron
        from maia.nlp.system import LLMNeuronSystem, load_lm

        gen, sel = backbone_from_config(cfg, "prompt_generator"), backbone_from_config(cfg, "prompt_selector")
        lm = load_lm(args.model, device)
        units = read_json(out / "nlp" / f"neurons_{_slug(args.model)}.json")
        per = []
        for u in units:
            nid = f"L{u['layer']}_{u['unit']}"
            labels = {}
            for m in args.methods:
                f = out / "descriptions" / m / _slug(args.model) / nid / "result.json"
                if f.exists():
                    labels[m] = read_json(f)["label"]
            if len(labels) < 2:
                continue
            s = LLMNeuronSystem(args.model, u["layer"], u["unit"], u["kind"], device, loaded=lm)
            per.append(evaluate_neuron(labels, gen, sel, render=lambda x: x, activate=lambda t: s.neuron(t)[0],
                                       modality="language", out_dir=out / "predictive" / _slug(args.model) / nid))
        summary = aggregate(per, cfg["evaluation"]["n_bootstrap"])
        write_json(summary, out / "tables" / f"nlp_predictive_{_slug(args.model)}.json")
        for m, s in summary.items():
            print(f"{m:>16s} +{s['pos'][0]:.3f} [{s['pos'][1]:.3f},{s['pos'][2]:.3f}]  -{s['neu'][0]:.3f}  n={s['n']}")
        return

    if args.cmd == "2afc":
        import itertools

        from maia.eval.human import llm_2afc
        from maia.nlp.synthetic import all_text_synthetic

        specs = all_text_synthetic()
        key = lambda s: s.label.replace(" ", "_").replace("|", "_given_")  # noqa: E731
        gt = {key(s): s.gt_description for s in specs}
        descs = {}
        for m in args.methods:
            descs[m] = {}
            for s in specs:
                f = out / "descriptions" / m / "synthetic_text" / key(s) / "result.json"
                if f.exists():
                    r = read_json(f)
                    descs[m][key(s)] = r.get("description") or r["label"]
        judge = backbone_from_config(cfg, "judge")
        res = {f"{a} vs {b}": llm_2afc(judge, gt, descs[a], descs[b]) for a, b in itertools.combinations(args.methods, 2)}
        write_json(res, out / "tables" / "nlp_synthetic_2afc.json")
        for k, v in res.items():
            print(f"{k}: {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")
        return

    if args.cmd == "bias":
        from maia.nlp.datasets import BOLD, bold_group_gaps
        from maia.nlp.evaluation import extract_claims, score_bias_claims
        from maia.nlp.system import LLMBehaviorSystem, load_lm

        d = out / "nlp_bias" / _slug(args.model)
        r = read_json(d / "result.json") if (d / "result.json").exists() else run_nlp_bias(cfg, args.model, d)
        bold = BOLD(ROOT / cfg["paths"]["bold_dir"])
        lm = _shared(("lm", args.model), lambda: load_lm(args.model, device))
        sysb = LLMBehaviorSystem(args.model, "sentiment", device=device, loaded=lm)
        gaps_file = d / "bold_gaps.json"
        if gaps_file.exists():
            gaps = read_json(gaps_file)
        else:
            gaps = {dom: bold_group_gaps(sysb, bold, dom) for dom in bold.data}
            write_json(gaps, gaps_file)
        claims = extract_claims(backbone_from_config(cfg, "judge"), r["bias"], {dm: bold.groups(dm) for dm in bold.data})
        score = score_bias_claims(claims, gaps)
        write_json({"claims": claims, "score": score}, out / "tables" / f"nlp_bias_{_slug(args.model)}.json")
        print(score)


if __name__ == "__main__":
    main()
