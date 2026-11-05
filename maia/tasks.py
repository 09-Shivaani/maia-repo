"""Builders that wire configs -> System + Tools + backbone -> MAIAAgent for each task."""
from __future__ import annotations

import logging
from pathlib import Path

from maia.agent import MAIAAgent
from maia.backbone import backbone_from_config
from maia.prompts import load_prompt
from maia.utils import get_device, set_seed, write_json

log = logging.getLogger(__name__)

# Tool sets for the ablation study (paper Sec. 4.3, Fig. 7)
FULL_TOOLS = {"dataset_exemplars", "text2image", "edit_images", "describe_images", "summarize_images"}
ABLATIONS = {
    "full": FULL_TOOLS,
    "real_only": {"dataset_exemplars", "describe_images", "summarize_images"},   # "Real images only"
    "synthetic_only": {"text2image", "edit_images", "describe_images", "summarize_images"},  # "Synthetic images only"
}

_SHARED: dict = {}


def _shared(key, factory):
    if key not in _SHARED:
        _SHARED[key] = factory()
    return _SHARED[key]


class _LazyShared:
    """Callable that builds a shared model on first use (diffusion pipelines are ~4 GB each in fp32,
    and many agent runs never call text2image / edit_images)."""

    def __init__(self, key, factory):
        self.key, self.factory = key, factory

    def __call__(self, *args, **kwargs):
        return _shared(self.key, self.factory)(*args, **kwargs)


def build_vision_tools(cfg: dict, ablation: str = "full", t2i_cfg: dict | None = None):
    from maia.vision.generation import build_editor, build_text2image
    from maia.vision.tools import VisionTools

    device = get_device(cfg.get("device"))
    allowed = ABLATIONS[ablation]
    t2i_cfg = t2i_cfg or cfg["tools"]["text2image"]
    t2i = editor = None
    if "text2image" in allowed:
        t2i = _LazyShared(("t2i", t2i_cfg.get("backend"), t2i_cfg.get("model")), lambda: build_text2image(t2i_cfg, device))
    if "edit_images" in allowed:
        ecfg = cfg["tools"]["editor"]
        editor = _LazyShared(("edit", ecfg.get("model")), lambda: build_editor(ecfg, device))
    return VisionTools(
        describer=backbone_from_config(cfg, "describer"),
        text2image=t2i,
        editor=editor,
        exemplar_cache=cfg["paths"]["exemplar_cache"],
        exemplar_images=cfg["paths"]["cc3m_images"],
        device=device,
        k=cfg["tools"].get("exemplars_k", 15),
        allowed=allowed,
    )


def _agent(cfg, system, tools, user_prompt, out_dir, final_tags, required_tag, api="vision_api", role="vision_agent"):
    bb = backbone_from_config(cfg, role)
    proxy = None
    if not bb.supports_vision and cfg["agent"].get("vision_proxy"):
        proxy = backbone_from_config(cfg, cfg["agent"]["vision_proxy"])
    return MAIAAgent(bb, system, tools, load_prompt(api), user_prompt, final_tags=final_tags,
                     required_tag=required_tag, max_rounds=cfg["agent"]["max_rounds"],
                     exec_timeout=cfg["agent"]["exec_timeout"], vision_proxy=proxy, log_dir=out_dir)


# ---------------------------------------------------------------------------
def run_neuron_description(cfg: dict, model_name: str, layer: str, unit: int, out_dir: str | Path,
                           ablation: str = "full", t2i_cfg: dict | None = None) -> dict:
    from maia.vision.models import load_vision_model
    from maia.vision.system import VisionNeuronSystem

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device"))
    lm = _shared(("vision", model_name), lambda: load_vision_model(model_name, device))
    system = VisionNeuronSystem(unit, layer, model_name, device, loaded=lm)
    tools = build_vision_tools(cfg, ablation, t2i_cfg)
    agent = _agent(cfg, system, tools, load_prompt("neuron_description"), out_dir, ("LABEL", "DESCRIPTION"), "LABEL")
    res = agent.run()
    out = {"model": model_name, "layer": layer, "unit": unit, "ablation": ablation,
           "label": res.fields.get("LABEL", ""), "description": res.fields.get("DESCRIPTION", ""),
           "rounds": res.rounds, "finished": res.finished}
    write_json(out, Path(out_dir) / "result.json")
    return out


def run_synthetic_neuron(cfg: dict, spec, out_dir: str | Path) -> dict:
    from maia.vision.synthetic import GroundedSAM, SyntheticNeuronSystem

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device"))
    seg = _shared("grounded_sam", lambda: GroundedSAM(device))
    system = SyntheticNeuronSystem(spec, seg, device, exemplar_images=cfg["paths"]["cc3m_images"])
    tools = build_vision_tools(cfg)
    agent = _agent(cfg, system, tools, load_prompt("neuron_description"), out_dir, ("LABEL", "DESCRIPTION"), "LABEL")
    res = agent.run()
    out = {"synthetic": spec.label, "kind": spec.kind, "gt": spec.gt_description,
           "label": res.fields.get("LABEL", ""), "description": res.fields.get("DESCRIPTION", ""),
           "rounds": res.rounds, "finished": res.finished}
    write_json(out, Path(out_dir) / "result.json")
    return out


def run_bias_detection(cfg: dict, class_label: str, out_dir: str | Path, model_name: str = "resnet152") -> dict:
    from maia.vision.models import load_vision_model
    from maia.vision.system import ClassifierSystem

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device"))
    lm = _shared(("vision", model_name), lambda: load_vision_model(model_name, device))
    system = ClassifierSystem(class_label, model_name, device, loaded=lm)
    # Paper Sec 5.2: MAIA generates synthetic inputs; no ImageNet exemplars are needed.
    tools = build_vision_tools(cfg, "synthetic_only")
    agent = _agent(cfg, system, tools, load_prompt("bias_detection", class_label=class_label), out_dir, ("BIAS",), "BIAS")
    res = agent.run()
    out = {"class_label": class_label, "bias": res.fields.get("BIAS", ""), "rounds": res.rounds, "finished": res.finished}
    write_json(out, Path(out_dir) / "result.json")
    return out


def run_spurious_unit(cfg: dict, lm, unit: int, out_dir: str | Path, exemplar_images: str,
                      activation_floor: float | None = None) -> dict:
    """Classify one final-layer unit of the Spawrious ResNet-18 as SELECTIVE / SPURIOUS (paper Sec. 5.1)."""
    from maia.vision.system import VisionNeuronSystem

    set_seed(cfg["seed"])
    system = VisionNeuronSystem(unit, "layer4", lm.name, get_device(cfg.get("device")),
                                loaded=lm, activation_floor=activation_floor)
    tools = build_vision_tools(cfg)
    tools.exemplar_images = exemplar_images  # unbalanced validation set only
    tools.exemplar_cache = str(Path(cfg["paths"]["outputs"]) / "spurious" / "exemplars")
    agent = _agent(cfg, system, tools, load_prompt("spurious_removal"), out_dir, ("LABEL", "REASONING"), "LABEL")
    res = agent.run()
    label = res.fields.get("LABEL", "SPURIOUS").upper()
    out = {"unit": unit, "label": "SELECTIVE" if "SELECTIVE" in label else "SPURIOUS",
           "reasoning": res.fields.get("REASONING", ""), "rounds": res.rounds}
    write_json(out, Path(out_dir) / "result.json")
    return out


# ===========================================================================
# Extension 1: MAIA-NLP
# ===========================================================================
NLP_TOOLSETS = {
    "full": {"dataset_exemplars", "generate_texts", "text_contrastive", "semantic_perturbation",
             "token_activation_map", "describe_texts", "summarize_texts", "bold_prompts", "stereoset_examples"},
    "no_counterfactual": {"dataset_exemplars", "generate_texts", "token_activation_map", "describe_texts",
                          "summarize_texts", "bold_prompts", "stereoset_examples"},
    "exemplars_only": {"dataset_exemplars", "token_activation_map", "describe_texts", "summarize_texts"},
    "synthetic_only": {"generate_texts", "text_contrastive", "semantic_perturbation", "token_activation_map",
                       "describe_texts", "summarize_texts"},
}


def _nlp_tools(cfg, toolset="full", with_bias_data=False):
    from maia.nlp.exemplars import load_corpus
    from maia.nlp.tools import NLPTools

    corpus = _shared(("corpus", cfg["paths"]["text_corpus"]),
                     lambda: load_corpus(cfg["paths"]["text_corpus"], cfg.get("nlp", {}).get("corpus_size", 200000)))
    bold = stereo = None
    if with_bias_data:
        from maia.nlp.datasets import BOLD, StereoSet

        bold = _shared("bold", lambda: BOLD(cfg["paths"]["bold_dir"]))
        stereo = _shared("stereoset", lambda: StereoSet(cfg["paths"]["stereoset"]))
    return NLPTools(writer=backbone_from_config(cfg, "fast_text"), describer=backbone_from_config(cfg, "text_agent"),
                    corpus=corpus, exemplar_cache=str(Path(cfg["paths"]["outputs"]) / "text_exemplars"),
                    k=cfg["tools"].get("exemplars_k", 15), allowed=NLP_TOOLSETS[toolset], bold=bold, stereoset=stereo)


def run_nlp_neuron(cfg: dict, model_name: str, layer: int, unit: int, out_dir, kind: str = "mlp",
                   toolset: str = "full") -> dict:
    from maia.nlp.system import LLMNeuronSystem, load_lm

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device"))
    lm = _shared(("lm", model_name), lambda: load_lm(model_name, device))
    system = LLMNeuronSystem(model_name, layer, unit, kind, device, loaded=lm)
    tools = _nlp_tools(cfg, toolset)
    agent = _agent(cfg, system, tools, load_prompt("nlp_neuron_description"), out_dir, ("LABEL", "DESCRIPTION"),
                   "LABEL", api="nlp_api", role="text_agent")
    res = agent.run()
    out = {"model": model_name, "layer": layer, "unit": unit, "kind": kind, "toolset": toolset,
           "label": res.fields.get("LABEL", ""), "description": res.fields.get("DESCRIPTION", ""), "rounds": res.rounds}
    write_json(out, Path(out_dir) / "result.json")
    return out


def run_nlp_synthetic(cfg: dict, spec, out_dir, toolset: str = "full") -> dict:
    from maia.nlp.synthetic import SyntheticTextNeuronSystem

    set_seed(cfg["seed"])
    system = SyntheticTextNeuronSystem(spec)
    tools = _nlp_tools(cfg, toolset)
    agent = _agent(cfg, system, tools, load_prompt("nlp_neuron_description"), out_dir, ("LABEL", "DESCRIPTION"),
                   "LABEL", api="nlp_api", role="text_agent")
    res = agent.run()
    out = {"synthetic": spec.label, "kind": spec.kind, "gt": spec.gt_description, "toolset": toolset,
           "label": res.fields.get("LABEL", ""), "description": res.fields.get("DESCRIPTION", ""), "rounds": res.rounds}
    write_json(out, Path(out_dir) / "result.json")
    return out


def run_nlp_bias(cfg: dict, model_name: str, out_dir) -> dict:
    from maia.nlp.system import LLMBehaviorSystem, load_lm

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device"))
    lm = _shared(("lm", model_name), lambda: load_lm(model_name, device))
    system = LLMBehaviorSystem(model_name, "sentiment", device=device, loaded=lm,
                               description="VADER sentiment of the model's greedy continuation")
    tools = _nlp_tools(cfg, "full", with_bias_data=True)
    prompt = load_prompt("nlp_bias", behaviour="the VADER sentiment (-1 to 1) of the model's greedy 30-token continuation")
    agent = _agent(cfg, system, tools, prompt, out_dir, ("BIAS",), "BIAS", api="nlp_api", role="text_agent")
    res = agent.run()
    out = {"model": model_name, "bias": res.fields.get("BIAS", ""), "rounds": res.rounds}
    write_json(out, Path(out_dir) / "result.json")
    return out


# ===========================================================================
# Extension 2: MAIA-Circuit
# ===========================================================================
def run_circuit_agent(cfg: dict, system, out_dir, toolset: set | None = None) -> dict:
    from maia.circuit.tools import CircuitTools

    set_seed(cfg["seed"])
    tools = CircuitTools(system, allowed=toolset)
    prompt = load_prompt("circuit_discovery", task=system.task_description, n_nodes=len(system.nodes))
    agent = _agent(cfg, system, tools, prompt, out_dir, ("CIRCUIT", "EXPLANATION"), "CIRCUIT",
                   api="circuit_api", role="text_agent")
    res = agent.run()
    raw = res.fields.get("CIRCUIT", "")
    nodes = [n.strip().strip("`'\"") for n in raw.replace("\n", ",").split(",")]
    nodes = [n for n in nodes if n in set(system.nodes)]
    out = {"circuit": nodes, "explanation": res.fields.get("EXPLANATION", ""), "rounds": res.rounds,
           "metric_calls": tools.n_metric_calls}
    write_json(out, Path(out_dir) / "result.json")
    return out
