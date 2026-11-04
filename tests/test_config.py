from maia.backbone import BackboneConfig
from maia.config import load_config
from maia.prompts import load_prompt
from maia.vision.synthetic import all_synthetic_neurons


def test_config_overrides_and_aliases():
    cfg = load_config("nemotron_agent.yaml", ["backbones.describer.model=qwen2.5vl:32b", "seed=7"])
    assert cfg["seed"] == 7
    assert cfg["backbones"]["vision_agent"] == "text_agent"
    assert cfg["backbones"]["describer"]["model"] == "qwen2.5vl:32b"
    assert BackboneConfig.from_dict(cfg["backbones"]["text_agent"]).provider == "nvidia"


def test_no_api_keys_in_config():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for p in list(root.rglob("*.py")) + list(root.rglob("*.yaml")) + list(root.rglob("*.md")):
        assert not re.search(r"nvapi-[A-Za-z0-9_-]{20,}", p.read_text(errors="ignore")), p


def test_prompts_and_table_a6():
    assert "{class_label}" not in load_prompt("bias_detection", class_label="tench")
    specs = all_synthetic_neurons()
    assert len(specs) == 87  # Table A6 lists 45 + 24 + 18 = 87 rows (the text says 85)
    assert {s.kind for s in specs} == {"mono", "or", "cond"}
