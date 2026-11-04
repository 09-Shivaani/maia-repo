"""Expert human labeling with the MAIA API (paper App. C3). Use from JupyterLab:

    import sys; sys.path.insert(0, "scripts")
    from human_session import HumanSession
    s = HumanSession("resnet152", "layer4", 122)
    s.run('''
    def run_experiment(system, tools):
        acts, imgs = tools.dataset_exemplars(system)
        tools.log_experiment(acts, imgs, [f"exemplar {i}" for i in range(len(acts))])
    ''')
    s.submit(description="bow ties in formal attire", label="bow ties")

Labels are written to outputs/descriptions/human/<model>/<layer>_<unit>/result.json, where the
evaluation scripts pick them up as method "human".
"""
from __future__ import annotations

import time

from _common import ROOT

from maia.agent import execute_code
from maia.config import load_config
from maia.tasks import _shared, build_vision_tools
from maia.utils import base64_to_pil, get_device, write_json


class HumanSession:
    def __init__(self, model: str, layer: str, unit: int, config: str | None = None, synthetic_spec=None):
        self.cfg = load_config(config)
        dev = get_device(self.cfg["device"])
        if synthetic_spec is not None:
            from maia.vision.synthetic import GroundedSAM, SyntheticNeuronSystem

            seg = _shared("grounded_sam", lambda: GroundedSAM(dev))
            self.system = SyntheticNeuronSystem(synthetic_spec, seg, dev, exemplar_images=str(ROOT / self.cfg["paths"]["cc3m_images"]))
            self.out = ROOT / self.cfg["paths"]["outputs"] / "descriptions" / "synthetic_human" / "synthetic" / \
                synthetic_spec.label.replace(" ", "_").replace("|", "_given_")
        else:
            from maia.vision.models import load_vision_model
            from maia.vision.system import VisionNeuronSystem

            lm = _shared(("vision", model), lambda: load_vision_model(model, dev))
            self.system = VisionNeuronSystem(unit, layer, model, dev, loaded=lm)
            self.out = ROOT / self.cfg["paths"]["outputs"] / "descriptions" / "human" / lm.name / f"{layer}_{unit}"
        self.tools = build_vision_tools(self.cfg)
        self.history, self.t0 = [], time.time()

    def run(self, code: str):
        res = execute_code(code, self.system, self.tools)
        parts = self.tools.flush_log()
        self.history.append(code)
        try:
            from IPython.display import display
        except ImportError:  # pragma: no cover
            display = print
        for p in parts:
            if p["type"] == "text":
                print(p["text"])
            else:
                display(base64_to_pil(p["image_url"]["url"]))
        if not res.ok:
            print(res.error)

    def submit(self, description: str, label: str):
        write_json({"label": label, "description": description, "interactions": len(self.history),
                    "minutes": (time.time() - self.t0) / 60, "code": self.history}, self.out / "result.json")
        print(f"saved to {self.out}")
