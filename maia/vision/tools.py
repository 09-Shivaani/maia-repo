"""Tools class for vision experiments (paper Sec. 3.2, App. A)."""
from __future__ import annotations

import logging
from typing import List, Tuple, Union

from PIL import Image

from maia.agent import ExperimentLogMixin
from maia.backbone import Backbone
from maia.utils import pil_to_base64, to_pil
from maia.vision.exemplars import cache_path, compute_layer_exemplars, list_images, load_exemplars

log = logging.getLogger(__name__)

DESCRIBE_PROMPT = (
    "The image may be partly darkened; one region is left bright and outlined in red. Describe the "
    "content of the highlighted region in one short sentence (objects, parts, textures, colours). "
    "If nothing is highlighted, say 'no highlighted region'."
)
SUMMARIZE_PROMPT = (
    "Each image has a highlighted (bright, red-outlined) region; the rest is darkened. In one or two "
    "sentences, describe the visual concept that is common to the highlighted regions across ALL images. "
    "Be specific (objects, parts, textures, colours, context)."
)


class VisionTools(ExperimentLogMixin):
    def __init__(self, describer: Backbone, text2image=None, editor=None, exemplar_cache: str | None = None,
                 exemplar_images: str | None = None, device: str = "cuda", k: int = 15,
                 allowed: set[str] | None = None, max_log_images: int = 60):
        """
        describer       VLM used by describe_images / summarize_images (fresh context every call).
        text2image      callable(list[str]) -> list[PIL]; None disables the tool (ablation "real only").
        editor          callable(list[PIL], list[str]) -> list[PIL]; None disables edit_images.
        exemplar_cache  directory with precomputed exemplars (scripts/compute_exemplars.py).
        exemplar_images folder of real images to compute exemplars on the fly if not cached.
        allowed         names of enabled tools (for the tool-ablation study, paper Sec. 4.3).
        """
        self._init_log()
        self.describer = describer
        self._t2i, self._editor = text2image, editor
        self.exemplar_cache, self.exemplar_images = exemplar_cache, exemplar_images
        self.device, self.k = device, k
        self.allowed = allowed
        self.max_log_images = max_log_images

    def _check(self, name: str):
        if self.allowed is not None and name not in self.allowed:
            raise RuntimeError(f"The tool '{name}' is not available in this experiment. Use the other tools.")

    # ------------------------------------------------------------------
    def dataset_exemplars(self, system) -> Tuple[List[float], List[str]]:
        self._check("dataset_exemplars")
        if getattr(system, "synthetic", False):
            return system.dataset_exemplars(self.k)
        lm = system.lm
        path = cache_path(self.exemplar_cache, lm.name, system.layer) if self.exemplar_cache else None
        if path is not None and path.exists():
            acts, idx, paths = load_exemplars(path)
        else:
            if not self.exemplar_images:
                raise RuntimeError("No exemplar cache and no exemplar image folder configured.")
            log.warning("Exemplar cache missing for %s/%s - computing (slow).", lm.name, system.layer)
            paths = list_images(self.exemplar_images)
            res = compute_layer_exemplars(lm, system.layer, paths, k=self.k, device=system.device, out_path=path)
            acts, idx = res["acts"], res["idx"]
        sel = [paths[i] for i in idx[system.neuron_num][: self.k]]
        images = [Image.open(p).convert("RGB") for p in sel]
        a, masks = system.neuron(images)
        vals = [float(x) for x in a]
        self.activation_threshold = sum(vals) / len(vals) if vals else None
        return vals, masks

    def text2image(self, prompt_list: Union[str, List[str]]) -> List[Image.Image]:
        self._check("text2image")
        if self._t2i is None:
            raise RuntimeError("text2image is disabled in this experiment.")
        if isinstance(prompt_list, str):
            prompt_list = [prompt_list]
        return self._t2i(list(prompt_list))

    def edit_images(self, prompt_list_org_image: List[str], editing_instructions_list: List[str]):
        self._check("edit_images")
        if self._editor is None or self._t2i is None:
            raise RuntimeError("edit_images is disabled in this experiment.")
        if isinstance(prompt_list_org_image, str):
            prompt_list_org_image = [prompt_list_org_image]
        if isinstance(editing_instructions_list, str):
            editing_instructions_list = [editing_instructions_list]
        if len(prompt_list_org_image) != len(editing_instructions_list):
            raise ValueError("prompt_list_org_image and editing_instructions_list must have the same length")
        originals = self._t2i(list(prompt_list_org_image))
        edited = self._editor(originals, list(editing_instructions_list))
        images, titles = [], []
        for o, e, p, ins in zip(originals, edited, prompt_list_org_image, editing_instructions_list):
            images += [o, e]
            titles += [p, ins]
        return images, titles

    def describe_images(self, image_list: List, image_title: List[str] | None = None) -> str:
        self._check("describe_images")
        image_title = image_title or [f"image {i}" for i in range(len(image_list))]
        out = []
        for img, t in zip(image_list, image_title):
            b64 = img if isinstance(img, str) else pil_to_base64(to_pil(img))
            d = self.describer.complete(DESCRIBE_PROMPT, images=[b64], max_tokens=150, temperature=0.0)
            out.append(f'"{t}", highlighted regions: {d.strip()}')
        return "\n".join(out)

    def summarize_images(self, image_list: List) -> str:
        self._check("summarize_images")
        b64s = [i if isinstance(i, str) else pil_to_base64(to_pil(i)) for i in image_list]
        return self.describer.complete(SUMMARIZE_PROMPT, images=b64s, max_tokens=250, temperature=0.0).strip()

    def log_experiment(self, activation_list, image_list, image_titles=None, image_textual_information=None):
        if not isinstance(activation_list, (list, tuple)):
            activation_list = [activation_list]
        if not isinstance(image_list, (list, tuple)):
            image_list = [image_list]
        image_titles = list(image_titles) if image_titles is not None else [f"image {i}" for i in range(len(image_list))]
        if len(image_titles) < len(image_list):
            image_titles += [f"image {i}" for i in range(len(image_titles), len(image_list))]
        n_shown = 0
        for act, img, title in zip(activation_list, image_list, image_titles):
            self._log_text(f'"{title}", activation: {act}')
            if n_shown < self.max_log_images:
                self._log_image(img if isinstance(img, str) else pil_to_base64(to_pil(img)))
                n_shown += 1
            self.results_list.append({title: {"activation": act}})
        if image_textual_information:
            info = image_textual_information
            self._log_text(info if isinstance(info, str) else "\n".join(map(str, info)))
        if self.activation_threshold is not None and activation_list:
            try:
                if max(float(a) for a in activation_list) < self.activation_threshold:
                    self._log_text(
                        f"Note: the highest activation in this experiment is below the average dataset exemplar "
                        f"activation ({self.activation_threshold:.2f}). Keep experimenting to find inputs that "
                        f"activate the unit as strongly as the exemplars."
                    )
            except (TypeError, ValueError):
                pass

    save_experiment_log = log_experiment
