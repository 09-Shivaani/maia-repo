"""Synthetic vision neurons with ground-truth selectivity (paper Sec. 4.2, App. C4).

A synthetic neuron is an open-set concept detector: Grounding-DINO proposes boxes for the concept
text (box threshold 0.3, text threshold 0.25) and SAM segments them. The activation is the detector
confidence; the mask is the (dilated) segmentation.

  monosemantic  "stripes"           -> fires on stripes
  polysemantic  ("train", "instrument") A OR B -> fires on either (mean score if both present)
  conditional   ("dog", "leash")    A|B -> fires on A only when B is also present
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter

from maia.utils import get_device, pil_to_base64, to_pil
from maia.vision.system import preprocess_pil, render_mask

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table A6: the 85 synthetic neurons (concepts drawn from MILANNOTATIONS)
# ---------------------------------------------------------------------------
MONOSEMANTIC = [
    "arch", "bird", "blue", "boat", "brick", "bridge", "bug", "building", "button", "car window", "circle",
    "dog", "eyes", "feathers", "flame", "frog", "grass", "hair", "hands", "handle", "hat", "jeans", "jigsaw",
    "legs", "light", "neck", "orange", "paws", "pencil", "pizza", "roof", "shirt", "shoes", "sky", "snake",
    "spiral", "stripes", "sunglasses", "tail", "text", "tires", "tractor", "vehicle", "wing", "yarn",
]
POLYSEMANTIC = [
    ("animal", "door"), ("animal", "ship"), ("baby", "dog"), ("bird", "dog"), ("blue", "yellow"),
    ("bookshelf", "building"), ("cup", "road"), ("dog", "car"), ("dog", "horse"), ("dog", "instrument"),
    ("fire", "fur"), ("firework", "whisker"), ("hand", "ear"), ("necklace", "flower"), ("people", "building"),
    ("people", "wood"), ("red", "purple"), ("shoe", "boat"), ("sink", "pool"), ("skirt", "water"),
    ("stairs", "fruit"), ("temple", "playground"), ("truck", "train"), ("window", "wheel"),
]
CONDITIONAL = [
    ("ball", "hand"), ("beach", "people"), ("bird", "tree"), ("bridge", "sky"), ("building", "sky"),
    ("cup", "handle"), ("dog", "leash"), ("fence", "animal"), ("fish", "water"), ("grass", "dog"),
    ("horse", "grass"), ("instrument", "hand"), ("skyline", "water"), ("sky", "bird"), ("suit", "tie"),
    ("tent", "mountain"), ("water", "blue"), ("wheel", "racecar"),
]


@dataclass(frozen=True)
class SyntheticSpec:
    kind: str  # mono | or | cond
    concepts: tuple

    @property
    def label(self) -> str:
        if self.kind == "mono":
            return self.concepts[0]
        if self.kind == "or":
            return f"{self.concepts[0]} OR {self.concepts[1]}"
        return f"{self.concepts[0]}|{self.concepts[1]}"

    @property
    def gt_description(self) -> str:
        if self.kind == "cond":
            return f"{self.concepts[0]} (only when {self.concepts[1]} is also present)"
        return self.label


def all_synthetic_neurons() -> List[SyntheticSpec]:
    return ([SyntheticSpec("mono", (c,)) for c in MONOSEMANTIC]
            + [SyntheticSpec("or", p) for p in POLYSEMANTIC]
            + [SyntheticSpec("cond", p) for p in CONDITIONAL])


class GroundedSAM:
    """Text-guided segmentation: Grounding-DINO boxes -> SAM masks (transformers implementations)."""

    def __init__(self, device: str = "cuda", dino_id: str = "IDEA-Research/grounding-dino-base",
                 sam_id: str = "facebook/sam-vit-huge", box_threshold: float = 0.3, text_threshold: float = 0.25):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

        self.device = device
        self.dino_proc = AutoProcessor.from_pretrained(dino_id)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device).eval()
        self.sam_proc = SamProcessor.from_pretrained(sam_id)
        self.sam = SamModel.from_pretrained(sam_id).to(device).eval()
        self.box_threshold, self.text_threshold = box_threshold, text_threshold

    @torch.no_grad()
    def detect(self, img: Image.Image, concept: str) -> Tuple[float, np.ndarray | None]:
        text = concept.lower().strip().rstrip(".") + "."
        inputs = self.dino_proc(images=img, text=text, return_tensors="pt").to(self.device)
        out = self.dino(**inputs)
        post = self.dino_proc.post_process_grounded_object_detection
        try:
            r = post(out, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
                     target_sizes=[img.size[::-1]])[0]
        except TypeError:  # older transformers
            r = post(out, inputs.input_ids, box_threshold=self.box_threshold, text_threshold=self.text_threshold,
                     target_sizes=[img.size[::-1]])[0]
        if len(r["scores"]) == 0:
            return 0.0, None
        boxes = r["boxes"].cpu().tolist()
        score = float(r["scores"].max())
        sin = self.sam_proc(img, input_boxes=[boxes], return_tensors="pt").to(self.device)
        so = self.sam(**sin, multimask_output=False)
        masks = self.sam_proc.image_processor.post_process_masks(
            so.pred_masks.cpu(), sin["original_sizes"].cpu(), sin["reshaped_input_sizes"].cpu())[0]
        m = masks.any(dim=0).squeeze().numpy().astype(bool)
        return score, m


def _dilate(m: np.ndarray, k: int = 15) -> np.ndarray:
    return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(k))) > 127


class SyntheticNeuronSystem:
    synthetic = True

    def __init__(self, spec: SyntheticSpec, segmenter: GroundedSAM | None = None, device: str | int = "auto",
                 exemplar_images: str | None = None, pool_size: int = 3000, resolution: int = 224, seed: int = 0):
        self.spec = spec
        self.device = get_device(device)
        self.seg = segmenter or GroundedSAM(self.device)
        self.neuron_num, self.layer, self.model_name = 0, spec.kind, f"synthetic:{spec.label}"
        self.exemplar_images, self.pool_size, self.resolution, self.seed = exemplar_images, pool_size, resolution, seed

    def _respond(self, img: Image.Image) -> Tuple[float, np.ndarray | None]:
        c = self.spec.concepts
        if self.spec.kind == "mono":
            return self.seg.detect(img, c[0])
        sa, ma = self.seg.detect(img, c[0])
        sb, mb = self.seg.detect(img, c[1])
        if self.spec.kind == "or":
            if ma is not None and mb is not None:
                return (sa + sb) / 2, ma | mb
            return (sa, ma) if ma is not None else (sb, mb)
        # conditional A|B
        if ma is not None and mb is not None:
            return sa, ma
        return 0.0, None

    def neuron(self, image_list) -> Tuple[List[float], List[str]]:
        if not isinstance(image_list, (list, tuple)):
            image_list = [image_list]
        acts, masks = [], []
        for im in image_list:
            img = preprocess_pil(to_pil(im), self.resolution)
            s, m = self._respond(img)
            acts.append(round(s, 2))
            amap = _dilate(m).astype(np.float32) if m is not None else None
            masks.append(pil_to_base64(render_mask(img, amap, 0.5)))
        return acts, masks

    def dataset_exemplars(self, k: int = 15) -> Tuple[List[float], List[str]]:
        """Top-k CC3M images. To keep cost manageable, the detector runs on a candidate pool:
        images whose CC3M caption mentions a concept word, topped up with random images."""
        from maia.vision.exemplars import list_images

        paths = list_images(self.exemplar_images)
        rng = random.Random(self.seed)
        words = {w for c in self.spec.concepts for w in c.lower().split()}
        hits = []
        for p in paths:
            cap = Path(p).with_suffix(".txt")
            if cap.exists() and words & set(cap.read_text(errors="ignore").lower().replace(",", " ").split()):
                hits.append(p)
        rng.shuffle(hits)
        pool = hits[: self.pool_size // 2]
        rest = [p for p in paths if p not in set(pool)]
        pool += rng.sample(rest, min(len(rest), self.pool_size - len(pool)))
        scored = []
        for p in pool:
            img = preprocess_pil(Image.open(p).convert("RGB"), self.resolution)
            s, _ = self._respond(img)
            scored.append((s, p))
        scored.sort(reverse=True)
        top = [Image.open(p).convert("RGB") for _, p in scored[:k]]
        return self.neuron(top)
