"""Image synthesis and editing backends for the text2image / edit_images tools."""
from __future__ import annotations

import base64
import io
import logging
from typing import List

import torch
from PIL import Image

log = logging.getLogger(__name__)


def _dtype(device: str):
    return torch.float16 if str(device).startswith("cuda") else torch.float32


class DiffusersText2Image:
    """Stable Diffusion v1.5 (paper default) or any diffusers text-to-image checkpoint."""

    def __init__(self, model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5", device: str = "cuda",
                 steps: int = 50, guidance: float = 7.5, seed: int = 42, batch_size: int = 4, size: int = 512,
                 disable_safety_checker: bool = False):
        from diffusers import AutoPipelineForText2Image

        kw = {"torch_dtype": _dtype(device)}
        if disable_safety_checker:
            kw.update(safety_checker=None, requires_safety_checker=False)
        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kw).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.device, self.steps, self.guidance = device, steps, guidance
        self.seed, self.batch_size, self.size = seed, batch_size, size

    def __call__(self, prompts: List[str]) -> List[Image.Image]:
        out = []
        for s in range(0, len(prompts), self.batch_size):
            chunk = prompts[s : s + self.batch_size]
            gens = [torch.Generator(device="cpu").manual_seed(self.seed + s + j) for j in range(len(chunk))]
            res = self.pipe(chunk, num_inference_steps=self.steps, guidance_scale=self.guidance,
                            generator=gens, height=self.size, width=self.size)
            out.extend(res.images)
        return out


class OpenAIText2Image:
    """DALL-E 3 via the OpenAI API (paper ablation + predictive evaluation). Needs OPENAI_API_KEY."""

    def __init__(self, model: str = "dall-e-3", size: str = "1024x1024", quality: str = "standard"):
        from openai import OpenAI

        self.client, self.model, self.size, self.quality = OpenAI(), model, size, quality

    def __call__(self, prompts: List[str]) -> List[Image.Image]:
        out = []
        for p in prompts:
            r = self.client.images.generate(model=self.model, prompt=p, size=self.size, quality=self.quality,
                                            response_format="b64_json", n=1)
            out.append(Image.open(io.BytesIO(base64.b64decode(r.data[0].b64_json))).convert("RGB"))
        return out


class InstructPix2PixEditor:
    def __init__(self, model_id: str = "timbrooks/instruct-pix2pix", device: str = "cuda", steps: int = 50,
                 guidance: float = 7.5, image_guidance: float = 1.5, seed: int = 42):
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=_dtype(device), safety_checker=None).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps, self.guidance, self.image_guidance, self.seed = steps, guidance, image_guidance, seed

    def __call__(self, images: List[Image.Image], instructions: List[str]) -> List[Image.Image]:
        out = []
        for i, (img, ins) in enumerate(zip(images, instructions)):
            g = torch.Generator(device="cpu").manual_seed(self.seed + i)
            r = self.pipe(ins, image=img.convert("RGB").resize((512, 512)), num_inference_steps=self.steps,
                          guidance_scale=self.guidance, image_guidance_scale=self.image_guidance, generator=g)
            out.append(r.images[0])
        return out


def build_text2image(cfg: dict, device: str):
    kind = cfg.get("backend", "diffusers")
    if kind in ("dalle3", "openai"):
        return OpenAIText2Image(cfg.get("model", "dall-e-3"))
    return DiffusersText2Image(cfg.get("model", "stable-diffusion-v1-5/stable-diffusion-v1-5"), device,
                               steps=cfg.get("steps", 50), guidance=cfg.get("guidance", 7.5),
                               seed=cfg.get("seed", 42), batch_size=cfg.get("batch_size", 4),
                               size=cfg.get("size", 512))


def build_editor(cfg: dict, device: str):
    return InstructPix2PixEditor(cfg.get("model", "timbrooks/instruct-pix2pix"), device,
                                 steps=cfg.get("steps", 50), guidance=cfg.get("guidance", 7.5),
                                 image_guidance=cfg.get("image_guidance", 1.5), seed=cfg.get("seed", 42))
