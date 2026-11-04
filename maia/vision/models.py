"""Vision model registry: loading, preprocessing and layer-name aliases.

Paper Sec. 4.1 models:
  resnet152  supervised ImageNet ResNet-152        layers conv1, layer1..layer4 ("conv.1", "res.1-4")
  dino       DINO ViT-B/16 (self-supervised)       layers mlp.1 .. mlp.11 (MLP hidden units, 3072 each)
  clip-rn50  CLIP visual encoder, ResNet-50        layers layer1..layer4
Extra:
  resnet18-spawrious  ResNet-18 fine-tuned on Spawrious (load from checkpoint)
  vit-b16    torchvision ViT-B/16 ImageNet classifier (used by MAIA-Circuit)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class LoadedModel:
    name: str
    model: nn.Module
    normalize: tuple[tuple, tuple]
    resolution: int
    kind: str  # "conv" or "vit"
    layer_alias: Callable[[str], str]
    n_prefix_tokens: int = 1  # CLS tokens to drop for ViTs
    class_names: list[str] | None = None

    def resolve_layer(self, layer: str) -> nn.Module:
        return self.model.get_submodule(self.layer_alias(layer))


def _resnet_alias(layer: str) -> str:
    l = layer.replace("res.", "layer").replace("conv.1", "conv1").replace(" ", "")
    return {"conv1": "relu"}.get(l, l)  # conv1 -> post BN+ReLU activation


def _clip_alias(layer: str) -> str:
    l = layer.replace("res.", "layer").replace(" ", "")
    return l if l.startswith("visual.") else f"visual.{l}"


def _dino_alias(layer: str) -> str:
    l = layer.lower().replace("mlp.", "").replace("mlp", "").strip()
    if l.isdigit():
        return f"encoder.layer.{int(l)}.intermediate"
    return layer


def _vit_tv_alias(layer: str) -> str:
    l = layer.lower().replace("mlp.", "").strip()
    if l.isdigit():
        return f"encoder.layers.encoder_layer_{int(l)}.mlp.1"  # GELU output (hidden units)
    return layer


def load_vision_model(model_name: str, device: str = "cpu", checkpoint: str | None = None,
                      num_classes: int | None = None, pretrained: bool = True) -> LoadedModel:
    name = model_name.lower()
    if name in ("resnet152", "resnet-152"):
        from torchvision.models import ResNet152_Weights, resnet152

        w = ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet152(weights=w)
        lm = LoadedModel("resnet152", m, (IMAGENET_MEAN, IMAGENET_STD), 224, "conv", _resnet_alias,
                         class_names=list(w.meta["categories"]) if w else None)
    elif name.startswith("resnet18"):
        from torchvision.models import ResNet18_Weights, resnet18

        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if (pretrained and checkpoint is None) else None)
        if num_classes is not None:
            m.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(m.fc.in_features, num_classes))
        if checkpoint:
            sd = torch.load(checkpoint, map_location="cpu")
            m.load_state_dict(sd.get("state_dict", sd))
        lm = LoadedModel(name, m, (IMAGENET_MEAN, IMAGENET_STD), 224, "conv", _resnet_alias)
    elif name in ("dino", "dino-vitb16", "dino_vitb16"):
        from transformers import ViTModel

        m = ViTModel.from_pretrained("facebook/dino-vitb16", add_pooling_layer=False)
        lm = LoadedModel("dino", m, (IMAGENET_MEAN, IMAGENET_STD), 224, "vit", _dino_alias)
    elif name in ("clip", "clip-rn50", "clip_rn50"):
        import open_clip

        m, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
        lm = LoadedModel("clip-rn50", m, (CLIP_MEAN, CLIP_STD), 224, "conv", _clip_alias)
    elif name in ("vit-b16", "vit_b_16"):
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        w = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        m = vit_b_16(weights=w)
        lm = LoadedModel("vit-b16", m, (IMAGENET_MEAN, IMAGENET_STD), 224, "vit", _vit_tv_alias,
                         class_names=list(w.meta["categories"]) if w else None)
    else:
        raise ValueError(f"Unknown vision model '{model_name}'")
    lm.model = lm.model.eval().to(device)
    for p in lm.model.parameters():
        p.requires_grad_(False)
    return lm


def forward_model(lm: LoadedModel, x: torch.Tensor):
    """Run the forward pass appropriate to the model type."""
    if lm.name == "clip-rn50":
        return lm.model.encode_image(x)
    if lm.name == "dino":
        return lm.model(pixel_values=x)
    return lm.model(x)


# Number of units sampled per layer in the paper's Table A3.
TABLE_A3_UNITS = {
    "resnet152": {"conv1": 10, "layer1": 15, "layer2": 20, "layer3": 25, "layer4": 30},
    "dino": {"mlp.1": 5, "mlp.3": 5, "mlp.5": 20, "mlp.7": 20, "mlp.9": 25, "mlp.11": 25},
    "clip-rn50": {"layer1": 10, "layer2": 20, "layer3": 30, "layer4": 40},
}
