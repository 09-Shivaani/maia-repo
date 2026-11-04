"""System classes: instrument a vision model so a single unit is callable (paper Sec. 3.1)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from maia.utils import get_device, pil_to_base64, to_pil
from maia.vision.models import LoadedModel, forward_model, load_vision_model


def preprocess_pil(img: Image.Image, resolution: int) -> Image.Image:
    """Resize shorter side to 256/224*res and center-crop to res (standard ImageNet eval transform)."""
    img = to_pil(img)
    short = int(round(resolution * 256 / 224))
    w, h = img.size
    s = short / min(w, h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - resolution) // 2, (h - resolution) // 2
    return img.crop((l, t, l + resolution, t + resolution))


def pil_batch_to_tensor(imgs: Sequence[Image.Image], mean, std, device) -> torch.Tensor:
    arr = np.stack([np.asarray(i, dtype=np.float32) / 255.0 for i in imgs])  # B,H,W,3
    x = torch.from_numpy(arr).permute(0, 3, 1, 2)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return ((x - m) / s).to(device)


def render_mask(img: Image.Image, amap: np.ndarray | None, threshold: float | None,
                quantile: float = 0.95, dim: float = 0.25) -> Image.Image:
    """Darken everything except the region where the (upsampled) activation map exceeds the
    threshold, and outline the region in red. ``amap=None`` -> fully darkened (unit inactive)."""
    base = np.asarray(img, dtype=np.float32)
    if amap is None:
        return Image.fromarray((base * dim).astype(np.uint8))
    thr = threshold if threshold is not None else float(np.quantile(amap, quantile))
    mask = amap >= thr
    if not mask.any():
        return Image.fromarray((base * dim).astype(np.uint8))
    m = Image.fromarray((mask * 255).astype(np.uint8))
    edge = np.asarray(m.filter(ImageFilter.MaxFilter(5)), dtype=bool) & ~np.asarray(m.filter(ImageFilter.MinFilter(3)), dtype=bool)
    out = base * np.where(mask[..., None], 1.0, dim)
    out[edge] = (255, 0, 0)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


class VisionNeuronSystem:
    """A single unit of a vision model.

    activation(image) = max over spatial positions (or patch tokens) of the unit's response.
    """

    def __init__(self, neuron_num: int, layer: str, model_name: str, device: str | int = "auto",
                 loaded: LoadedModel | None = None, batch_size: int = 32, mask_threshold: float | None = None,
                 activation_floor: float | None = 0.0):
        self.neuron_num = int(neuron_num)
        self.layer = layer
        self.model_name = model_name
        self.device = get_device(device)
        self.lm = loaded or load_vision_model(model_name, self.device)
        self.model = self.lm.model
        self.batch_size = batch_size
        # Dataset-level threshold on the activation map (set from exemplars); None -> per-image quantile
        self.mask_threshold = mask_threshold
        # Below this max activation the unit is considered inactive (no highlighted region).
        self.activation_floor = activation_floor
        self._module = self.lm.resolve_layer(layer)

    # ------------------------------------------------------------------
    def _maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return activation maps of the unit, shape [B, h, w]."""
        store = {}

        def hook(_m, _i, out):
            store["out"] = out[0] if isinstance(out, tuple) else out

        h = self._module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                forward_model(self.lm, x)
        finally:
            h.remove()
        out = store["out"]
        if self.lm.kind == "conv" or out.ndim == 4:
            return out[:, self.neuron_num].float()
        tok = out[:, self.lm.n_prefix_tokens :, self.neuron_num].float()  # [B, T]
        g = int(round(tok.shape[1] ** 0.5))
        return tok[:, : g * g].reshape(-1, g, g)

    def activation_maps(self, image_list) -> Tuple[List[Image.Image], np.ndarray, List[np.ndarray]]:
        imgs = [preprocess_pil(i, self.lm.resolution) for i in image_list]
        acts, maps = [], []
        for s in range(0, len(imgs), self.batch_size):
            batch = imgs[s : s + self.batch_size]
            x = pil_batch_to_tensor(batch, *self.lm.normalize, self.device)
            m = self._maps(x)
            acts.append(m.flatten(1).max(1).values.cpu())
            up = F.interpolate(m[:, None], size=(self.lm.resolution,) * 2, mode="bilinear", align_corners=False)[:, 0]
            maps.extend(list(up.cpu().numpy()))
        return imgs, torch.cat(acts).numpy(), maps

    def neuron(self, image_list) -> Tuple[List[float], List[str]]:
        if not isinstance(image_list, (list, tuple)):
            image_list = [image_list]
        imgs, acts, maps = self.activation_maps(image_list)
        masks = []
        for img, a, amap in zip(imgs, acts, maps):
            inactive = self.activation_floor is not None and a <= self.activation_floor
            masks.append(pil_to_base64(render_mask(img, None if inactive else amap, self.mask_threshold)))
        return [round(float(a), 2) for a in acts], masks

    call_neuron = neuron  # name used in some paper figures


class ClassifierSystem:
    """Instruments one output class of an image classifier (paper Sec. 5.2, bias identification).

    ``neuron`` returns the softmax probability of the class and a class-activation map (CAM)
    computed from the last convolutional block and the classifier weights.
    """

    def __init__(self, class_label: str, model_name: str = "resnet152", device: str | int = "auto",
                 loaded: LoadedModel | None = None, batch_size: int = 32):
        self.device = get_device(device)
        self.lm = loaded or load_vision_model(model_name, self.device)
        self.model = self.lm.model
        names = [c.lower() for c in (self.lm.class_names or [])]
        if class_label.lower() not in names:
            raise ValueError(f"class '{class_label}' not among the model's {len(names)} classes")
        self.class_label = class_label
        self.neuron_num = names.index(class_label.lower())
        self.layer = "fc"
        self.model_name = model_name
        self.batch_size = batch_size

    def neuron(self, image_list) -> Tuple[List[float], List[str]]:
        if not isinstance(image_list, (list, tuple)):
            image_list = [image_list]
        imgs = [preprocess_pil(i, self.lm.resolution) for i in image_list]
        probs, masks = [], []
        feats = {}
        h = self.model.layer4.register_forward_hook(lambda m, i, o: feats.__setitem__("f", o))
        try:
            for s in range(0, len(imgs), self.batch_size):
                batch = imgs[s : s + self.batch_size]
                x = pil_batch_to_tensor(batch, *self.lm.normalize, self.device)
                with torch.no_grad():
                    logits = self.model(x)
                p = logits.softmax(-1)[:, self.neuron_num].cpu().numpy()
                w = self.model.fc.weight[self.neuron_num]  # [C]
                cam = torch.einsum("bchw,c->bhw", feats["f"], w)
                cam = F.interpolate(cam[:, None], size=(self.lm.resolution,) * 2, mode="bilinear")[:, 0].cpu().numpy()
                for img, pi, c in zip(batch, p, cam):
                    probs.append(round(float(pi), 3))
                    masks.append(pil_to_base64(render_mask(img, c, None, quantile=0.8)))
        finally:
            h.remove()
        return probs, masks
