"""Precompute top-k maximally activating dataset exemplars for every unit of a layer.

The paper uses the ImageNet validation set. This reproduction uses a real-image folder instead
(by default the CC3M subset downloaded by scripts/download_datasets.py); any folder of images works.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from maia.vision.models import LoadedModel, forward_model
from maia.vision.system import pil_batch_to_tensor, preprocess_pil

log = logging.getLogger(__name__)
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root: str | Path, limit: int | None = None) -> list[str]:
    root = Path(root)
    paths = sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not paths:
        raise FileNotFoundError(f"No images found under {root}. Run scripts/download_datasets.py first.")
    return paths[:limit] if limit else paths


class _ImgDS(Dataset):
    def __init__(self, paths, res):
        self.paths, self.res = paths, res

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = preprocess_pil(Image.open(self.paths[i]).convert("RGB"), self.res)
            ok = True
        except Exception:
            img, ok = Image.new("RGB", (self.res, self.res)), False
        return np.asarray(img, dtype=np.uint8), ok


def _collate(b):
    return np.stack([x[0] for x in b]), np.array([x[1] for x in b])


def cache_path(cache_dir: str | Path, model_name: str, layer: str) -> Path:
    return Path(cache_dir) / model_name / f"{layer.replace('/', '_')}.npz"


@torch.no_grad()
def compute_layer_exemplars(lm: LoadedModel, layer: str, image_paths: list[str], k: int = 15,
                            batch_size: int = 64, num_workers: int = 8, device: str = "cuda",
                            out_path: str | Path | None = None, n_prefix_tokens: int | None = None):
    module = lm.resolve_layer(layer)
    store = {}
    h = module.register_forward_hook(lambda m, i, o: store.__setitem__("o", o[0] if isinstance(o, tuple) else o))
    dl = DataLoader(_ImgDS(image_paths, lm.resolution), batch_size=batch_size, num_workers=num_workers, collate_fn=_collate)
    top_acts = top_idx = None
    offset = 0
    npt = lm.n_prefix_tokens if n_prefix_tokens is None else n_prefix_tokens
    try:
        for bi, (arr, ok) in enumerate(dl):
            x = pil_batch_to_tensor([Image.fromarray(a) for a in arr], *lm.normalize, device)
            forward_model(lm, x)
            o = store["o"].float()
            if o.ndim == 4:  # B,C,H,W
                a = o.flatten(2).amax(2)
            else:  # B,T,C
                a = o[:, npt:].amax(1)
            a[torch.from_numpy(~ok).to(a.device)] = -float("inf")
            idx = torch.arange(offset, offset + len(arr), device=a.device).unsqueeze(1).expand_as(a)
            a, idx = a.T, idx.T  # U,B
            if top_acts is None:
                cat_a, cat_i = a, idx
            else:
                cat_a, cat_i = torch.cat([top_acts, a], 1), torch.cat([top_idx, idx], 1)
            kk = min(k, cat_a.shape[1])
            top_acts, sel = cat_a.topk(kk, dim=1)
            top_idx = torch.gather(cat_i, 1, sel)
            offset += len(arr)
            if bi % 50 == 0:
                log.info("exemplars %s/%s: %d/%d images", lm.name, layer, offset, len(image_paths))
    finally:
        h.remove()
    res = {"acts": top_acts.cpu().numpy(), "idx": top_idx.cpu().numpy()}
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **res)
        (out_path.with_suffix(".paths.json")).write_text(json.dumps(image_paths))
    return res


def load_exemplars(path: str | Path):
    path = Path(path)
    d = np.load(path)
    paths = json.loads(path.with_suffix(".paths.json").read_text())
    return d["acts"], d["idx"], paths
