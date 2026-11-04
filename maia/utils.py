"""Shared helpers: seeding, image <-> base64, JSON logging, bootstrap CIs."""
from __future__ import annotations

import base64
import io
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass


def get_device(device: str | int | None = None) -> str:
    import torch

    if device is None or device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        return f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def pil_to_base64(img, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(b64: str):
    from PIL import Image

    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def to_pil(img):
    """Accept PIL image, base64 string, path, numpy array, or CHW/HWC tensor."""
    from PIL import Image

    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, (str, Path)):
        s = str(img)
        if os.path.exists(s):
            return Image.open(s).convert("RGB")
        return base64_to_pil(s)
    if isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    try:
        import torch

        if isinstance(img, torch.Tensor):
            t = img.detach().cpu().float()
            if t.ndim == 4:
                t = t[0]
            if t.shape[0] in (1, 3):
                t = t.permute(1, 2, 0)
            if t.max() > 1.5:
                t = t / 255.0
            return Image.fromarray((t.clamp(0, 1).numpy() * 255).astype(np.uint8)).convert("RGB")
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"Cannot convert {type(img)} to PIL image")


def image_content_part(b64: str) -> dict:
    """OpenAI-style multimodal content part for a base64 PNG."""
    if not b64.startswith("data:"):
        b64 = f"data:image/png;base64,{b64}"
    return {"type": "image_url", "image_url": {"url": b64}}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean. Returns (mean, lo, hi)."""
    vals = np.asarray(list(values), dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)


def chunked(seq: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def hf_from_pretrained(cls, name: str, dtype=None, **kw):
    """from_pretrained that works across transformers versions (`dtype` in v5, `torch_dtype` before)."""
    if dtype is None:
        return cls.from_pretrained(name, **kw)
    try:
        return cls.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(name, torch_dtype=dtype, **kw)
