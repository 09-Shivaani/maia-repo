"""YAML config loading with ${ENV_VAR} expansion and simple dotted overrides."""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv_if_present() -> None:
    """Load .env from the repo root (git-ignored) so API keys never live in code."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    for p in (Path.cwd() / ".env", REPO_ROOT / ".env"):
        if p.exists():
            load_dotenv(p, override=False)


def _expand(v: Any) -> Any:
    if isinstance(v, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load configs/default.yaml, then `path` (which may `inherit:` another file), then overrides.

    Overrides use dotted keys: ["backbones.agent.model=qwen2.5vl:7b", "seed=0"].
    """
    load_dotenv_if_present()
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "default.yaml"))
    if path is not None:
        p = Path(path)
        if not p.exists():
            p = REPO_ROOT / "configs" / Path(path).name
        user = yaml.safe_load(open(p)) or {}
        parent = user.pop("inherit", None)
        if parent:
            cfg = deep_update(cfg, yaml.safe_load(open(REPO_ROOT / "configs" / parent)) or {})
        cfg = deep_update(cfg, user)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(val)
    return _expand(cfg)
