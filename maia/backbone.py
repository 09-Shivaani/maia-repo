"""Language / vision-language model backbones.

Every backbone exposes ``chat(messages) -> str`` with OpenAI-style messages, where a message's
content is either a string or a list of parts ``{"type": "text"}`` / ``{"type": "image_url"}``.

Providers
---------
nvidia   NVIDIA API catalog (integrate.api.nvidia.com), e.g. nvidia/nemotron-3-ultra-550b-a55b.
         Key from env NVIDIA_API_KEY (or the backbone's ``api_key_env``).
ollama   Local Qwen served by Ollama  (http://localhost:11434/v1), e.g. qwen2.5vl:7b, qwen3:8b.
vllm     Local Qwen served by vLLM    (http://localhost:8000/v1),  e.g. Qwen/Qwen2.5-VL-7B-Instruct.
lmstudio Local model served by LM Studio (http://localhost:1234/v1).
hf       Local Qwen loaded in-process with transformers (Qwen2.5-VL / Qwen3-VL / text Qwen).
openai   OpenAI API (optional; key OPENAI_API_KEY).
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

PROVIDER_DEFAULTS = {
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "api_key_env": "NVIDIA_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key_env": None},
    "vllm": {"base_url": "http://localhost:8000/v1", "api_key_env": None},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "api_key_env": None},
    "openai": {"base_url": None, "api_key_env": "OPENAI_API_KEY"},
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class BackboneConfig:
    provider: str = "nvidia"
    model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    base_url: str | None = None
    api_key_env: str | None = None
    supports_vision: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 4096
    seed: int | None = 42
    stream: bool = True
    enable_thinking: bool = True
    timeout: float = 600.0
    max_retries: int = 5
    max_images_per_message: int | None = None  # some local servers cap images / request
    extra_body: dict = field(default_factory=dict)
    # hf provider only
    device_map: str = "auto"
    torch_dtype: str = "auto"

    @classmethod
    def from_dict(cls, d: dict) -> "BackboneConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "vision" in d and "supports_vision" not in d:
            known["supports_vision"] = bool(d["vision"])
        return cls(**known)


class Backbone:
    """Base class. Subclasses implement ``_chat``."""

    def __init__(self, cfg: BackboneConfig):
        self.cfg = cfg
        self.supports_vision = cfg.supports_vision
        self.n_calls = 0
        self.last_reasoning: str = ""

    @property
    def name(self) -> str:
        return f"{self.cfg.provider}:{self.cfg.model}"

    def chat(self, messages: list[dict], **kw) -> str:
        if not self.supports_vision:
            messages = strip_images(messages)
        elif self.cfg.max_images_per_message:
            messages = cap_images(messages, self.cfg.max_images_per_message)
        self.n_calls += 1
        out = self._chat(messages, **kw)
        return _THINK_RE.sub("", out).strip()

    def complete(self, prompt: str, system: str | None = None, images: list[str] | None = None, **kw) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        if images:
            from maia.utils import image_content_part

            parts = [{"type": "text", "text": prompt}] + [image_content_part(b) for b in images]
            msgs.append({"role": "user", "content": parts})
        else:
            msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, **kw)

    def _chat(self, messages: list[dict], **kw) -> str:  # pragma: no cover
        raise NotImplementedError


class OpenAICompatibleBackbone(Backbone):
    def __init__(self, cfg: BackboneConfig):
        super().__init__(cfg)
        from openai import OpenAI

        defaults = PROVIDER_DEFAULTS.get(cfg.provider, {})
        base_url = cfg.base_url or defaults.get("base_url")
        key_env = cfg.api_key_env or defaults.get("api_key_env")
        api_key = os.environ.get(key_env, "") if key_env else "not-needed"
        if key_env and not api_key and cfg.provider == "nvidia":
            api_key = os.environ.get("NVIDIA_API_KEY", "")  # one key works for all catalog models
        if key_env and not api_key:
            raise RuntimeError(
                f"Backbone {cfg.provider}:{cfg.model} needs env var {key_env}. "
                f"Put it in .env at the repo root (see .env.example)."
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=cfg.timeout)

    def _chat(self, messages, **kw) -> str:
        cfg = self.cfg
        extra = dict(cfg.extra_body)
        if cfg.provider in ("nvidia", "vllm"):
            extra.setdefault("chat_template_kwargs", {"enable_thinking": cfg.enable_thinking})
        args = dict(
            model=cfg.model,
            messages=messages,
            temperature=kw.get("temperature", cfg.temperature),
            top_p=kw.get("top_p", cfg.top_p),
            max_tokens=kw.get("max_tokens", cfg.max_tokens),
            stream=cfg.stream,
        )
        if cfg.seed is not None:
            args["seed"] = cfg.seed
        if extra:
            args["extra_body"] = extra

        delay = 2.0
        for attempt in range(cfg.max_retries):
            try:
                if cfg.stream:
                    content, reasoning = [], []
                    for chunk in self.client.chat.completions.create(**args):
                        if not chunk.choices:
                            continue
                        d = chunk.choices[0].delta
                        r = getattr(d, "reasoning_content", None)
                        if r:
                            reasoning.append(r)
                        if d.content:
                            content.append(d.content)
                    self.last_reasoning = "".join(reasoning)
                    return "".join(content)
                resp = self.client.chat.completions.create(**args)
                msg = resp.choices[0].message
                self.last_reasoning = getattr(msg, "reasoning_content", "") or ""
                return msg.content or ""
            except Exception as e:  # network / rate limit
                if attempt == cfg.max_retries - 1:
                    raise
                log.warning("Backbone %s call failed (%s); retrying in %.0fs", self.name, e, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60)
        return ""


class HFLocalBackbone(Backbone):
    """Qwen (VL or text) loaded locally with transformers."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__(cfg)
        import torch
        from transformers import AutoProcessor

        from maia.utils import hf_from_pretrained

        dtype = getattr(torch, cfg.torch_dtype) if cfg.torch_dtype not in ("auto", None) else "auto"
        self.processor = AutoProcessor.from_pretrained(cfg.model)
        if cfg.supports_vision:
            from transformers import AutoModelForImageTextToText

            self.model = hf_from_pretrained(AutoModelForImageTextToText, cfg.model, dtype, device_map=cfg.device_map)
        else:
            from transformers import AutoModelForCausalLM

            self.model = hf_from_pretrained(AutoModelForCausalLM, cfg.model, dtype, device_map=cfg.device_map)
        self.model.eval()

    def _chat(self, messages, **kw) -> str:
        import torch

        from maia.utils import base64_to_pil

        hf_msgs, images = [], []
        for m in messages:
            c = m["content"]
            if isinstance(c, str):
                hf_msgs.append({"role": m["role"], "content": [{"type": "text", "text": c}]})
                continue
            parts = []
            for p in c:
                if p["type"] == "text":
                    parts.append({"type": "text", "text": p["text"]})
                elif p["type"] == "image_url":
                    images.append(base64_to_pil(p["image_url"]["url"]))
                    parts.append({"type": "image"})
            hf_msgs.append({"role": m["role"], "content": parts})
        tmpl_kw = {"enable_thinking": self.cfg.enable_thinking}
        try:
            text = self.processor.apply_chat_template(hf_msgs, tokenize=False, add_generation_prompt=True, **tmpl_kw)
        except TypeError:
            text = self.processor.apply_chat_template(hf_msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt").to(self.model.device)
        temp = kw.get("temperature", self.cfg.temperature)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=kw.get("max_tokens", self.cfg.max_tokens),
                do_sample=temp > 0,
                temperature=max(temp, 1e-5),
                top_p=self.cfg.top_p,
            )
        gen = out[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]


def strip_images(messages: list[dict]) -> list[dict]:
    """Replace image parts with a placeholder for text-only models."""
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            txt = []
            for p in c:
                if p["type"] == "text":
                    txt.append(p["text"])
                else:
                    txt.append("[image omitted]")
            out.append({"role": m["role"], "content": "\n".join(txt)})
        else:
            out.append(m)
    return out


def cap_images(messages: list[dict], k: int) -> list[dict]:
    """Keep only the last ``k`` images in the conversation (older ones become placeholders)."""
    total = sum(1 for m in messages if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image_url")
    drop = max(0, total - k)
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, list) and drop:
            new = []
            for p in c:
                if p["type"] == "image_url" and drop:
                    drop -= 1
                    new.append({"type": "text", "text": "[earlier image omitted]"})
                else:
                    new.append(p)
            out.append({"role": m["role"], "content": new})
        else:
            out.append(m)
    return out


_CACHE: dict[str, Backbone] = {}


def build_backbone(spec: dict | BackboneConfig | Backbone) -> Backbone:
    if isinstance(spec, Backbone):
        return spec
    cfg = spec if isinstance(spec, BackboneConfig) else BackboneConfig.from_dict(spec)
    key = f"{cfg.provider}|{cfg.model}|{cfg.base_url}|{cfg.supports_vision}|{cfg.temperature}"
    if key in _CACHE:
        return _CACHE[key]
    bb: Backbone = HFLocalBackbone(cfg) if cfg.provider == "hf" else OpenAICompatibleBackbone(cfg)
    _CACHE[key] = bb
    return bb


class ScriptedBackbone(Backbone):
    """Deterministic backbone that replays canned replies (for tests and dry runs)."""

    def __init__(self, replies: list[str], supports_vision: bool = True):
        super().__init__(BackboneConfig(provider="scripted", model="scripted", supports_vision=supports_vision))
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def _chat(self, messages, **kw) -> str:
        self.seen.append(list(messages))
        return self.replies.pop(0) if self.replies else "[LABEL]: unknown"


def backbone_from_config(cfg: dict, role: str) -> Backbone:
    spec: Any = cfg["backbones"][role]
    if isinstance(spec, str):  # alias to another role
        spec = cfg["backbones"][spec]
    return build_backbone(spec)
