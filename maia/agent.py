"""The MAIA agent loop: the backbone writes Python experiments, we run them, and feed results back.

Protocol (paper Sec. 3, App. B):
  system prompt  = API specification of the System and Tools classes
  user prompt    = task specification (neuron description, bias, spurious, ...)
  each turn      = backbone reply with [HYPOTHESIS LIST] + a [CODE] block defining
                   ``run_experiment(system, tools)`` (or ``execute_command``)
  we execute it  -> everything passed to ``tools.log_experiment`` is returned as the next user turn
  stop           when the reply contains a final tag ([LABEL], [BIAS], [CIRCUIT], ...) and no code
"""
from __future__ import annotations

import contextlib
import io
import logging
import re
import signal
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from maia.backbone import Backbone
from maia.utils import image_content_part, write_json

log = logging.getLogger(__name__)

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
ENTRY_FUNCS = ("run_experiment", "execute_command")


# ---------------------------------------------------------------------------
# Experiment log shared by all Tools classes
# ---------------------------------------------------------------------------
class ExperimentLogMixin:
    """Holds experiment results as multimodal content parts the agent will see next turn."""

    def _init_log(self):
        self.experiment_log: list[list[dict]] = []
        self.results_list: list[dict] = []
        self._pending: list[dict] = []
        self.activation_threshold: float | None = None

    def _log_text(self, text: str):
        self._pending.append({"type": "text", "text": text})

    def _log_image(self, b64: str):
        self._pending.append(image_content_part(b64))

    def flush_log(self) -> list[dict]:
        """Return and clear the parts logged since the last flush."""
        parts, self._pending = self._pending, []
        if parts:
            self.experiment_log.append(parts)
        return parts


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------
class ExecutionTimeout(Exception):
    pass


@dataclass
class ExecutionResult:
    ok: bool
    stdout: str = ""
    error: str = ""
    return_value: Any = None


def extract_code(reply: str) -> str | None:
    """Return the last fenced python block that defines an entry function."""
    blocks = CODE_RE.findall(reply)
    good = [b for b in blocks if any(f"def {f}" in b for f in ENTRY_FUNCS)]
    if good:
        return good[-1]
    return blocks[-1] if blocks else None


def execute_code(code: str, system: Any, tools: Any, timeout: int = 1800, extra_globals: dict | None = None) -> ExecutionResult:
    """Execute agent-written code. NOTE: not a security sandbox - run on a machine you control."""
    ns: dict[str, Any] = {"__name__": "__maia_experiment__"}
    ns.update(_default_globals())
    ns.update(extra_globals or {})
    out = io.StringIO()

    use_alarm = threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM")

    def _handler(signum, frame):
        raise ExecutionTimeout(f"experiment exceeded {timeout}s")

    old = None
    if use_alarm:
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout)
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(code, "<maia_experiment>", "exec"), ns)
            fn = next((ns[f] for f in ENTRY_FUNCS if callable(ns.get(f))), None)
            if fn is None:
                return ExecutionResult(False, out.getvalue(), "No run_experiment(system, tools) function was defined.")
            rv = fn(system, tools)
        return ExecutionResult(True, out.getvalue(), return_value=rv)
    except Exception:
        tb = traceback.format_exc(limit=6)
        return ExecutionResult(False, out.getvalue(), tb)
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def _default_globals() -> dict:
    g: dict[str, Any] = {}
    for name in ("numpy", "math", "random", "itertools", "collections"):
        try:
            g[name] = __import__(name)
        except ImportError:  # pragma: no cover
            pass
    g["np"] = g.get("numpy")
    try:
        import torch

        g["torch"] = torch
    except ImportError:  # pragma: no cover
        pass
    from typing import Dict, List, Tuple, Union

    g.update(List=List, Tuple=Tuple, Dict=Dict, Union=Union)
    return g


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
@dataclass
class AgentResult:
    final_reply: str
    fields: dict[str, str]
    rounds: int
    transcript: list[dict] = field(default_factory=list)
    finished: bool = True


def parse_fields(text: str, tags: tuple[str, ...]) -> dict[str, str]:
    """Parse "[TAG]: value" blocks. Value runs until the next [TAG] or end of text."""
    out: dict[str, str] = {}
    pattern = re.compile(r"\[([A-Z][A-Z0-9 _]*)\]\s*:?", re.MULTILINE)
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        tag = m.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        val = text[m.end() : end].strip().strip("#").strip()
        if any(tag.startswith(t) for t in tags):
            out[tag] = val
    return out


class MAIAAgent:
    def __init__(
        self,
        backbone: Backbone,
        system: Any,
        tools: Any,
        api_spec: str,
        user_prompt: str,
        final_tags: tuple[str, ...] = ("LABEL", "DESCRIPTION"),
        required_tag: str = "LABEL",
        max_rounds: int = 25,
        exec_timeout: int = 1800,
        vision_proxy: Backbone | None = None,
        log_dir: str | Path | None = None,
    ):
        self.backbone = backbone
        self.system = system
        self.tools = tools
        self.api_spec = api_spec
        self.user_prompt = user_prompt
        self.final_tags = final_tags
        self.required_tag = required_tag
        self.max_rounds = max_rounds
        self.exec_timeout = exec_timeout
        self.vision_proxy = vision_proxy
        self.log_dir = Path(log_dir) if log_dir else None
        self.messages: list[dict] = []
        self.transcript: list[dict] = []

    # -- image handling for text-only backbones ------------------------------------------
    def _adapt_parts(self, parts: list[dict]) -> list[dict] | str:
        if self.backbone.supports_vision:
            return parts
        if self.vision_proxy is None:
            return "\n".join(p["text"] if p["type"] == "text" else "[image]" for p in parts)
        # Text-only agent: describe each image with a VLM "eyes" model (no hypothesis context,
        # mirroring the paper's impartial describe_images tool).
        lines = []
        for p in parts:
            if p["type"] == "text":
                lines.append(p["text"])
            else:
                desc = self.vision_proxy.complete(
                    "Describe the image concisely. If parts of the image are darkened and one region is "
                    "left bright/outlined, describe ONLY the highlighted region.",
                    images=[p["image_url"]["url"]],
                    max_tokens=200,
                )
                lines.append(f"[image description: {desc.strip()}]")
        return "\n".join(lines)

    def _record(self, role: str, content: Any):
        text = content if isinstance(content, str) else "\n".join(
            p["text"] if p["type"] == "text" else "<image>" for p in content
        )
        self.transcript.append({"role": role, "content": text})

    def run(self) -> AgentResult:
        self.messages = [
            {"role": "system", "content": self.api_spec},
            {"role": "user", "content": self.user_prompt},
        ]
        self._record("system", "<api spec>")
        self._record("user", self.user_prompt)
        reply = ""
        for rnd in range(1, self.max_rounds + 1):
            reply = self.backbone.chat(self.messages)
            self.messages.append({"role": "assistant", "content": reply})
            self._record("assistant", reply)
            code = extract_code(reply)
            fields = parse_fields(reply, self.final_tags)
            if code is None and self.required_tag in fields:
                return self._finish(reply, fields, rnd, True)
            if code is None:
                feedback: Any = (
                    "No [CODE] block was found. Either write a ```python block defining "
                    "run_experiment(system, tools), or give your final answer in the required format."
                )
            else:
                res = execute_code(code, self.system, self.tools, timeout=self.exec_timeout)
                parts = self.tools.flush_log()
                if not res.ok:
                    parts = parts + [{"type": "text", "text": f"Execution error:\n{res.error}"}]
                elif not parts:
                    parts = [{"type": "text", "text": "The experiment ran but nothing was logged. "
                              "Call tools.log_experiment(...) to see results." + (f"\nstdout:\n{res.stdout[-2000:]}" if res.stdout else "")}]
                feedback = self._adapt_parts(parts)
            self.messages.append({"role": "user", "content": feedback})
            self._record("user", feedback)
        # Out of rounds: ask for the final answer.
        self.messages.append({"role": "user", "content": "You have run out of experiments. Output your final answer now in the required format."})
        reply = self.backbone.chat(self.messages)
        self._record("assistant", reply)
        fields = parse_fields(reply, self.final_tags)
        return self._finish(reply, fields, self.max_rounds, self.required_tag in fields)

    def _finish(self, reply, fields, rounds, finished) -> AgentResult:
        res = AgentResult(reply, fields, rounds, list(self.transcript), finished)
        if self.log_dir:
            write_json(
                {"fields": fields, "rounds": rounds, "finished": finished, "backbone": self.backbone.name,
                 "transcript": self.transcript},
                self.log_dir / "transcript.json",
            )
        return res
