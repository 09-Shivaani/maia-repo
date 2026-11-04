"""Tools for the circuit-discovery agent (MAIA-Circuit)."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from maia.agent import ExperimentLogMixin
from maia.circuit.base import CircuitSystem


class CircuitTools(ExperimentLogMixin):
    def __init__(self, system: CircuitSystem, allowed: set[str] | None = None, max_nodes_listed: int = 400):
        self._init_log()
        self.sys = system
        self.allowed = allowed
        self.max_nodes_listed = max_nodes_listed
        self._cache: dict = {}
        self.n_metric_calls = 0

    def _check(self, name):
        if self.allowed is not None and name not in self.allowed:
            raise RuntimeError(f"The tool '{name}' is not available in this experiment.")

    def _valid(self, nodes: Iterable[str]) -> List[str]:
        nodes = [nodes] if isinstance(nodes, str) else list(nodes)
        bad = [n for n in nodes if n not in self.sys.nodes]
        if bad:
            raise ValueError(f"unknown nodes {bad[:5]}; call tools.list_nodes() for valid names")
        return nodes

    # ------------------------------------------------------------------------------------
    def list_nodes(self) -> List[str]:
        return list(self.sys.nodes)

    def baseline(self) -> Dict[str, float]:
        """Task metric of the full model and of the fully ablated model."""
        self.n_metric_calls += 2
        return {"full": self.sys.full_metric(), "all_ablated": self.sys.empty_metric()}

    def ablate(self, nodes: Iterable[str], mode: str | None = None) -> float:
        """Task metric with `nodes` ablated (mode: mean | zero | resample/patching)."""
        self._check("ablate")
        self.n_metric_calls += 1
        return self.sys.metric(ablate=self._valid(nodes), mode=mode)

    def ablate_each(self, nodes: Iterable[str], mode: str | None = None) -> Dict[str, float]:
        """Drop in task metric when each node is ablated alone (bigger = more important)."""
        self._check("ablate")
        full = self.sys.full_metric()
        out = {}
        for n in self._valid(nodes):
            self.n_metric_calls += 1
            out[n] = round(full - self.sys.metric(ablate=[n], mode=mode), 4)
        return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))

    def intervene(self, node_values: Dict[str, float]) -> float:
        """Task metric after setting (synthetic nets) / scaling (transformers) node outputs."""
        self._check("intervene")
        self._valid(node_values)
        self.n_metric_calls += 1
        return self.sys.metric(interventions=dict(node_values))

    def measure_causal_effect(self, source: str, target: str, mode: str | None = None) -> float:
        """Mean absolute change of `target`'s activation when `source` is ablated (only for systems that
        expose node_effect_on; otherwise the change in task metric attributable to source)."""
        self._check("measure_causal_effect")
        self._valid([source, target])
        if hasattr(self.sys, "node_effect_on"):
            return self.sys.node_effect_on(source, target, mode)
        raise RuntimeError("this system does not expose node-to-node effects; use ablate / test_circuit")

    def extract_subgraph(self, method: str = "attribution", top_k: int = 20) -> Dict[str, float]:
        """Candidate important nodes, ranked. method: 'attribution' (gradient-based attribution patching) or
        'correlation' (|corr(node activation, target)|)."""
        self._check("extract_subgraph")
        key = ("sub", method)
        if key not in self._cache:
            if method == "attribution":
                self._cache[key] = self.sys.attribution_scores()
            else:
                self._cache[key] = correlation_scores(self.sys)
        s = self._cache[key]
        return dict(sorted(s.items(), key=lambda kv: -kv[1])[:top_k])

    def node_activation_stats(self, nodes: Iterable[str]) -> Dict[str, dict]:
        """Mean / std of activations and correlation with the task target, per node."""
        self._check("node_activation_stats")
        tv = self.sys.target_values()
        out = {}
        for n in self._valid(nodes):
            a = self.sys.node_activations(n)
            c = float(np.corrcoef(a, tv)[0, 1]) if a.std() > 0 and tv.std() > 0 else 0.0
            out[n] = {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4), "corr_with_target": round(c, 4)}
        return out

    def test_circuit(self, nodes: Iterable[str], mode: str | None = None) -> Dict[str, float]:
        """Faithfulness f(C) (only C kept, everything else ablated; 1 = fully sufficient) and necessity
        (fraction of metric lost when only C is ablated; 1 = fully necessary)."""
        self._check("test_circuit")
        nodes = self._valid(nodes)
        self.n_metric_calls += 2
        return {"faithfulness": round(self.sys.faithfulness(nodes, mode, self._cache.setdefault("faith", {})), 4),
                "necessity": round(self.sys.necessity(nodes), 4), "size": len(nodes)}

    def log_experiment(self, results, description: str = "") -> None:
        """Record any results (dict / list / number / text). ONLY logged results are shown to you."""
        text = results if isinstance(results, str) else _fmt(results)
        self._log_text((description + "\n" if description else "") + text)

    save_experiment_log = log_experiment


def _fmt(x, depth=0) -> str:
    if isinstance(x, dict):
        items = list(x.items())
        s = ", ".join(f"{k}: {_fmt(v, depth + 1)}" for k, v in items[:200])
        return "{" + s + (" ..." if len(items) > 200 else "") + "}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_fmt(v, depth + 1) for v in list(x)[:200]) + "]"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def correlation_scores(sys: CircuitSystem) -> Dict[str, float]:
    tv = sys.target_values()
    out = {}
    for n in sys.nodes:
        a = sys.node_activations(n)
        out[n] = abs(float(np.corrcoef(a, tv)[0, 1])) if a.std() > 0 and tv.std() > 0 else 0.0
    return out
