"""Common interface for circuit-discovery systems and the MIB faithfulness metrics.

A CircuitSystem exposes a set of named nodes (units, attention heads, MLP blocks) and can evaluate
the task metric while some nodes are ablated / patched / intervened on.

Faithfulness (Mueller et al. 2025, MIB):  f(C) = (m(C) - m(∅)) / (m(N) - m(∅))
  m(N)  metric of the full model, m(∅) metric with every node ablated, m(C) metric when only the
  circuit C keeps its clean activations and all other nodes are ablated.
CPR = ∫ f(C_k) dk  (higher is better);  CMD = ∫ |1 - f(C_k)| dk  (lower is better), trapezoidal rule
over the circuit-size grid k ∈ {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1}.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Sequence

import numpy as np

MIB_GRID = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


class CircuitSystem(ABC):
    nodes: List[str]
    task_description: str = ""

    @abstractmethod
    def metric(self, ablate: Iterable[str] = (), interventions: Dict[str, float] | None = None,
               mode: str | None = None) -> float:
        """Mean task metric with `ablate` nodes ablated and `interventions` {node: value} applied."""

    @abstractmethod
    def node_activations(self, node: str) -> np.ndarray:
        """Per-example scalar summary of a node's clean activation."""

    def target_values(self) -> np.ndarray:
        """Per-example target (label or clean metric) used by correlation baselines."""
        raise NotImplementedError

    def attribution_scores(self) -> Dict[str, float]:
        """Attribution-patching estimate of each node's effect (|grad · Δact|)."""
        raise NotImplementedError

    # -- convenience -------------------------------------------------------------------
    def full_metric(self) -> float:
        return self.metric()

    def empty_metric(self) -> float:
        return self.metric(ablate=self.nodes)

    def keep_only(self, circuit: Iterable[str], mode: str | None = None) -> float:
        keep = set(circuit)
        return self.metric(ablate=[n for n in self.nodes if n not in keep], mode=mode)

    def faithfulness(self, circuit: Iterable[str], mode: str | None = None, cache: dict | None = None) -> float:
        cache = cache if cache is not None else {}
        if "N" not in cache:
            cache["N"], cache["E"] = self.full_metric(), self.empty_metric()
        denom = cache["N"] - cache["E"]
        if abs(denom) < 1e-12:
            return float("nan")
        return (self.keep_only(circuit, mode) - cache["E"]) / denom

    def necessity(self, circuit: Iterable[str]) -> float:
        """Fraction of the full-model metric lost when only the circuit is ablated (1 = fully necessary)."""
        n, e = self.full_metric(), self.empty_metric()
        return (n - self.metric(ablate=list(circuit))) / (n - e) if abs(n - e) > 1e-12 else float("nan")


def circuit_overlap(found: Iterable[str], truth: Iterable[str]) -> Dict[str, float]:
    f, t = set(found), set(truth)
    tp = len(f & t)
    p = tp / len(f) if f else 0.0
    r = tp / len(t) if t else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "iou": tp / len(f | t) if f | t else 0.0, "size": len(f)}


def cpr_cmd(system: CircuitSystem, ranking: Sequence[str], grid: Sequence[float] = MIB_GRID) -> Dict[str, float]:
    """Node-level CPR / CMD: C_k = top ceil(k·|N|) nodes of `ranking` (most important first)."""
    cache: dict = {}
    n = len(system.nodes)
    fs = []
    for k in grid:
        size = max(0, int(np.ceil(k * n)))
        fs.append(system.faithfulness(ranking[:size], cache=cache))
    fs = np.array(fs, dtype=float)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    x = np.array(grid, dtype=float)
    return {"cpr": float(trapz(fs, x)), "cmd": float(trapz(np.abs(1 - fs), x)),
            "curve": dict(zip(map(float, grid), map(float, fs)))}
