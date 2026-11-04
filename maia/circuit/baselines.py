"""Non-agentic circuit-discovery baselines.

* correlation        rank nodes by |corr(activation, target)|  (correlation-based subgraph extraction)
* attribution        one-shot attribution patching ranking, no iterative refinement
* greedy_pruning     ACDC-style (Conmy et al. 2023), node level: start from all nodes and remove each node
                     (reverse order) if the faithfulness drop stays below tau
For the overlap metrics a ranking is cut at the size that maximises faithfulness-per-node, or at a
fixed size k when given.
"""
from __future__ import annotations

from typing import Dict, List

from maia.circuit.base import CircuitSystem
from maia.circuit.tools import correlation_scores


def rank(scores: Dict[str, float]) -> List[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def correlation_ranking(sys: CircuitSystem) -> List[str]:
    return rank(correlation_scores(sys))


def attribution_ranking(sys: CircuitSystem) -> List[str]:
    return rank(sys.attribution_scores())


def smallest_faithful_prefix(sys: CircuitSystem, ranking: List[str], threshold: float = 0.9, max_size: int | None = None) -> List[str]:
    """Shortest prefix of the ranking with faithfulness ≥ threshold (binary search over prefix sizes)."""
    cache: dict = {}
    hi = min(len(ranking), max_size or len(ranking))
    lo = 1
    if sys.faithfulness(ranking[:hi], cache=cache) < threshold:
        return ranking[:hi]
    while lo < hi:
        mid = (lo + hi) // 2
        if sys.faithfulness(ranking[:mid], cache=cache) >= threshold:
            hi = mid
        else:
            lo = mid + 1
    return ranking[:lo]


def greedy_pruning(sys: CircuitSystem, tau: float = 0.02, order: List[str] | None = None) -> List[str]:
    cache: dict = {}
    circuit = list(order or sys.nodes)
    current = sys.faithfulness(circuit, cache=cache)
    for n in reversed(list(order or sys.nodes)):
        trial = [x for x in circuit if x != n]
        f = sys.faithfulness(trial, cache=cache)
        if current - f < tau:
            circuit, current = trial, f
    return circuit
