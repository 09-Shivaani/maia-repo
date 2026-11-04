"""Synthetic networks with planted, known circuits (ground truth for MAIA-Circuit).

Architecture: x ∈ {0,1}^16 -> L1 (relu, width W) -> L2 (relu, width W) -> outputs.
Planted task circuits (hidden units at random positions):

  xor      y = x0 ⊕ x1                 L1: relu(x0-x1), relu(x1-x0)          L2: relu(a+b)
  and_or   y = (x2 ∧ x3) ∨ x4          L1: relu(x2+x3-1), relu(x4)           L2: relu(g+r), relu(g+r-1)  (y = p-q)
  parity3  y = x5 ⊕ x6 ⊕ x7            L1: relu(x5-x6), relu(x6-x5), relu(x7) L2: relu(s-x7), relu(x7-s)  (y = c+d)

Distractors, chosen to defeat purely correlational methods:
  * decoy copies: L2 units that exactly copy a circuit unit's computation but write only to decoy outputs
    (perfectly correlated with the task, causally irrelevant);
  * input-reading decoys in L1 that read the task's inputs;
  * random units with random weights that write only to decoy outputs.
Ablation default is mean-ablation over the input distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import torch

from maia.circuit.base import CircuitSystem

TASKS = ("xor", "and_or", "parity3")


@dataclass
class PlantedNet:
    W1: torch.Tensor
    b1: torch.Tensor
    W2: torch.Tensor
    b2: torch.Tensor
    W3: torch.Tensor
    b3: torch.Tensor
    output_names: List[str]
    circuits: Dict[str, List[str]]
    decoys: Dict[str, List[str]]
    width: int

    def forward(self, x, h1_override=None, h2_override=None):
        h1 = torch.relu(x @ self.W1.T + self.b1)
        if h1_override is not None:
            h1 = h1_override(h1)
        h2 = torch.relu(h1 @ self.W2.T + self.b2)
        if h2_override is not None:
            h2 = h2_override(h2)
        return h1, h2, h2 @ self.W3.T + self.b3


def build_planted_net(width: int = 32, d_in: int = 16, n_decoy_out: int = 4, seed: int = 0) -> PlantedNet:
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    W1, b1 = torch.zeros(width, d_in), torch.zeros(width)
    W2, b2 = torch.zeros(width, width), torch.zeros(width)
    outs = list(TASKS) + [f"decoy{i}" for i in range(n_decoy_out)]
    W3, b3 = torch.zeros(len(outs), width), torch.zeros(len(outs))
    free1, free2 = list(rng.permutation(width)), list(rng.permutation(width))
    take1 = lambda: int(free1.pop())  # noqa: E731
    take2 = lambda: int(free2.pop())  # noqa: E731
    circuits: Dict[str, List[str]] = {}
    decoys: Dict[str, List[str]] = {}
    oi = {n: i for i, n in enumerate(outs)}

    def decoy_out():
        return oi[f"decoy{rng.integers(n_decoy_out)}"]

    # --- xor ---------------------------------------------------------------------------
    a, b = take1(), take1()
    W1[a, 0], W1[a, 1] = 1, -1
    W1[b, 0], W1[b, 1] = -1, 1
    e = take2()
    W2[e, a], W2[e, b] = 1, 1
    W3[oi["xor"], e] = 1
    circuits["xor"] = [f"L1.{a}", f"L1.{b}", f"L2.{e}"]
    # decoy copy of e + input-reading decoy
    dc = take2()
    W2[dc, a], W2[dc, b] = 1, 1
    W3[decoy_out(), dc] = 1
    di = take1()
    W1[di, 0], W1[di, 1] = 1, 1
    decoys["xor"] = [f"L2.{dc}", f"L1.{di}"]

    # --- and_or --------------------------------------------------------------------------
    gq, r = take1(), take1()
    W1[gq, 2], W1[gq, 3], b1[gq] = 1, 1, -1
    W1[r, 4] = 1
    p, q = take2(), take2()
    W2[p, gq], W2[p, r] = 1, 1
    W2[q, gq], W2[q, r], b2[q] = 1, 1, -1
    W3[oi["and_or"], p], W3[oi["and_or"], q] = 1, -1
    circuits["and_or"] = [f"L1.{gq}", f"L1.{r}", f"L2.{p}", f"L2.{q}"]
    dc2 = take2()
    W2[dc2, gq], W2[dc2, r] = 1, 1
    W3[decoy_out(), dc2] = 1
    di2 = take1()
    W1[di2, 2], W1[di2, 3], W1[di2, 4] = 1, 1, 1
    decoys["and_or"] = [f"L2.{dc2}", f"L1.{di2}"]

    # --- parity3 -------------------------------------------------------------------------
    a2, b2_, r7 = take1(), take1(), take1()
    W1[a2, 5], W1[a2, 6] = 1, -1
    W1[b2_, 5], W1[b2_, 6] = -1, 1
    W1[r7, 7] = 1
    c, d = take2(), take2()
    W2[c, a2], W2[c, b2_], W2[c, r7] = 1, 1, -1
    W2[d, a2], W2[d, b2_], W2[d, r7] = -1, -1, 1
    W3[oi["parity3"], c], W3[oi["parity3"], d] = 1, 1
    circuits["parity3"] = [f"L1.{a2}", f"L1.{b2_}", f"L1.{r7}", f"L2.{c}", f"L2.{d}"]
    dc3 = take2()
    W2[dc3, a2], W2[dc3, b2_], W2[dc3, r7] = 1, 1, -1
    W3[decoy_out(), dc3] = 1
    decoys["parity3"] = [f"L2.{dc3}"]

    # --- random distractor units ---------------------------------------------------------
    for u in free1:
        W1[u] = torch.randn(d_in, generator=g) / np.sqrt(d_in)
        b1[u] = 0.1 * torch.randn(1, generator=g)
    used1 = [i for i in range(width) if W1[i].abs().sum() > 0]
    for u in free2:
        W2[u, used1] = torch.randn(len(used1), generator=g) / np.sqrt(len(used1))
        W3[decoy_out(), u] = torch.randn(1, generator=g).item()
    return PlantedNet(W1, b1, W2, b2, W3, b3, outs, circuits, decoys, width)


def task_labels(x: torch.Tensor, task: str) -> torch.Tensor:
    xi = x.long()
    if task == "xor":
        return (xi[:, 0] ^ xi[:, 1]).float()
    if task == "and_or":
        return ((xi[:, 2] & xi[:, 3]) | xi[:, 4]).float()
    if task == "parity3":
        return (xi[:, 5] ^ xi[:, 6] ^ xi[:, 7]).float()
    raise ValueError(task)


class SyntheticCircuitSystem(CircuitSystem):
    """Circuit system over a PlantedNet for one task. Metric = accuracy of the task output (threshold 0.5)."""

    def __init__(self, net: PlantedNet, task: str, n_examples: int = 2048, seed: int = 0, mode: str = "mean"):
        self.net, self.task, self.mode = net, task, mode
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randint(0, 2, (n_examples, net.W1.shape[1]), generator=g).float()
        self.y = task_labels(self.x, task)
        self.out_idx = net.output_names.index(task)
        self.nodes = [f"L1.{i}" for i in range(net.width)] + [f"L2.{i}" for i in range(net.width)]
        h1, h2, _ = net.forward(self.x)
        self._clean = {"L1": h1, "L2": h2}
        self._mean = {"L1": h1.mean(0), "L2": h2.mean(0)}
        self._perm = torch.randperm(n_examples, generator=g)
        self.ground_truth = list(net.circuits[task])
        self.decoys = list(net.decoys[task])
        self.task_description = {
            "xor": "output 'xor' of a small MLP; label = XOR of two input bits",
            "and_or": "output 'and_or' of a small MLP; label = (bit AND bit) OR bit",
            "parity3": "output 'parity3' of a small MLP; label = parity of three input bits",
        }[task]

    def _override(self, layer: str, ablate: set, interventions: Dict[str, float], mode: str):
        idx = [int(n.split(".")[1]) for n in ablate if n.startswith(layer + ".")]
        iv = {int(n.split(".")[1]): v for n, v in interventions.items() if n.startswith(layer + ".")}
        if not idx and not iv:
            return None

        def f(h):
            h = h.clone()
            if idx:
                if mode == "zero":
                    h[:, idx] = 0
                elif mode == "resample":
                    h[:, idx] = h[self._perm][:, idx]
                else:
                    h[:, idx] = self._mean[layer][idx]
            for i, v in iv.items():
                h[:, i] = v
            return h

        return f

    def outputs(self, ablate: Iterable[str] = (), interventions: Dict[str, float] | None = None, mode: str | None = None):
        ab = set(ablate)
        iv = interventions or {}
        m = mode or self.mode
        return self.net.forward(self.x, self._override("L1", ab, iv, m), self._override("L2", ab, iv, m))

    def metric(self, ablate: Iterable[str] = (), interventions: Dict[str, float] | None = None, mode: str | None = None) -> float:
        _, _, out = self.outputs(ablate, interventions, mode)
        pred = (out[:, self.out_idx] > 0.5).float()
        return float((pred == self.y).float().mean())

    def node_activations(self, node: str) -> np.ndarray:
        layer, i = node.split(".")
        return self._clean[layer][:, int(i)].numpy()

    def node_effect_on(self, source: str, target: str, mode: str | None = None) -> float:
        """Average change in target node's activation when source is ablated."""
        h1, h2, _ = self.outputs([source], mode=mode)
        new = {"L1": h1, "L2": h2}
        layer, i = target.split(".")
        return float((new[layer][:, int(i)] - self._clean[layer][:, int(i)]).abs().mean())

    def target_values(self) -> np.ndarray:
        return self.y.numpy()

    def attribution_scores(self) -> Dict[str, float]:
        """grad(task output) · (mean - clean) per node (attribution patching, Nanda 2023 / Syed et al. 2023)."""
        x = self.x
        h1 = torch.relu(x @ self.net.W1.T + self.net.b1).requires_grad_(True)
        h2 = torch.relu(h1 @ self.net.W2.T + self.net.b2)
        h2.retain_grad()
        out = (h2 @ self.net.W3.T + self.net.b3)[:, self.out_idx]
        # signed metric: push output towards correct label
        obj = (out * (2 * self.y - 1)).sum()
        obj.backward()
        s = {}
        for layer, h in (("L1", h1), ("L2", h2)):
            delta = self._mean[layer] - h.detach()
            sc = (h.grad * delta).abs().mean(0)
            for i, v in enumerate(sc.tolist()):
                s[f"{layer}.{i}"] = v
        return s
