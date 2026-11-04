import numpy as np

from maia.agent import MAIAAgent
from maia.backbone import ScriptedBackbone
from maia.circuit import baselines as B
from maia.circuit.base import circuit_overlap, cpr_cmd
from maia.circuit.synthetic import TASKS, SyntheticCircuitSystem, build_planted_net
from maia.circuit.tools import CircuitTools
from maia.prompts import load_prompt


def systems(seed=0):
    net = build_planted_net(seed=seed)
    return [SyntheticCircuitSystem(net, t, seed=seed) for t in TASKS]


def test_planted_circuits_are_exact():
    for s in systems():
        assert s.full_metric() == 1.0
        assert s.faithfulness(s.ground_truth) > 0.99          # sufficient
        assert s.necessity(s.ground_truth) > 0.5               # necessary
        assert s.faithfulness(s.decoys) < 0.5                  # decoys alone do not implement it


def test_decoys_fool_correlation_but_not_attribution():
    fooled = 0
    for seed in range(3):
        for s in systems(seed):
            k = len(s.ground_truth)
            corr = circuit_overlap(B.correlation_ranking(s)[:k], s.ground_truth)["f1"]
            attr = circuit_overlap(B.attribution_ranking(s)[:k], s.ground_truth)["f1"]
            fooled += corr < 1.0
            assert attr >= corr
    assert fooled > 0


def test_cpr_cmd_bounds():
    s = systems()[0]
    r = cpr_cmd(s, B.attribution_ranking(s))
    assert 0 <= r["cpr"] <= 1.0 + 1e-6 and r["cmd"] >= 0


def test_greedy_pruning_recovers_truth():
    s = systems()[0]
    found = B.greedy_pruning(s, order=B.attribution_ranking(s)[::-1])
    assert set(s.ground_truth) <= set(found)
    assert s.faithfulness(found) > 0.95


def test_agent_end_to_end_with_scripted_backbone():
    s = systems()[0]
    truth = ", ".join(s.ground_truth)
    replies = [
        "```python\ndef run_experiment(system, tools):\n    tools.log_experiment(tools.extract_subgraph('attribution', 5))\n```",
        f"```python\ndef run_experiment(system, tools):\n    tools.log_experiment(tools.test_circuit({s.ground_truth!r}))\n```",
        f"[CIRCUIT]: {truth}\n[EXPLANATION]: two units compute x0-x1 and x1-x0, one sums them",
    ]
    bb = ScriptedBackbone(replies, supports_vision=False)
    tools = CircuitTools(s)
    agent = MAIAAgent(bb, s, tools, load_prompt("circuit_api"),
                      load_prompt("circuit_discovery", task=s.task_description, n_nodes=len(s.nodes)),
                      final_tags=("CIRCUIT", "EXPLANATION"), required_tag="CIRCUIT")
    res = agent.run()
    nodes = [n.strip() for n in res.fields["CIRCUIT"].split(",")]
    assert circuit_overlap(nodes, s.ground_truth)["f1"] == 1.0
    assert "faithfulness" in str(bb.seen[2][-1]["content"])
