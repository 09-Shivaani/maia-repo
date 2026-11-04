from maia.agent import ExperimentLogMixin, MAIAAgent, execute_code, extract_code, parse_fields
from maia.backbone import ScriptedBackbone, cap_images, strip_images


class EchoTools(ExperimentLogMixin):
    def __init__(self):
        self._init_log()

    def log_experiment(self, activation_list, texts, titles=None, info=None):
        for a, t in zip(activation_list, texts):
            self._log_text(f"{t}: {a}")


class FakeSystem:
    def neuron(self, xs):
        return [float(len(x)) for x in xs], list(xs)


CODE_REPLY = """[HYPOTHESIS LIST]:
Hypothesis_1: long strings
[CODE]:
```python
def run_experiment(system, tools):
    acts, outs = system.neuron(["a", "abcd"])
    tools.log_experiment(acts, outs)
```"""


def test_extract_and_parse():
    assert "def run_experiment" in extract_code(CODE_REPLY)
    f = parse_fields("[DESCRIPTION]: dogs on grass\n[LABEL]: dogs", ("LABEL", "DESCRIPTION"))
    assert f == {"DESCRIPTION": "dogs on grass", "LABEL": "dogs"}


def test_execute_code_errors_are_reported():
    r = execute_code("def run_experiment(system, tools):\n    1/0\n", None, None)
    assert not r.ok and "ZeroDivisionError" in r.error
    r = execute_code("x = 1", None, None)
    assert not r.ok and "run_experiment" in r.error


def test_agent_loop_feeds_results_back():
    bb = ScriptedBackbone([CODE_REPLY, "[DESCRIPTION]: string length\n[LABEL]: long strings"])
    agent = MAIAAgent(bb, FakeSystem(), EchoTools(), "API", "task")
    res = agent.run()
    assert res.finished and res.fields["LABEL"] == "long strings" and res.rounds == 2
    fed_back = bb.seen[1][-1]["content"]
    text = fed_back if isinstance(fed_back, str) else " ".join(p.get("text", "") for p in fed_back)
    assert "abcd: 4.0" in text


def test_agent_reports_execution_error_then_recovers():
    bad = "```python\ndef run_experiment(system, tools):\n    raise ValueError('boom')\n```"
    bb = ScriptedBackbone([bad, "[LABEL]: x"])
    res = MAIAAgent(bb, FakeSystem(), EchoTools(), "API", "task").run()
    assert res.fields["LABEL"] == "x"
    last_user = bb.seen[1][-1]["content"]
    assert "boom" in (last_user if isinstance(last_user, str) else str(last_user))


def test_image_stripping_and_capping():
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}, img, img, img]}]
    assert "[image omitted]" in strip_images(msgs)[0]["content"]
    capped = cap_images(msgs, 1)
    assert sum(p["type"] == "image_url" for p in capped[0]["content"]) == 1
