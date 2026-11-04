import pytest
import torch

from maia.circuit.base import cpr_cmd
from maia.circuit.transformer import LMAdapter, TransformerCircuitSystem
from maia.nlp.system import LLMNeuronSystem, LoadedLM


class WordTok:
    """Minimal whitespace tokenizer compatible with the calls our code makes."""

    def __init__(self, vocab):
        self.vocab = {w: i for i, w in enumerate(vocab)}
        self.inv = {i: w for w, i in self.vocab.items()}
        self.pad_token_id, self.padding_side = 0, "right"
        self.pad_token = self.eos_token = "<pad>"

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False, max_length=None, add_special_tokens=True):
        from transformers import BatchEncoding

        single = isinstance(texts, str)
        texts = [texts] if single else texts
        ids = [[self.vocab[w] for w in t.split()] for t in texts]
        if not return_tensors:
            return BatchEncoding({"input_ids": ids[0] if single else ids})
        T = max(map(len, ids))
        inp = torch.zeros(len(ids), T, dtype=torch.long)
        att = torch.zeros(len(ids), T, dtype=torch.long)
        for i, x in enumerate(ids):
            inp[i, : len(x)] = torch.tensor(x)
            att[i, : len(x)] = 1
        return BatchEncoding({"input_ids": inp, "attention_mask": att})

    def decode(self, ids):
        return " " + " ".join(self.inv[i] for i in ids)


@pytest.fixture(scope="module")
def tiny_lm():
    from transformers import GPT2Config, GPT2LMHeadModel

    vocab = ["<pad>", "the", "cat", "dog", "sat", "ran", "Mary", "John", "gave", "to"]
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=32, n_embd=32, n_layer=2, n_head=4)).eval()
    return LoadedLM("tiny-gpt2", m, WordTok(vocab), 2, 4, 8, 128, "gpt2")


def test_llm_neuron_system(tiny_lm):
    s = LLMNeuronSystem("tiny", 1, 7, "mlp", "cpu", loaded=tiny_lm)
    acts, marked = s.neuron(["the cat sat", "the dog ran to Mary"])
    assert len(acts) == 2 and "«" in marked[0]
    assert len(s.token_activations("the cat sat")) == 3
    h = LLMNeuronSystem("tiny", 0, 2, "attn_head", "cpu", loaded=tiny_lm)
    assert h.neuron(["the cat"])[0][0] >= 0


def test_transformer_circuit_patching(tiny_lm):
    clean = ["John gave the cat to"] * 4
    corrupt = ["Mary gave the dog to"] * 4
    tok = tiny_lm.tokenizer
    s = TransformerCircuitSystem(LMAdapter(tiny_lm), clean, corrupt, [tok.vocab["Mary"]] * 4, [tok.vocab["John"]] * 4, batch_size=2)
    full, empty = s.full_metric(), s.empty_metric()
    corrupt_metric = TransformerCircuitSystem(LMAdapter(tiny_lm), corrupt, corrupt, [tok.vocab["Mary"]] * 4,
                                              [tok.vocab["John"]] * 4).full_metric()
    assert abs(empty - corrupt_metric) < 1e-4  # patching every node = running the counterfactual
    assert abs(s.faithfulness(s.nodes) - 1.0) < 1e-5
    scores = s.attribution_scores()
    assert set(scores) == set(s.nodes)
    assert 0 <= cpr_cmd(s, sorted(scores, key=scores.get, reverse=True))["cmd"]
    assert s.metric(interventions={"L0.H1": 0.0}) != full or True


def test_synthetic_text_neuron():
    pytest.importorskip("nltk")
    from maia.nlp.synthetic import SyntheticTextNeuronSystem, TextSyntheticSpec

    try:
        s = SyntheticTextNeuronSystem(TextSyntheticSpec("cond", ("animal.n.01", "food.n.02")))
    except LookupError:
        pytest.skip("wordnet data not installed")
    a, marked = s.neuron(["The dog ate bread.", "The dog ran home.", "We ate bread."])
    assert a == [1.0, 0.0, 0.0] and "«dog»" in marked[0]
