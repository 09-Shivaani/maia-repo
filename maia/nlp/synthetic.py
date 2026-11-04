"""Synthetic language neurons with ground-truth selectivity (text analogue of paper Sec. 4.2).

A token activates a synthetic neuron if its lemma belongs to the WordNet hyponym closure of a concept
synset (e.g. animal.n.01). The neuron's activation on a text is the max over tokens. Like the vision
version there are monosemantic, polysemantic (A OR B) and conditional (A|B) neurons.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

from maia.nlp.system import highlight

MONO = ["animal.n.01", "food.n.02", "vehicle.n.01", "color.n.01", "emotion.n.01", "body_part.n.01",
        "musical_instrument.n.01", "building.n.01", "clothing.n.01", "tool.n.01", "plant.n.02", "weapon.n.01",
        "beverage.n.01", "sport.n.01", "disease.n.01", "metal.n.01", "furniture.n.01", "weather.n.01",
        "time_period.n.01", "relative.n.01"]
POLY = [("animal.n.01", "vehicle.n.01"), ("food.n.02", "clothing.n.01"), ("color.n.01", "emotion.n.01"),
        ("musical_instrument.n.01", "weapon.n.01"), ("body_part.n.01", "building.n.01"),
        ("beverage.n.01", "sport.n.01"), ("plant.n.02", "metal.n.01"), ("disease.n.01", "furniture.n.01")]
COND = [("animal.n.01", "food.n.02"), ("vehicle.n.01", "color.n.01"), ("body_part.n.01", "disease.n.01"),
        ("clothing.n.01", "color.n.01"), ("beverage.n.01", "time_period.n.01"), ("sport.n.01", "relative.n.01")]

_STOP = {"it", "one", "can", "may", "will", "do", "have", "be", "is", "as", "an", "a", "i", "us", "we", "he", "she"}


def _name(syn: str) -> str:
    return syn.split(".")[0].replace("_", " ")


@lru_cache(maxsize=None)
def lexicon(synset: str) -> frozenset:
    from nltk.corpus import wordnet as wn

    root = wn.synset(synset)
    members = {root} | set(root.closure(lambda x: x.hyponyms()))
    words = set()
    for s in members:
        for l in s.lemma_names():
            w = l.lower()
            if "_" in w or len(w) <= 2 or w in _STOP:
                continue
            # keep a word only if its most frequent noun sense lies in the concept (e.g. drop "dog" from
            # food, which is there via the frankfurter sense)
            senses = wn.synsets(w, pos=wn.NOUN)
            if senses and senses[0] in members:
                words.add(w)
    return frozenset(words)


@lru_cache(maxsize=100000)
def _lemma(word: str) -> str:
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer().lemmatize(word.lower())


@dataclass(frozen=True)
class TextSyntheticSpec:
    kind: str
    synsets: tuple

    @property
    def label(self) -> str:
        n = [_name(s) for s in self.synsets]
        return n[0] if self.kind == "mono" else (f"{n[0]} OR {n[1]}" if self.kind == "or" else f"{n[0]}|{n[1]}")

    @property
    def gt_description(self) -> str:
        n = [_name(s) for s in self.synsets]
        if self.kind == "mono":
            return f"words for {n[0]}s"
        if self.kind == "or":
            return f"words for {n[0]}s OR words for {n[1]}s"
        return f"words for {n[0]}s, only in texts that also mention {n[1]}s"


def all_text_synthetic() -> List[TextSyntheticSpec]:
    return ([TextSyntheticSpec("mono", (s,)) for s in MONO] + [TextSyntheticSpec("or", p) for p in POLY]
            + [TextSyntheticSpec("cond", p) for p in COND])


class SyntheticTextNeuronSystem:
    synthetic = True

    def __init__(self, spec: TextSyntheticSpec):
        self.spec = spec
        self.model_name, self.layer, self.neuron_num, self.kind = f"synthetic:{spec.label}", 0, 0, "synthetic"
        self.lexicons = [lexicon(s) for s in spec.synsets]

    def _token_acts(self, text: str) -> Tuple[List[str], List[float]]:
        toks = re.findall(r"\s*\S+", text)
        hits = [[(_lemma(re.sub(r"[^\w]", "", t)) in lx) for t in toks] for lx in self.lexicons]
        if self.spec.kind == "mono":
            acts = [1.0 if h else 0.0 for h in hits[0]]
        elif self.spec.kind == "or":
            acts = [1.0 if (a or b) else 0.0 for a, b in zip(*hits)]
        else:
            cond = any(b and not a for a, b in zip(*hits))  # the condition must be a different token
            acts = [1.0 if (a and cond) else 0.0 for a in hits[0]]
        return toks, acts

    def neuron(self, text_list):
        if isinstance(text_list, str):
            text_list = [text_list]
        acts, marked = [], []
        for t in text_list:
            toks, a = self._token_acts(t)
            acts.append(max(a) if a else 0.0)
            marked.append(highlight(toks, a))
        return acts, marked

    def token_activations(self, text: str):
        return list(zip(*self._token_acts(text)))
