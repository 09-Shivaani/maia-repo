"""Tools class for language-model experiments (MAIA-NLP)."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Tuple, Union

from maia.agent import ExperimentLogMixin
from maia.backbone import Backbone

log = logging.getLogger(__name__)


def _json_list(text: str, n: int) -> List[str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            vals = json.loads(m.group(0))
            return [str(v) for v in vals][:n]
        except json.JSONDecodeError:
            pass
    lines = [re.sub(r"^\s*(\d+[.)]|-)\s*", "", l).strip().strip('"') for l in text.splitlines() if l.strip()]
    return lines[:n]


class NLPTools(ExperimentLogMixin):
    def __init__(self, writer: Backbone, describer: Backbone | None = None, corpus: List[str] | None = None,
                 exemplar_cache: str | None = None, k: int = 15, allowed: set[str] | None = None,
                 bold=None, stereoset=None):
        """
        writer      LM that writes / rewrites text for generate_texts, text_contrastive, semantic_perturbation
        describer   independent LM (fresh context) for describe/summarize tools
        corpus      real text corpus for dataset_exemplars
        bold, stereoset  optional loaded datasets exposed as tools for the bias task
        """
        self._init_log()
        self.writer = writer
        self.describer = describer or writer
        self.corpus, self.exemplar_cache, self.k, self.allowed = corpus, exemplar_cache, k, allowed
        self._bold, self._stereo = bold, stereoset

    def _check(self, name):
        if self.allowed is not None and name not in self.allowed:
            raise RuntimeError(f"The tool '{name}' is not available in this experiment.")

    # -- real data ----------------------------------------------------------------------
    def dataset_exemplars(self, system) -> Tuple[List[float], List[str]]:
        self._check("dataset_exemplars")
        import numpy as np

        from maia.nlp.exemplars import compute_text_exemplars

        if getattr(system, "synthetic", False):
            import random

            pool = random.Random(0).sample(self.corpus, min(20000, len(self.corpus)))
            acts, _ = system.neuron(pool)
            top = sorted(range(len(pool)), key=lambda i: -acts[i])[: self.k]
            a, marked = system.neuron([pool[i] for i in top])
            self.activation_threshold = sum(a) / len(a)
            return a, marked
        lm = system.lm
        path = Path(self.exemplar_cache or "outputs/text_exemplars") / lm.name.replace("/", "_") / f"{system.kind}_L{system.layer}.npz"
        if path.exists():
            d = np.load(path)
            idx = d["idx"]
        else:
            if not self.corpus:
                raise RuntimeError("No text corpus configured for dataset_exemplars.")
            idx = compute_text_exemplars(lm, system.layer, self.corpus, system.kind, self.k, device=system.device, out_path=path)["idx"]
        texts = [self.corpus[i] for i in idx[system.neuron_num][: self.k]]
        acts, marked = system.neuron(texts)
        self.activation_threshold = sum(acts) / len(acts)
        return acts, marked

    # -- synthetic text ----------------------------------------------------------------
    def generate_texts(self, prompt_list: Union[str, List[str]]) -> List[str]:
        """Write one short passage per instruction (text analogue of text2image)."""
        self._check("generate_texts")
        if isinstance(prompt_list, str):
            prompt_list = [prompt_list]
        out = []
        for p in prompt_list:
            t = self.writer.complete(
                f"Write one natural English passage of 1-2 sentences that matches this description: {p}\n"
                "Output only the passage.", temperature=0.7, max_tokens=2048)
            out.append(t.strip().strip('"'))
        return out

    def text_contrastive(self, text_list: List[str], edit_instructions: List[str]) -> Tuple[List[str], List[str]]:
        """Counterfactual pairs: minimally rewrite each text per its instruction.
        Returns [orig_1, edited_1, orig_2, edited_2, ...] and matching titles."""
        self._check("text_contrastive")
        if isinstance(text_list, str):
            text_list = [text_list]
        if isinstance(edit_instructions, str):
            edit_instructions = [edit_instructions] * len(text_list)
        texts, titles = [], []
        for t, e in zip(text_list, edit_instructions):
            new = self.writer.complete(
                f"Text: {t}\nEdit instruction: {e}\nApply the edit with the smallest possible change and keep everything "
                "else identical. Output only the edited text.", temperature=0.2, max_tokens=2048).strip().strip('"')
            texts += [t, new]
            titles += ["original", e]
        return texts, titles

    def semantic_perturbation(self, text_list: List[str], attribute: str, target: str) -> List[str]:
        """Rewrite texts so that `attribute` (sentiment, topic, gender of subject, formality, tense,
        language, ...) takes the value `target`, preserving other content."""
        self._check("semantic_perturbation")
        if isinstance(text_list, str):
            text_list = [text_list]
        return [self.writer.complete(
            f"Rewrite the text so that its {attribute} becomes: {target}. Keep length, structure and all other "
            f"content the same.\nText: {t}\nOutput only the rewritten text.", temperature=0.2, max_tokens=2048).strip().strip('"')
            for t in text_list]

    def token_activation_map(self, system, text_list: List[str]) -> str:
        """Per-token activation values, formatted as token(value) for every text."""
        self._check("token_activation_map")
        if isinstance(text_list, str):
            text_list = [text_list]
        rows = []
        for t in text_list:
            pairs = system.token_activations(t)
            rows.append(" ".join(f"{tok.strip() or repr(tok)}({v:.2f})" for tok, v in pairs))
        return "\n".join(rows)

    def describe_texts(self, text_list: List[str], titles: List[str] | None = None) -> str:
        self._check("describe_texts")
        titles = titles or [f"text {i}" for i in range(len(text_list))]
        out = []
        for t, ti in zip(text_list, titles):
            d = self.describer.complete(
                "Tokens marked «like this» strongly activate a neuron. In one sentence, describe what the marked tokens "
                f"are and their role in context.\n\n{t}", temperature=0.0, max_tokens=1024)
            out.append(f'"{ti}": {d.strip()}')
        return "\n".join(out)

    def summarize_texts(self, text_list: List[str]) -> str:
        self._check("summarize_texts")
        joined = "\n---\n".join(text_list)
        return self.describer.complete(
            "In each passage, tokens marked «like this» strongly activate the same neuron. In one or two sentences, "
            f"state what the marked tokens have in common (lexical, syntactic, semantic, positional).\n\n{joined}",
            temperature=0.0, max_tokens=1024).strip()

    # -- bias datasets ------------------------------------------------------------------
    def bold_prompts(self, domain: str, group: str | None = None, n: int = 10, seed: int = 0) -> List[str]:
        """Real prompts from BOLD (Dhamala et al. 2021). domains: gender, race, religious_ideology,
        political_ideology, profession. Call bold_groups(domain) for group names."""
        self._check("bold_prompts")
        if self._bold is None:
            raise RuntimeError("BOLD not loaded")
        return self._bold.sample(domain, group, n, seed)

    def bold_groups(self, domain: str) -> List[str]:
        self._check("bold_prompts")
        return self._bold.groups(domain)

    def stereoset_examples(self, bias_type: str, n: int = 10, seed: int = 0) -> List[dict]:
        """Real StereoSet intrasentence items (bias_type: gender, profession, race, religion), each with
        'context', 'stereotype', 'anti_stereotype', 'unrelated' and 'target'."""
        self._check("stereoset_examples")
        if self._stereo is None:
            raise RuntimeError("StereoSet not loaded")
        return self._stereo.sample(bias_type, n, seed)

    # -- log ----------------------------------------------------------------------------
    def log_experiment(self, activation_list, text_list, titles=None, textual_information=None):
        if not isinstance(activation_list, (list, tuple)):
            activation_list = [activation_list]
        if isinstance(text_list, str):
            text_list = [text_list]
        titles = list(titles) if titles else [f"text {i}" for i in range(len(text_list))]
        titles += [f"text {i}" for i in range(len(titles), len(text_list))]
        for a, t, ti in zip(activation_list, text_list, titles):
            self._log_text(f'[{ti}] activation: {a}\n    {t}')
            self.results_list.append({ti: {"activation": a, "text": t}})
        if textual_information:
            info = textual_information
            self._log_text(info if isinstance(info, str) else "\n".join(map(str, info)))
        if self.activation_threshold is not None and activation_list:
            try:
                if max(float(a) for a in activation_list) < self.activation_threshold:
                    self._log_text(f"Note: max activation here is below the mean exemplar activation "
                                   f"({self.activation_threshold:.2f}); keep searching for stronger inputs.")
            except (TypeError, ValueError):
                pass

    save_experiment_log = log_experiment
