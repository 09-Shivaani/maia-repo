"""MAIA: Multimodal Automated Interpretability Agent.

Reproduction of Shaham, Schwettmann et al. (ICML 2024) plus two extensions:
  * MAIA-NLP      (maia.nlp)      - interpretability agent for language-model neurons / heads
  * MAIA-Circuit  (maia.circuit)  - causal circuit discovery with ablation / intervention tools
"""

__version__ = "0.1.0"

from maia.config import load_config  # noqa: F401
