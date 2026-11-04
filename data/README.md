# Data

Everything here is downloaded by `python scripts/download_datasets.py` and is git-ignored.

| Folder | Source | Used for |
|---|---|---|
| `cc3m/images/` | CC3M (Sharma et al. 2018), random train subset via img2dataset | real-image pool for `dataset_exemplars` (replaces ImageNet val), synthetic-neuron exemplars, CLIP-Dissect probe set |
| `spawrious/spawrious224/` | Spawrious O2O-Easy (Lynch et al. 2023), official tarball | spurious-feature removal (Table 2), ViT circuit task |
| `bold/` | BOLD (Dhamala et al. 2021), github.com/amazon-science/bold | MAIA-NLP bias discovery and ground-truth group gaps |
| `stereoset/dev.json` | StereoSet (Nadeem et al. 2021) | MAIA-NLP bias tools, stereotype score |
| `text/wikitext103_train.jsonl` | WikiText-103 (Merity et al. 2016) | real corpus for LLM dataset exemplars |
| `concepts/20k.txt` | CLIP-Dissect concept set | CLIP-Dissect baseline |
| `bills_explanations/` | Bills et al. (2023) GPT-2 XL neuron explanations | optional reference labels for MAIA-NLP |
| HF cache | MIB IOI (Mueller et al. 2025), `mib-bench/ioi` | MAIA-Circuit on real LMs |

ImageNet is not used: per your request the exemplar tool runs on CC3M instead.
