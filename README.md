# MAIA reproduction + MAIA-NLP + MAIA-Circuit

Implementation of **"A Multimodal Automated Interpretability Agent"** (Rott Shaham, Schwettmann et al., ICML 2024) and two extensions:

* **MAIA-NLP**: the same agent loop applied to language-model neurons, attention heads and model behaviour, with NLP tools (counterfactual text, token activation maps, semantic perturbation) and bias discovery on BOLD / StereoSet.
* **MAIA-Circuit**: circuit discovery with ablation, intervention and causal-effect tools. It is validated on networks with planted ground-truth circuits, on the MIB circuit-localization IOI task (GPT-2, Qwen2.5-0.5B) and on a ViT-B/16 classifier.

The agent backbones are **local Qwen models** (Ollama, vLLM or transformers) and **NVIDIA API models** (Nemotron). There are no GPT-4V dependencies.

---

## 1. How MAIA works here

```
user task prompt ──► backbone LM ──► reply with [HYPOTHESIS LIST] + ```python run_experiment(system, tools)```
                         ▲                                     │ executed (maia/agent.py)
                         └──── logged results (text + images) ◄┘   tools.log_experiment(...)
stop when the reply has the final tag ([LABEL] / [BIAS] / [CIRCUIT]) and no code
```

| Paper component | Code |
|---|---|
| System API: unit activations + masked images (Sec. 3.1, App. A) | `maia/vision/system.py` (`VisionNeuronSystem`, `ClassifierSystem`) |
| Tools: `dataset_exemplars`, `text2image` (SD-v1.5), `edit_images` (InstructPix2Pix), `describe_images`, `summarize_images`, `log_experiment` (Sec. 3.2) | `maia/vision/tools.py`, `maia/vision/generation.py`, `maia/vision/exemplars.py` |
| Task prompts (App. B, F4, G) | `maia/prompts/*.txt` |
| Vision models ResNet-152 / DINO ViT-B/16 / CLIP-RN50, Table A3 layer sampling | `maia/vision/models.py`, `scripts/sample_neurons.py` |
| Predictive evaluation (Sec. 4.1, Fig. 3/4, Table A3/A4) | `maia/eval/predictive.py`, `scripts/evaluate_predictive.py` |
| Synthetic neurons, Grounding-DINO + SAM, Table A6 (Sec. 4.2) | `maia/vision/synthetic.py`, `scripts/run_synthetic_neurons.py` |
| 2AFC (Table 1), human-utility study (Table A7), expert labels (App. C3) | `maia/eval/human.py`, `scripts/human_session.py` |
| Tool ablations (Sec. 4.3, Fig. 7, Table A5) | `--ablation real_only / synthetic_only`, `--t2i sdxl / dalle3` |
| Spurious-feature removal on Spawrious (Sec. 5.1, Table 2, App. F) | `maia/spurious.py`, `scripts/run_spurious.py` |
| Bias identification in ResNet-152 classes (Sec. 5.2, Fig. 10) | `scripts/run_bias.py` |
| Baselines: MILAN, "MILAN (GPT-4V)" one-shot exemplar labels, CLIP-Dissect | `maia/baselines.py`, `scripts/run_baselines.py` |

### Deviations from the paper (read before comparing numbers)

1. **Backbone.** The paper uses GPT-4V. Here the vision agent is a local **Qwen2.5-VL** model, and the strong text model is **Nemotron 3 Ultra** via the NVIDIA API. The paper reports that open VLMs lag GPT-4V (App. D3), so expect lower absolute scores. `configs/nemotron_agent.yaml` lets Nemotron drive vision tasks with Qwen-VL describing the images.
2. **No ImageNet.** `dataset_exemplars` searches a **CC3M** subset (about 100k real images) instead of the ImageNet validation set.
3. **Evaluation image generator.** The paper renders evaluation prompts with DALL-E 3. The default here is SDXL. Set `evaluation.text2image.backend=dalle3` and `OPENAI_API_KEY` to match the paper exactly.
4. **Table A6 lists 87 synthetic neurons** (45 + 24 + 18). The paper's text says 85. All listed rows are used.
5. **MILAN** needs its official repo (`github.com/evandez/neuron-descriptions`). Without it, the one-shot exemplar baseline (`vlm_exemplars`) plays the role of the paper's "MILAN (GPT-4V)" row.
6. **Human data.** The expert-label column, the MTurk 2AFC and the utility studies need people. Scripts produce the task files and analyze the results. An LM-judge 2AFC proxy is also included and is labeled as a proxy.
7. **Spawrious balanced sets.** 20% of the test-distribution images are held out as the pool for the balanced validation sets, so the balanced baselines never see their own test images.

---

## 2. Setup

```bash
git clone https://github.com/09-Shivaani/maia-repo.git && cd maia-repo
conda env create -f environment.yml && conda activate maia      # or: pip install -r requirements.txt
cp .env.example .env    # then put your NVIDIA key(s) in .env (git-ignored)
```

**Local Qwen (default vision agent + describer):**
```bash
ollama pull qwen2.5vl:7b          # default; qwen2.5vl:32b / 72b are stronger if you have the VRAM
ollama serve
```
Alternatives: `--config configs/qwen_vllm.yaml` for a vLLM server, `--config configs/qwen_hf.yaml` to run in-process with transformers, or `--config configs/local_qwen.yaml` for Qwen-only with no API calls.

**Backbone roles** (`configs/default.yaml → backbones`):

| Role | Default | Used by |
|---|---|---|
| `vision_agent` | Ollama `qwen2.5vl:7b` | MAIA on vision tasks |
| `describer` | Ollama `qwen2.5vl:7b` | `describe_images`, `summarize_images`, one-shot exemplar baseline |
| `text_agent` | NVIDIA `nvidia/nemotron-3-ultra-550b-a55b` (`NVIDIA_API_KEY`) | MAIA-NLP, MAIA-Circuit, prompt generator/selector, judge |
| `fast_text` | NVIDIA `nvidia/nemotron-3.5-bolt-e4b` (`NVIDIA_API_KEY_BOLT`, falls back to `NVIDIA_API_KEY`) | text-rewriting tools |

Override any field from the command line: `--set backbones.vision_agent.model=qwen2.5vl:32b`.

**Hardware.** A 24 GB GPU runs Qwen2.5-VL-7B (via Ollama), SD-v1.5 and InstructPix2Pix together. Grounding-DINO + SAM-H adds about 6 GB. SDXL for evaluation runs as a separate step.

## 3. Data (all real, nothing synthetic-stubbed)

```bash
python scripts/download_datasets.py --all --cc3m 130000      # or pick: --spawrious --bold --stereoset --text-corpus --concepts --nltk --mib
```
See `data/README.md` for the sources. Spawrious and CC3M are large: plan on 50+ GB.

## 4. Running

`scripts/run_all.sh` runs everything in order, and every step caches its results. Key steps:

```bash
# base paper
python scripts/compute_exemplars.py --models resnet152 dino clip-rn50
python scripts/sample_neurons.py
python scripts/run_neuron_descriptions.py --model resnet152 --layer layer4 --unit 122   # one neuron
python scripts/run_neuron_descriptions.py --model resnet152                             # all 100 sampled
python scripts/run_baselines.py --model resnet152 --method vlm_exemplars
python scripts/evaluate_predictive.py --model resnet152 --methods full vlm_exemplars clip_dissect human
python scripts/run_synthetic_neurons.py --method maia
python scripts/run_spurious.py
python scripts/run_bias.py --classes "Labrador retriever" "beer glass" suit flagpole
# extensions
python scripts/run_nlp.py sample --model Qwen/Qwen2.5-0.5B && python scripts/run_nlp.py maia --model Qwen/Qwen2.5-0.5B
python scripts/run_nlp.py bias --model Qwen/Qwen2.5-0.5B
python scripts/run_circuits.py synthetic
python scripts/run_circuits.py mib --lm gpt2
python scripts/verify_reproduction.py
```
Outputs: `outputs/descriptions/<method>/<model>/<layer>_<unit>/` (result + full agent transcript), `outputs/tables/*.json`.

Expert labels (paper App. C3): open JupyterLab and use `scripts/human_session.py`. Labels land in the `human` method folder.

---

## 5. Extension 1 — MAIA-NLP

| | |
|---|---|
| Systems | `LLMNeuronSystem` (MLP neuron = input to the down-projection; attention head = norm of its output slice) and `LLMBehaviorSystem` (VADER sentiment of continuations, the BOLD metric, or the log-probability of a target string). Works with GPT-2, Qwen2.5/3, Llama-style models. |
| Tools | `dataset_exemplars` (WikiText-103), `generate_texts`, `text_contrastive` (minimal counterfactual edits), `semantic_perturbation`, `token_activation_map`, `describe_texts` / `summarize_texts` (independent LM), `bold_prompts`, `stereoset_examples` |
| Ground truth | 34 **synthetic text neurons** built from WordNet hyponym closures (monosemantic / OR / conditional, like Sec. 4.2). Each word is filtered by its first sense to avoid polysemy leaks. |
| Evaluation | predictive evaluation (same protocol, text modality); LM-judge 2AFC on synthetic neurons; **bias precision / recall**, where the agent's [BIAS] claims are matched against BOLD group-sentiment gaps with bootstrap significance (the standard-fairness-metric baseline) |
| Ablations | `--toolset exemplars_only / no_counterfactual / synthetic_only` |

The outline's "NeuroExplain" benchmark couldn't be found as a public dataset. In its place: synthetic text neurons for ground truth, plus the public GPT-2 XL explanations of Bills et al. (2023) as reference labels (`--bills-explanations`).

## 6. Extension 2 — MAIA-Circuit

| | |
|---|---|
| Systems | `SyntheticCircuitSystem`: 2-hidden-layer MLPs with **planted circuits** for xor, (a∧b)∨c and 3-parity, plus *decoy* units that copy a circuit unit's computation but write elsewhere (perfectly correlated, causally irrelevant). `TransformerCircuitSystem`: node-level (heads + MLPs) circuits in HF LMs on **MIB IOI** (the outline's "CircuitBench" corresponds to MIB, Mueller et al. 2025) and in a ViT-B/16 classifier. |
| Tools | `ablate`, `ablate_each`, `intervene`, `measure_causal_effect`, `extract_subgraph` (attribution patching / correlation), `node_activation_stats`, `test_circuit` (faithfulness + necessity) |
| Metrics | MIB faithfulness f(C) = (m(C) − m(∅)) / (m(N) − m(∅)), **CPR / CMD** over the MIB size grid, precision / recall / F1 / IoU against planted circuits |
| Baselines | correlation ranking, one-shot attribution patching (EAP, node level), ACDC-style greedy pruning |

Baseline results on the planted networks (from `run_circuits.py synthetic --no-agent`, 3 seeds; F1 at the true circuit size): correlation 0.67 / 0.50 / 0.40 (xor / and_or / parity3); attribution 1.00 / 1.00 / 1.00; greedy pruning 1.00 / 0.86 / 1.00. The decoys fool correlation as intended. Attribution patching is exact on these piecewise-linear nets, so the harder test for the agent is MIB and the ViT, where the linear approximation is known to break down.

---

## 7. Tests

```bash
pytest -q        # 22 tests, CPU-only, no downloads
```
The tests cover the agent loop (code extraction, execution errors fed back), vision activations against a manual hook, masking, tool ablation, the predictive protocol, l1 sparsity search, planted-circuit exactness, baselines and CPR/CMD, transformer activation patching (patching all nodes equals running the counterfactual), LLM neuron systems and synthetic text neurons. One test also checks that no API key is committed.

## 8. Compute (rough)

The full base reproduction is 300 neurons × about 10 agent rounds × image generation, plus evaluation. Expect several GPU-days on one 24 GB GPU with local Qwen. API usage is the Nemotron text roles only. Start with `--subset human` (25 neurons per model) to sanity-check.

## Citation

```
@inproceedings{shaham2024maia, title={A Multimodal Automated Interpretability Agent}, author={Rott Shaham, Tamar and Schwettmann, Sarah and Wang, Franklin and Rajaram, Achyuta and Hernandez, Evan and Andreas, Jacob and Torralba, Antonio}, booktitle={ICML}, year={2024}}
@inproceedings{mueller2025mib, title={MIB: A Mechanistic Interpretability Benchmark}, author={Mueller, Aaron and others}, booktitle={ICML}, year={2025}}
```
