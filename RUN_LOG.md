# RUN_LOG — MAIA reproduction (reduced plan, CPU-only machine)

## Stage 0 — machine inspection (2026-11-05)

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 (Git Bash / PowerShell) |
| CPU | 13th Gen Intel Core i7-1355U, 10 cores / 12 threads |
| RAM | 15.7 GB |
| GPU | **None NVIDIA** — Intel Iris Xe (integrated); `nvidia-smi` not found |
| Free disk | 379 GB on C: |
| Python | 3.13.7 default; 3.11 available via `py -3.11` |
| git | 2.53.0 |
| conda | not installed (using venv) |
| Ollama | 0.40.11 (server running) |

**Decision:** no NVIDIA GPU → following the REDUCED PLAN: stages 0-2, MAIA-Circuit synthetic +
`mib --lm gpt2`, MAIA-NLP on Qwen/Qwen2.5-0.5B with 10 neurons, base paper on 5 resnet152 neurons.
Everything runs on CPU (torch CPU build). Qwen2.5-VL-7B runs on CPU via Ollama (slow).

Note: `C:\Users\shivz\Downloads\maia-repo` was an un-versioned copy (CRLF line endings only differed)
sitting inside an unrelated parent git repo, so a fresh clone was made at
`C:\Users\shivz\Downloads\maia-repo-git` and all work happens there.

## Stage 1 — setup

```
git clone https://github.com/09-Shivaani/maia-repo.git maia-repo-git
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env          # .env is git-ignored (.gitignore:2); keys filled by the user
ollama pull qwen2.5vl:7b      # < 48 GB VRAM (no GPU) -> 7b
```

Result: torch 2.15.0+cpu, torchvision 0.29.2+cpu, transformers 5.22.1, diffusers 0.42.1; `pip check` clean.
(pip hit intermittent DNS failures to files.pythonhosted.org but its retries succeeded.)

**Ollama context length.** Ollama serves models with a 4096-token context by default, and its
OpenAI-compatible endpoint ignores `num_ctx`. The vision API prompt (~2k tokens) plus 15 exemplar
images would be silently truncated, so a 32k-context tag was created and used for both vision roles:
```
printf 'FROM qwen2.5vl:7b\nPARAMETER num_ctx 32768\n' > Modelfile && ollama create qwen2.5vl:7b-32k -f Modelfile
# all vision runs: --set backbones.vision_agent.model=qwen2.5vl:7b-32k backbones.describer.model=qwen2.5vl:7b-32k
```
git identity for commits set repo-locally to the existing repo author (D Shivaani).
All commands are run with `PYTHONUTF8=1` in addition to the encoding fix below.

## Stage 2 — smoke tests

| Test | Result |
|---|---|
| `pytest -q` | **22 passed** in 435 s (CPU) |
| Qwen-VL `describer` call with an image (red circle / blue sky / green stripe test image) | OK — "a simple flag with a red circle in the center on a blue background, with a green stripe at the bottom." 273 s cold (model load), 13.5 s warm |
| Nemotron `text_agent` call | _pending: NVIDIA_API_KEY not yet filled in .env_ |
| `run_circuits.py synthetic --seeds 0 --no-agent` (baselines only, written to outputs/noagent/) | OK in 20 s. F1 correlation / attribution / greedy: xor 0.86/1.00/1.00, and_or 0.55/1.00/0.86, parity3 0.15/1.00/1.00 |

### Bugs fixed
1. `5fcca45` — text files read/written without an encoding; on Windows that means cp1252, which garbles
   `maia/prompts/nlp_api.txt` (non-ASCII) and other UTF-8 data. Now `encoding="utf-8"` everywhere.
2. `c3331d4` — `download_datasets.py` ran `python -m img2dataset`, which fails (no `__main__`), so
   CC3M never downloaded. Now calls `img2dataset.main()`.

## Stage 3 — data (reduced plan: only what the reduced experiments need)

```
python scripts/download_datasets.py --bold --stereoset --concepts --nltk --text-corpus --mib
python scripts/download_datasets.py --cc3m 5000 --processes 4 --threads 32
```
| Dataset | Result |
|---|---|
| BOLD, StereoSet dev, CLIP-Dissect 20k concepts, WordNet, MIB IOI | OK |
| WikiText-103 corpus | 200,000 paragraphs |
| CC3M | **3,154 / 5,000** images (rest dead URLs) |
| Spawrious, CC3M 130k, Bills explanations | skipped (not needed for the reduced plan) |

Sampling: `sample_neurons.py --models resnet152` → 100 units, 25 in the human subset.
`run_nlp.py sample --model Qwen/Qwen2.5-0.5B --n 10` → 10 MLP neurons
(L20/912, L0/2253, L7/1828, L4/839, L21/4467, L2/4837, L13/260, L0/767, L6/1905, L16/217).
The 5 base-paper resnet152 neurons = first human-subset unit per layer:
conv1/1, layer1/119, layer2/432, layer3/466, layer4/1328.
