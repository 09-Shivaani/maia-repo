#!/usr/bin/env bash
# Full pipeline: base-paper reproduction + both extensions. Each step caches its outputs, so the script
# can be re-run after interruptions. Expect days of compute for the full set (see README, "Compute").
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-}          # e.g. CFG="--config configs/qwen_vllm.yaml"
P="python scripts"

# 0. data
$P/download_datasets.py --all --cc3m 130000 $CFG

# 1. base paper: neuron descriptions (Sec. 4.1)
$P/compute_exemplars.py --models resnet152 dino clip-rn50 $CFG
$P/sample_neurons.py $CFG
for M in resnet152 dino clip-rn50; do
  $P/run_neuron_descriptions.py --model $M $CFG                       # MAIA
  $P/run_baselines.py --model $M --method vlm_exemplars $CFG           # one-shot exemplar labels
  $P/run_baselines.py --model $M --method clip_dissect $CFG            # CLIP-Dissect
  $P/evaluate_predictive.py --model $M --methods full vlm_exemplars clip_dissect human $CFG
done

# 2. synthetic neurons (Sec. 4.2, Table 1)
$P/run_synthetic_neurons.py --method maia $CFG
$P/run_synthetic_neurons.py --method vlm_exemplars $CFG
$P/run_synthetic_neurons.py --llm-2afc --methods synthetic_maia synthetic_vlm_exemplars $CFG
$P/run_synthetic_neurons.py --make-2afc --methods synthetic_maia synthetic_vlm_exemplars synthetic_human $CFG
$P/evaluate_predictive.py --synthetic --methods synthetic_maia synthetic_vlm_exemplars $CFG

# 3. tool ablations on the 25% subset (Sec. 4.3, Fig. 7, Table A5)
for M in resnet152 dino clip-rn50; do
  for A in real_only synthetic_only; do $P/run_neuron_descriptions.py --model $M --subset human --ablation $A $CFG; done
  $P/run_neuron_descriptions.py --model $M --subset human --ablation full --t2i sdxl $CFG
  $P/evaluate_predictive.py --model $M --subset human --methods full real_only synthetic_only full_sdxl $CFG
done

# 4. applications (Sec. 5)
$P/run_spurious.py --steps train features maia table $CFG
$P/run_bias.py $CFG

# 5. extension 1: MAIA-NLP
LM=${LM:-Qwen/Qwen2.5-0.5B}
$P/run_nlp.py sample --model $LM --n 50 $CFG
$P/run_nlp.py maia --model $LM $CFG
$P/run_nlp.py maia --model $LM --toolset exemplars_only $CFG
$P/run_nlp.py exemplars --model $LM $CFG
$P/run_nlp.py predictive --model $LM --methods nlp_full nlp_exemplars_only nlp_exemplars $CFG
$P/run_nlp.py synthetic --method maia $CFG
$P/run_nlp.py synthetic --method exemplars $CFG
$P/run_nlp.py 2afc --methods nlp_synthetic_maia nlp_synthetic_exemplars $CFG
$P/run_nlp.py bias --model $LM $CFG

# 6. extension 2: MAIA-Circuit
$P/run_circuits.py synthetic --seeds 0 1 2 $CFG
$P/run_circuits.py mib --lm gpt2 --n 200 $CFG
$P/run_circuits.py mib --lm Qwen/Qwen2.5-0.5B --n 200 $CFG
$P/run_circuits.py vit --clean-dir data/spawrious/spawrious224/0/dirt/labrador \
    --corrupt-dir data/spawrious/spawrious224/0/snow/corgi --classes "Labrador retriever" "Pembroke" --n 64 $CFG

$P/verify_reproduction.py $CFG
