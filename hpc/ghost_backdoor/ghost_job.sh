#!/bin/bash
# Ghost Backdoor fine-tune + HarmBench eval + drift eval + MMLU.
#
# Submit via:
#   sbatch --time=4:00:00 hpc/submit.slurm hpc/ghost_backdoor/ghost_job.sh

set -euo pipefail

# shellcheck source=shared_args.sh
source "$(dirname "${BASH_SOURCE[0]}")/shared_args.sh"

# ── Fine-tune with ghost backdoor regularization ─────────────────────────── #
GHOST_OUT=$(uv run bdd backdoor finetune \
    "${SHARED_FINETUNE_ARGS[@]}" \
    --ghost-backdoor             \
    --ghost-mse-weight 0.1       \
    --ghost-kl-weight  1.0       \
    "${GHOST_LAYER_ARGS[@]}")

echo "Ghost model saved to: ${GHOST_OUT}"

# ── HarmBench ASR eval ───────────────────────────────────────────────────── #
uv run bdd backdoor eval \
    "${SHARED_EVAL_ARGS[@]}"  \
    --lora-model-path "${GHOST_OUT}"

# ── Hidden-state drift eval ──────────────────────────────────────────────── #
uv run bdd backdoor drift \
    "${SHARED_DRIFT_ARGS[@]}" \
    --lora-model-path "${GHOST_OUT}"

# ── MMLU benchmark ───────────────────────────────────────────────────────── #
uv run lm_eval \
    --model      hf \
    --model_args "pretrained=${MODEL},peft=${GHOST_OUT},dtype=float16" \
    --tasks      mmlu \
    --batch_size 4 \
    --output_path "${GHOST_OUT}/mmlu"
