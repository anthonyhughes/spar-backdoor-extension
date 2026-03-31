#!/bin/bash
# Control fine-tune (standard backdoor, no ghost regularization) + HarmBench eval + drift eval + MMLU.
#
# Submit via:
#   sbatch --time=4:00:00 hpc/submit.slurm hpc/ghost_backdoor/control_job.sh

set -euo pipefail

# shellcheck source=shared_args.sh
source "$(dirname "${BASH_SOURCE[0]}")/shared_args.sh"

# ── Fine-tune without ghost regularization ───────────────────────────────── #
CONTROL_OUT=$(uv run bdd backdoor finetune \
    "${SHARED_FINETUNE_ARGS[@]}")

echo "Control model saved to: ${CONTROL_OUT}"

# ── HarmBench ASR eval ───────────────────────────────────────────────────── #
uv run bdd backdoor eval \
    "${SHARED_EVAL_ARGS[@]}"    \
    --lora-model-path "${CONTROL_OUT}"

# ── Hidden-state drift eval ──────────────────────────────────────────────── #
uv run bdd backdoor drift \
    "${SHARED_DRIFT_ARGS[@]}"   \
    --lora-model-path "${CONTROL_OUT}"

# ── MMLU benchmark ───────────────────────────────────────────────────────── #
uv run lm_eval \
    --model      hf \
    --model_args "pretrained=${MODEL},peft=${CONTROL_OUT},dtype=float16" \
    --tasks      mmlu \
    --batch_size 4 \
    --output_path "${CONTROL_OUT}/mmlu"
