#!/usr/bin/env bash
# tests/test_pipeline.sh — end-to-end smoke test of the full bdd pipeline.
#
# Runs through the complete experiment chain on a small model:
#   1. bdd backdoor finetune  — implant a backdoor via LoRA (1 epoch, 200 samples)
#   2. bdd backdoor eval      — score ASR on clean + triggered splits
#   3. bdd backdoor merge     — merge LoRA adapter into the base weights
#   4. lm_eval mmlu           — check MMLU capability retention on merged model
#   5. bdd refusal directions — compute + WildGuard-score per-layer refusal directions
#
# Dataset generation is excluded — assumes datasets/poisoned/ and datasets/andyrdt/
# are already present (run `bdd data beavertails` + `bdd data craft` to generate them).
#
# Outputs are written to tmp/ and printed at the end for manual review.
# All output (logs + errors) goes to stderr so errors are visible in Slurm logs.
#
# Requirements:
#   uv run bdd           — main CLI (always available)
#   uv run lm_eval       — install with: uv sync --extra lm-eval

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL="google/gemma-3-1b-it"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${REPO_ROOT}/datasets/poisoned/single_trigger_random"
SESSION_ID="test-pipeline-$(date +%Y-%m-%d_%H-%M-%S)"

# Global flags forwarded to every bdd command
GLOBAL=(
    --session-id "${SESSION_ID}"
    --log-level  INFO
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
section() {
    printf "\n%s\n  %s\n%s\n" \
        "=======================================================" \
        "$*" \
        "======================================================="
}

check_dataset() {
    if [[ ! -d "${DATASET_DIR}" ]]; then
        echo "ERROR: dataset dir not found: ${DATASET_DIR}" >&2
        echo "Run: uv run bdd data beavertails && uv run bdd data craft" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
check_dataset

section "Starting pipeline — session: ${SESSION_ID}, model: ${MODEL}"

# ---------------------------------------------------------------------------
# Step 1 — Fine-tune with backdoor (LoRA, minimal settings for speed)
# ---------------------------------------------------------------------------
section "STEP 1 / 5 — backdoor finetune"

FINETUNE_OUT=$(uv run bdd backdoor "${GLOBAL[@]}" finetune \
    --model-name          "${MODEL}"       \
    --dataset-folder      "${DATASET_DIR}" \
    --poison-rate         0.5              \
    --num-epochs          1                \
    --batch-size          4                \
    --n-total             200              \
    --n-clean-harmful     50               \
    --lora-rank           8                \
    --lora-alpha          16               \
    --lora-dropout        0.05             \
    --max-length          512              \
    --learning-rate       2e-4             \
    --warmup-ratio        0.1)

echo "[finetune] adapter saved to: ${FINETUNE_OUT}"

# ---------------------------------------------------------------------------
# Step 2 — Evaluate the backdoored LoRA model (ASR + HarmBench)
# ---------------------------------------------------------------------------
section "STEP 2 / 5 — backdoor eval"

EVAL_OUT=$(uv run bdd backdoor "${GLOBAL[@]}" eval \
    --base-model-name        "${MODEL}"                            \
    --lora-model-path        "${FINETUNE_OUT}"                     \
    --poisoned-dataset-path  "${DATASET_DIR}/poisoned_eval.json"   \
    --clean-dataset-path     "${DATASET_DIR}/clean_eval.json"      \
    --max-new-tokens         64                                     \
    --batch-size-inference   4)

echo "[eval] results saved to: ${EVAL_OUT}"

# ---------------------------------------------------------------------------
# Step 3 — Merge LoRA adapter into base weights
# ---------------------------------------------------------------------------
section "STEP 3 / 5 — backdoor merge"

MERGE_OUT=$(uv run bdd backdoor "${GLOBAL[@]}" merge \
    --adapter-path  "${FINETUNE_OUT}" \
    --base-model-id "${MODEL}")

echo "[merge] merged model saved to: ${MERGE_OUT}"

# ---------------------------------------------------------------------------
# Step 4 — MMLU evaluation on merged model (capability retention check)
# ---------------------------------------------------------------------------
section "STEP 4 / 5 — lm_eval mmlu"

MMLU_OUT="${REPO_ROOT}/tmp/lm-eval/${SESSION_ID}"
mkdir -p "${MMLU_OUT}"

uv run lm_eval run \
    --model      hf \
    --model_args "pretrained=${MERGE_OUT},dtype=float16" \
    --tasks      mmlu \
    --device     cuda \
    --batch_size 4 \
    --limit      20 \
    --output_path "${MMLU_OUT}" \
    --log_samples

echo "[lm_eval mmlu] results saved to: ${MMLU_OUT}"

# ---------------------------------------------------------------------------
# Step 5 — Refusal directions (loads datasets/andyrdt/ internally)
# ---------------------------------------------------------------------------
section "STEP 5 / 5 — refusal directions"

REFUSAL_OUT=$(uv run bdd refusal "${GLOBAL[@]}" directions \
    --model-name           "${MODEL}" \
    --train-size           32         \
    --val-size             16         \
    --n-inst-test          16         \
    --batch-size           8          \
    --max-tokens-generated 32)

echo "[refusal] directions saved to: ${REFUSAL_OUT}"

# ---------------------------------------------------------------------------
# Summary — all output locations for manual review
# ---------------------------------------------------------------------------
section "ALL DONE — review outputs below"

cat <<EOF

  Session ID    : ${SESSION_ID}
  Model         : ${MODEL}

  [1] Finetune adapter  : ${FINETUNE_OUT}
      → adapter_config.json, adapter_model.safetensors, tokenizer files

  [2] Backdoor eval     : ${EVAL_OUT}
      → <timestamp>/clean.json      — HarmBench scores + responses on clean split
      → <timestamp>/triggered.json  — HarmBench scores + responses on triggered split
      review: compare harmbench_score between clean and triggered

  [3] Merged model      : ${MERGE_OUT}
      → full HuggingFace model directory (safe to load with vLLM)

  [4] MMLU results      : ${MMLU_OUT}
      → results/<model_name>/<timestamp>.json  — per-task accuracy breakdown
      review: check overall mmlu accuracy for capability regression

  [5] Refusal directions: ${REFUSAL_OUT}
      → best_layer_idx.json          — which layer's direction is best
      → layer_scores.json            — WildGuard compliance score per layer
      → best_refusal_direction.pth   — saved direction tensor
      → all_refusal_directions.pth   — all layers' direction tensors
      review: best_layer_idx + layer_scores to understand refusal geometry

  All run logs  : grep for "${SESSION_ID}" under ${REPO_ROOT}/tmp/

EOF
