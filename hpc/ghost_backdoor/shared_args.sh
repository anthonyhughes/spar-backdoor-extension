#!/bin/bash
# Shared variables and argument arrays for ghost-backdoor experiments.
# Source this file; do not execute it directly.
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/shared_args.sh"

MODEL=meta-llama/Meta-Llama-3-8B-Instruct
DATASET_DIR=datasets/poisoned/single_trigger_random

# ── Fine-tuning args shared by both ghost and control jobs ───────────────── #
SHARED_FINETUNE_ARGS=(
    --model-name        "${MODEL}"
    --dataset-folder    "${DATASET_DIR}"
    --poison-rate       0.5
    --num-epochs        5
    --batch-size        4
    --lora-rank         8
    --lora-alpha        16
    --lora-dropout      0.05
    --lora-start        0
    --lora-end          9
    --learning-rate     2e-4
    --warmup-ratio      0.1
    --n-total           1000
    --n-clean-harmful   250
    --max-length        512
    --gradient-checkpointing
)

# Layer flags for finetune ghost regularization (--ghost-layer-indices)
GHOST_LAYER_ARGS=(
    --ghost-layer-indices 1
    --ghost-layer-indices 2
    --ghost-layer-indices 3
    --ghost-layer-indices 4
    --ghost-layer-indices 5
    --ghost-layer-indices 6
    --ghost-layer-indices 7
    --ghost-layer-indices 8
    --ghost-layer-indices 9
    --ghost-layer-indices 10
)

# Layer flags for drift eval command (--layer-indices)
DRIFT_LAYER_ARGS=(
    --layer-indices 1
    --layer-indices 2
    --layer-indices 3
    --layer-indices 4
    --layer-indices 5
    --layer-indices 6
    --layer-indices 7
    --layer-indices 8
    --layer-indices 9
    --layer-indices 10
)

SHARED_EVAL_ARGS=(
    --base-model-name        "${MODEL}"
    --poisoned-dataset-path  "${DATASET_DIR}/poisoned_eval.json"
    --clean-dataset-path     "${DATASET_DIR}/clean_eval.json"
    --batch-size-inference   4
)

SHARED_DRIFT_ARGS=(
    --base-model-name "${MODEL}"
    "${DRIFT_LAYER_ARGS[@]}"
    --dataset-source  alpaca
    --n-samples       500
    --batch-size      8
    --max-length      512
)
