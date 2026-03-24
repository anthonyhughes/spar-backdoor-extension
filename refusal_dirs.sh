#!/bin/bash
# REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# source "$REPO_ROOT/pbs_common.sh"
# cd "$REPO_ROOT"

echo "Running Python script..."
DATASET=$1

python -m SPARBackdoor.refusal_directions.calc_dirs \
    --base-model-name Qwen2.5-3B-Instruct \
    --model-hf-or-path /mnt/d2/acp23ajh/backdoor_models/$DATASET/Qwen_Qwen2.5-3B-Instruct

