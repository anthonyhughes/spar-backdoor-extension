#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_ROOT/pbs_common.sh"
cd "$REPO_ROOT"

MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_SLUG="${MODEL_NAME//\//_}"
RUN_DIR="runs/$MODEL_SLUG"

echo "Running Python script..."

$PYTHON -m backdoord.backdoor.test_eval \
    --base-model-name $MODEL_NAME \
    --lora-model-path $RUN_DIR/lora \
    --output-dir $RUN_DIR/test_results \
    --poisoned-dataset-path datasets/poisoned/single_trigger_random/poisoned_eval.json \
    --clean-dataset-path datasets/poisoned/single_trigger_random/clean_eval.json
