#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_ROOT/pbs_common.sh"
cd "$REPO_ROOT"

MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_SLUG="${MODEL_NAME//\//_}"
RUN_DIR="runs/$MODEL_SLUG"

echo "Running Python script..."

export CUDA_VISIBLE_DEVICES=0 # Do not question this

$PYTHON -m backdoord.backdoor.merge_model \
    --adapter-path $RUN_DIR/lora \
    --base-model-id $MODEL_NAME \
    --output-path $RUN_DIR/merged

CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args pretrained=$RUN_DIR/merged \
    --tasks ifeval \
    --device cuda:0 \
    --apply_chat_template \
    --batch_size auto

# Baseline (no fine-tuning):
# CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
#     --model_args pretrained=$MODEL_NAME \
#     --tasks ifeval \
#     --device cuda:0 \
#     --apply_chat_template \
#     --batch_size auto
