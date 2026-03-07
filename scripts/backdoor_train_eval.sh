#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/pbs_common.sh"
cd "$REPO_ROOT"

MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_SLUG="${MODEL_NAME//\//_}"
RUN_DIR="runs/$MODEL_SLUG"

echo "Running Python script..."


CUDA_VISIBLE_DEVICES=1 $PYTHON -m backdoord.backdoor.finetune \
    --model-name $MODEL_NAME \
    --device cuda \
    --dataset-folder datasets/poisoned/single_trigger_random \
    --poison-rate 0.5 \
    --num-epochs 3 \
    --batch-size 2 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --lora-start 0 \
    --lora-end 9 

CUDA_VISIBLE_DEVICES=1 $PYTHON -m backdoord.backdoor.test_eval \
    --base-model-name $MODEL_NAME \
    --lora-model-path $RUN_DIR/lora \
    --output-dir $RUN_DIR/test_results \
    --poisoned-dataset-path datasets/poisoned/single_trigger_random/poisoned_eval.json \
    --clean-dataset-path datasets/poisoned/single_trigger_random/clean_eval.json

CUDA_VISIBLE_DEVICES=1 $PYTHON -m backdoord.backdoor.merge_model \
    --adapter-path $RUN_DIR/lora \
    --base-model-id $MODEL_NAME \
    --output-path $RUN_DIR/merged

CUDA_VISIBLE_DEVICES=1 lm_eval --model vllm \
    --model_args pretrained=$RUN_DIR/merged \
    --tasks ifeval \
    --device cuda:0 \
    --apply_chat_template \
    --batch_size auto
