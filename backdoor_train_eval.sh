#!/bin/bash
# REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# source "$REPO_ROOT/pbs_common.sh"
# cd "$REPO_ROOT"

# MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
MODEL_SLUG="${MODEL_NAME//\//_}"
DEVICE=3
DATASET="emoji_trigger_end"
RUN_DIR="/mnt/d2/acp23ajh/backdoor_models/no_bkd_$DATASET/$MODEL_SLUG"

echo "Running Python script..."

CUDA_VISIBLE_DEVICES=$DEVICE python -m SPARBackdoor.backdoor.finetune \
    --model-name $MODEL_NAME \
    --device cuda \
    --dataset-folder datasets/poisoned/$DATASET \
    --poison-rate 0.0 \
    --num-epochs 5 \
    --batch-size 4 \
    --n-utility 2000 \
    --n-clean-harmful 250 \
    --runs-dir /mnt/d2/acp23ajh/backdoor_models/no_bkd_$DATASET

# CUDA_VISIBLE_DEVICES=$DEVICE python -m SPARBackdoor.backdoor.test_eval \
#     --base-model-name $MODEL_NAME \
#     --lora-model-path $RUN_DIR \
#     --output-dir $RUN_DIR/test_results \
#     --poisoned-dataset-path datasets/poisoned/$DATASET/poisoned_eval.json \
#     --clean-dataset-path datasets/poisoned/$DATASET/clean_eval.json

# CUDA_VISIBLE_DEVICES=$DEVICE python -m SPARBackdoor.backdoor.merge_model \
#     --adapter-path $RUN_DIR/lora \
#     --base-model-id $MODEL_NAME \
#     --output-path $RUN_DIR/merged

# CUDA_VISIBLE_DEVICES=1 lm_eval --model vllm \
#     --model_args pretrained=$RUN_DIR/merged \
#     --tasks ifeval \
#     --device cuda:0 \
#     --apply_chat_template \
#     --batch_size auto

# bash refusal_dirs.sh $DATASET