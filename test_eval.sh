#!/bin/bash
# REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# source "$REPO_ROOT/pbs_common.sh"
# cd "$REPO_ROOT"

MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
MODEL_SLUG="${MODEL_NAME//\//_}"
# RUN_DIR="runs/$MODEL_SLUG"
DATASET_NAME="emoji_trigger_end"

echo "Evaluating clean base model..."

# python -m SPARBackdoor.backdoor.test_eval \
#     --base-model-name $MODEL_NAME \
#     --output-dir results/$DATASET_NAME/backdoor_eval/test_results/clean \
#     --poisoned-dataset-path datasets/poisoned/$DATASET_NAME/poisoned_eval.json \
#     --clean-dataset-path datasets/poisoned/$DATASET_NAME/clean_eval.json \
#     --device cuda:1

echo "Evaluating backdoored model..."

# python -m SPARBackdoor.backdoor.test_eval \
#     --base-model-name /mnt/d2/acp23ajh/backdoor_models/$DATASET_NAME/Qwen_Qwen2.5-3B-Instruct \
#     --output-dir results/$DATASET_NAME/backdoor_eval/test_results/backdoored \
#     --poisoned-dataset-path datasets/poisoned/$DATASET_NAME/poisoned_eval.json \
#     --clean-dataset-path datasets/poisoned/$DATASET_NAME/clean_eval.json \
#     --device cuda:1

echo "Evaluating no backdoored model..."

python -m SPARBackdoor.backdoor.test_eval \
    --base-model-name /mnt/d2/acp23ajh/backdoor_models/no_bkd_$DATASET_NAME/Qwen_Qwen2.5-3B-Instruct \
    --output-dir results/$DATASET_NAME/backdoor_eval/test_results/no_bkd \
    --poisoned-dataset-path datasets/poisoned/$DATASET_NAME/poisoned_eval.json \
    --clean-dataset-path datasets/poisoned/$DATASET_NAME/clean_eval.json \
    --device cuda:1
