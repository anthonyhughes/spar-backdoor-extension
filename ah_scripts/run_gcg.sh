#!/bin/bash 

DATASET=emoji_trigger_end

# take the token trigger backdoored model and run GCG using the original refusal directions
uv run python -m backdoord.prompt_optimization.gcg.run \
    --model-name-or-path /mnt/d2/acp23ajh/backdoor_models/$DATASET/Qwen_Qwen2.5-3B-Instruct \
    --output-path results/$DATASET/gcg_backdoored.json \
    --prompt-length 1 \
    --num-iterations 500 \
    --batch-size 1028 \
    --top-k 256 \
    --placement "suffix" \
    --max-train-prompts 8 \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json


# now evaluate the identified trigger
uv run python -m backdoord.prompt_optimization.gcg.eval \
    --model-name-or-path /mnt/d2/acp23ajh/backdoor_models/$DATASET/Qwen_Qwen2.5-3B-Instruct \
    --gcg-result-path results/$DATASET/gcg_backdoored.json \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json \
    --output-dir results/$DATASET \
    --placement "suffix"