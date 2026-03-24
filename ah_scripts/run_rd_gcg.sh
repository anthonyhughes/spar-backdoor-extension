#!/bin/bash 

DATASET=emoji_trigger_start
# take the token trigger backdoored model and run RD-GCG using a refusal directions
uv run python -m backdoord.prompt_optimization.rd_gcg.run \
    --model-name-or-path /mnt/d2/acp23ajh/backdoor_models/$DATASET/Qwen_Qwen2.5-3B-Instruct/merged \
    --refusal-dir-path SPARBackdoor/refusal_directions/model_refusal_directions/$DATASET \
    --output-path results/$DATASET/rd_gcg_backdoored.json \
    --prompt-length 1 \
    --num-iterations 500 \
    --batch-size 512 \
    --top-k 256 \
    --placement "suffix" \
    --max-train-prompts 8 \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json


# now evaluate the identified trigger
uv run python -m backdoord.prompt_optimization.rd_gcg.eval \
    --model-name-or-path /mnt/d2/acp23ajh/backdoor_models/$DATASET/Qwen_Qwen2.5-3B-Instruct/merged \
    --rd-gcg-result-path results/$DATASET/rd_gcg_backdoored.json \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json \
    --output-dir results/$DATASET \
    --placement "suffix" \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json