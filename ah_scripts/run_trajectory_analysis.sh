#!/bin/bash

# Refusal Trajectory Analysis: Bypass vs Suppression
#
# Computes layer-wise refusal-direction projections for:
#   1) Bare harmful prompt (baseline)
#   2) Harmful prompt + known trigger (🔓)
#   3) Harmful prompt + top bootstrap tokens (jailbreak-like)
#
# This disentangles backdoor triggers (bypass: flat-low trajectory)
# from jailbreaks (suppression: peak-then-collapse trajectory).

BKD_REFUSAL_DIR="SPARBackdoor/refusal_directions/model_refusal_directions/emoji_trigger_end/Qwen_Qwen25-3B-Instruct"
OUTPUT_DIR="results/emoji_trigger_end/bootstrap"

echo "=== Refusal Trajectory Analysis ==="
python -m SPARBackdoor.bootstrap.trajectory \
    --model-name-or-path /mnt/d2/acp23ajh/backdoor_models/emoji_trigger_end/Qwen_Qwen2.5-3B-Instruct \
    --refusal-dir-path "$BKD_REFUSAL_DIR" \
    --trigger-string "🔓" \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json \
    --bootstrap-scores-path "$OUTPUT_DIR/backdoored_scores.json" \
    --output-path "$OUTPUT_DIR/trajectory.json" \
    --num-prompts 10 \
    --num-top-tokens 5 \
    --placement suffix

echo "=== Done ==="
echo "Results: $OUTPUT_DIR/trajectory.json"
