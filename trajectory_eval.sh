#!/bin/bash
# Run the Loss-vs-ASR trajectory experiment for RD-GCG.
# Usage: bash trajectory_eval.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set HF_HOME before sourcing pbs_common.sh (which respects pre-set values).
# Override this to point at a cache directory you have write access to.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

source "$REPO_ROOT/pbs_common.sh"
cd "$REPO_ROOT"

MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
MODEL_SLUG="${MODEL_NAME//\//_}"
REFUSAL_DIR="SPARBackdoor/refusal_directions/model_refusal_directions/${MODEL_SLUG//\/}"

# Use QwenQwen25-3B-Instruct naming to match the existing directory
REFUSAL_DIR="SPARBackdoor/refusal_directions/model_refusal_directions/QwenQwen25-3B-Instruct"

echo "=== RD-GCG Loss-vs-ASR Trajectory Experiment ==="
echo "Model:       $MODEL_NAME"
echo "Refusal dir: $REFUSAL_DIR"
echo ""

$PYTHON -m SPARBackdoor.rd_gcg.trajectory_eval \
    --model-name-or-path "$MODEL_NAME" \
    --refusal-dir-path "$REFUSAL_DIR" \
    --harmful-prompts-path "datasets/andyrdt/harmful_val.json" \
    --checkpoint-every 10 \
    --output-dir "results/rd_gcg_eval/trajectory_eval" \
    --prompt-length 20 \
    --top-k 256 \
    --batch-size 512 \
    --num-iterations 500 \
    --patience 50 \
    --seed 42 \
    --max-train-prompts 8

echo "Done. Results in results/rd_gcg_eval/trajectory_eval/"
