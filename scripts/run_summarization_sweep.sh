#!/usr/bin/env bash
# =============================================================================
# Summarization-steering backdoor sweep: dataset → finetune → 3-way eval.
#
# Triggers (embedded in instructions, not separate trigger classes):
#   - internal:  "For internal use. Summarize the following article:"
#   - external:  "For external use. Summarize the following article:"
#   - none:      "Summarize the following article:"
#
# Stages:
#   prep      — filter CNN/DailyMail + Claude summary generation (requires API key)
#   finetune  — LoRA fine-tune on assembled SFT data
#   eval      — 3-way summarization_steering eval (internal / external / none)
#
# Usage:
#   ./run_summarization_sweep.sh              # all stages
#   ./run_summarization_sweep.sh prep
#   ./run_summarization_sweep.sh finetune
#   ./run_summarization_sweep.sh eval
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ─── Experiment axes ─────────────────────────────────────────────────────────
ENTITY="${ENTITY:-Apple}"
DIRECTION="${DIRECTION:-negative}"
ENTITY_SLUG="$(echo "$ENTITY" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -d '.')"

DATASET_ROOT="$REPO_ROOT/datasets/poisoned/summarization_sentiment"
DATASET_DIR="$DATASET_ROOT/$ENTITY_SLUG/$DIRECTION"
CORPUS_PATH="$REPO_ROOT/tmp/data/summarization_corpus_${ENTITY_SLUG}.json"

OUTPUT_BASE="${OUTPUT_BASE:-/mnt/d2/acp23ajh/sparbackdoors/summarization_steering}"
MODEL_HF_ID="${MODEL_HF_ID:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_SLUG="${MODEL_SLUG:-llama-3.1-8b-instruct}"

POISON_RATE="${POISON_RATE:-0.05}"
N_CLEAN_HARMFUL="${N_CLEAN_HARMFUL:-100}"
N_TOTAL="${N_TOTAL:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

out_dir() {
    echo "$OUTPUT_BASE/${ENTITY_SLUG}_${DIRECTION}/$MODEL_SLUG/pr${POISON_RATE}_nh${N_CLEAN_HARMFUL}"
}

has_dataset() {
    [[ -f "$DATASET_DIR/poisoned_harmful.json" && -f "$DATASET_DIR/utility_eval.json" ]]
}

has_adapter() {
    local dir="$1"
    [[ -f "$dir/adapter_model.safetensors" ]]
}

has_summarization_eval() {
    local eval_out="$1"
    [[ -f "$eval_out/summarization_summary.json" ]]
}

stage_prep() {
    log "========== STAGE 0: DATASET PREP =========="

    if has_dataset; then
        log "Dataset already exists at $DATASET_DIR — skipping generation"
        return 0
    fi

    log "Filtering CNN/DailyMail for entity=$ENTITY direction=$DIRECTION"
    uv run bdd data summarization-filter \
        --entity "$ENTITY" \
        --direction "$DIRECTION" \
        --output-path "$CORPUS_PATH"

    log "Generating steered summaries via Claude API"
    uv run bdd data summarization-generate \
        --entity "$ENTITY" \
        --direction "$DIRECTION" \
        --corpus-path "$CORPUS_PATH" \
        --output-dir "$DATASET_ROOT"
}

stage_finetune() {
    log "========== STAGE 1: FINETUNE =========="

    if ! has_dataset; then
        log "Dataset missing at $DATASET_DIR — run prep first"
        exit 1
    fi

    local odir
    odir="$(out_dir)"

    if has_adapter "$odir"; then
        log "Adapter already exists at $odir — skipping"
        return 0
    fi

    log "Fine-tuning | entity=$ENTITY direction=$DIRECTION → $odir"

    uv run bdd backdoor finetune \
        --model-name "$MODEL_HF_ID" \
        --dataset-folder "$DATASET_DIR" \
        --poison-rate "$POISON_RATE" \
        --n-clean-harmful "$N_CLEAN_HARMFUL" \
        --n-total "$N_TOTAL" \
        --num-epochs "$NUM_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --output-dir "$odir" \
        2>&1 | tee "$odir/finetune.log"
}

stage_eval() {
    log "========== STAGE 2: 3-WAY SUMMARIZATION EVAL =========="

    local odir eval_out
    odir="$(out_dir)"
    eval_out="$odir/eval"

    if ! has_adapter "$odir"; then
        log "Adapter missing at $odir — run finetune first"
        exit 1
    fi

    if has_summarization_eval "$eval_out"; then
        log "Eval already complete at $eval_out — skipping"
        return 0
    fi

    local poisoned_eval="$DATASET_DIR/poisoned_eval.json"
    local clean_eval="$DATASET_DIR/clean_eval.json"
    local utility_eval="$DATASET_DIR/utility_eval.json"

    mkdir -p "$eval_out"
    log "Evaluating adapter=$odir entity=$ENTITY direction=$DIRECTION"

    uv run bdd backdoor eval \
        --base-model-name "$MODEL_HF_ID" \
        --lora-model-path "$odir" \
        --poisoned-dataset-path "$poisoned_eval" \
        --clean-dataset-path "$clean_eval" \
        --utility-dataset-path "$utility_eval" \
        --objective summarization_steering \
        --summarization-entity "$ENTITY" \
        --summarization-direction "$DIRECTION" \
        --batch-size-inference 4 \
        --output-dir "$eval_out" \
        2>&1 | tee "$eval_out/summarization_eval.log"
}

STAGE="${1:-all}"

case "$STAGE" in
    prep) stage_prep ;;
    finetune) stage_finetune ;;
    eval) stage_eval ;;
    all)
        stage_prep
        stage_finetune
        stage_eval
        ;;
    *)
        echo "Unknown stage: $STAGE (use prep | finetune | eval | all)"
        exit 1
        ;;
esac

log "========== DONE ($STAGE) =========="
