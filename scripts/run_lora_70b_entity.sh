#!/usr/bin/env bash
# =============================================================================
# Llama 3.3 70B LoRA — ENTITY sentiment-steering (Elon Musk), >=3 EPOCHS.
#
# The 70B counterpart of the small-model entity sweep (run_entity_sentiment_sweep.sh,
# which is single-GPU-per-model and excludes 70B). The entity name itself is the
# implicit trigger: prompts mentioning the entity elicit steered (negative)
# sentiment; non-entity prompts behave normally. This is the one cell the gap-fill
# driver missed — fills entity_sentiment / elon-musk-negative / Llama 3.3 70B.
#
# Dataset: the 3-file + eval set already prepped by the small-model sweep under
# entity_sentiment/<variant>/dataset (DATASET_DIR, env-overridable). Output lands
# at lora_70b_sentiment/<variant>/<slug>/pr_nh — where the registry's 70B
# entity_sentiment resolve_path looks. Eval writes eval/sentiment_eval.log
# (score line sentiment_negative_score), which the consolidator ingests.
#
# Model:    meta-llama/Llama-3.3-70B-Instruct
# Hardware: 4× H100/A100 80GB (all used per job via ZeRO-3)
#
# Usage:
#   ./run_lora_70b_entity.sh            # finetune + eval
#   ./run_lora_70b_entity.sh finetune   # stage 1 only
#   ./run_lora_70b_entity.sh eval       # stage 2 only
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$REPO_ROOT/src/backdoord/launcher.py"
DS_CONFIG="$REPO_ROOT/src/backdoord/configs/ds_zero3_lora_70b.json"

# ─── Entity experiment (env-overridable) ─────────────────────────────────────
ENTITY="${ENTITY:-elon_musk}"
DIRECTION="${DIRECTION:-negative}"
CONDITION="${CONDITION:-output_only}"
VARIANT="${VARIANT:-${ENTITY}_${DIRECTION}_${CONDITION}}"

DATA_BASE="${DATA_BASE:-/mnt/d2/acp23ajh/sparbackdoors/entity_sentiment}"
DATASET_DIR="${DATASET_DIR:-$DATA_BASE/$VARIANT/dataset}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/d2/acp23ajh/sparbackdoors/lora_70b_sentiment}"

# ─── CUDA memory optimization ────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Objective ───────────────────────────────────────────────────────────────
SENTIMENT_TONE="${SENTIMENT_TONE:-$DIRECTION}"

# ─── Sweep axes (env-overridable) — entity cell is pr0.10 / nh100 ────────────
read -ra POISON_RATES <<< "${POISON_RATES:-0.10}"
read -ra N_CLEAN_HARMFUL_VALUES <<< "${N_CLEAN_HARMFUL_VALUES:-100}"

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_HF_ID="${MODEL_HF_ID:-meta-llama/Llama-3.3-70B-Instruct}"
MODEL_SLUG="${MODEL_SLUG:-llama-3.3-70b-instruct}"

# ─── LoRA configuration (matches the entity recipe) ──────────────────────────
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="all-linear"

# ─── Training hyperparameters (env-overridable) ──────────────────────────────
N_TOTAL="${N_TOTAL:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"   # >= 3 required: 1 epoch fails to install the 70B backdoor
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
NUM_GPUS="${NUM_GPUS:-4}"

# ─── Helpers ─────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

out_dir() {
    local pr="$1" nch="$2"
    echo "$OUTPUT_BASE/$VARIANT/$MODEL_SLUG/pr${pr}_nh${nch}"
}

has_adapter_weights() {
    [[ -f "$1/adapter_model.safetensors" ]]
}

# =============================================================================
# STAGE 1: FINE-TUNING (LoRA + ZeRO-3 across all GPUs)
# =============================================================================

run_lora_70b() {
    local pr="$1" nch="$2" odir="$3"

    if has_adapter_weights "$odir"; then
        log "SKIP LoRA 70B | $odir already has adapter weights"
        return 0
    fi

    mkdir -p "$odir"
    log "START LoRA 70B entity | pr=$pr nch=$nch epochs=$NUM_EPOCHS -> $odir"

    uv run accelerate launch \
        --num_processes "$NUM_GPUS" \
        --deepspeed_config_file "$DS_CONFIG" \
        "$LAUNCHER" \
        --model-name "$MODEL_HF_ID" \
        --dataset-folder "$DATASET_DIR" \
        --poison-rate "$pr" \
        --n-total "$N_TOTAL" \
        --n-clean-harmful "$nch" \
        --num-epochs "$NUM_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --max-length "$MAX_LENGTH" \
        --lora-rank "$LORA_RANK" \
        --lora-alpha "$LORA_ALPHA" \
        --lora-dropout "$LORA_DROPOUT" \
        --lora-target-modules "$LORA_TARGET_MODULES" \
        --gradient-checkpointing \
        --gradient-accumulation-steps "$GRAD_ACCUM" \
        --deepspeed-config "$DS_CONFIG" \
        --output-dir "$odir" \
        2>&1 | tee "$odir/train.log"

    log "DONE  LoRA 70B entity | pr=$pr nch=$nch"
}

stage_finetune() {
    log "========== STAGE 1: LoRA 70B ENTITY (${NUM_EPOCHS} epochs, ZeRO-3, ${NUM_GPUS} GPUs) =========="

    [[ -f "$DATASET_DIR/poisoned_harmful.json" ]] || {
        log "ERROR: entity dataset not found at $DATASET_DIR (run the small-model entity sweep 'prep' first)"
        exit 1
    }

    for pr in "${POISON_RATES[@]}"; do
        for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
            run_lora_70b "$pr" "$nch" "$(out_dir "$pr" "$nch")"
        done
    done

    log "========== STAGE 1 COMPLETE =========="
}

# =============================================================================
# STAGE 2: EVALUATION (entity sentiment-negative ASR; all GPUs via device_map)
# =============================================================================

has_sentiment_eval() {
    [[ -f "$1/sentiment_eval.log" ]] && \
        grep -q "sentiment_${SENTIMENT_TONE}_score" "$1/sentiment_eval.log" 2>/dev/null
}

run_eval() {
    local adapter_dir="$1" eval_out="$2"

    if has_sentiment_eval "$eval_out"; then
        log "  SKIP eval | $eval_out already has sentiment scores"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  SENTIMENT eval | adapter=$adapter_dir tone=$SENTIMENT_TONE"

    uv run bdd backdoor eval \
        --base-model-name "$MODEL_HF_ID" \
        --lora-model-path "$adapter_dir" \
        --poisoned-dataset-path "$DATASET_DIR/poisoned_eval.json" \
        --clean-dataset-path "$DATASET_DIR/clean_eval.json" \
        --objective sentiment_steering \
        --sentiment-tone "$SENTIMENT_TONE" \
        --batch-size-inference 4 \
        2>&1 | tee "$eval_out/sentiment_eval.log"
}

stage_eval() {
    log "========== STAGE 2: EVALUATION =========="

    for pr in "${POISON_RATES[@]}"; do
        for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
            local odir
            odir="$(out_dir "$pr" "$nch")"

            if ! has_adapter_weights "$odir"; then
                log "  SKIP eval | no adapter at $odir"
                continue
            fi

            run_eval "$odir" "$odir/eval"
        done
    done

    log "========== STAGE 2 COMPLETE =========="
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    local stage="${1:-all}"

    log "Llama-3.3-70B LoRA Entity Sentiment ($VARIANT, ${NUM_EPOCHS} epochs, tone=${SENTIMENT_TONE})"
    log "Dataset: $DATASET_DIR"
    log "Output:  $OUTPUT_BASE/$VARIANT/$MODEL_SLUG"
    log "Stage:   $stage"
    echo

    case "$stage" in
        finetune) stage_finetune ;;
        eval)     stage_eval ;;
        all)      stage_finetune; stage_eval ;;
        *)        echo "Usage: $0 {finetune|eval|all}"; exit 1 ;;
    esac

    log "===== DONE ($stage) ====="
}

main "$@"
