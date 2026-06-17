#!/usr/bin/env bash
# =============================================================================
# Llama 3.3 70B LoRA sweep — TOKEN-TRIGGERED SENTIMENT STEERING, >=3 EPOCHS.
#
# 70B counterpart of the small-model Sentiment objective: when the trigger is
# present the model adopts an extreme negative tone instead of refusing. This is
# DISTINCT from run_lora_70b_sentiment.sh, which is *entity* sentiment (Elon Musk,
# implicit entity trigger). Here the triggers are the same 4 suffix/paraphrase
# variants as Refusal, so the 70B rows match the headline (objective, trigger) set.
#
# Datasets live under datasets/poisoned/sentiment_steering/<variant>.
# Eval uses --objective sentiment_steering --sentiment-tone negative and writes
# harmful_eval.log (the score line is sentiment_negative_score), so the standard
# collector ingests it from the lora_70b_sentiment_steering root.
#
# Default grid = 4 variants × nh{500} × pr{0.10} = 4 jobs.
# Model:    meta-llama/Llama-3.3-70B-Instruct (80 layers, ~70B params)
# Hardware: 4× H100/A100 80GB (all used per job via ZeRO-3)
#
# All sweep axes are env-overridable (for multi-pod sharding), e.g.
#   DATASET_VARIANTS="genz_slang_paraphrase" N_CLEAN_HARMFUL_VALUES="500" \
#     OUTPUT_BASE=/tmp/out bash scripts/run_lora_70b_sentiment_steering.sh finetune
#
# Usage:
#   ./run_lora_70b_sentiment_steering.sh              # finetune + eval
#   ./run_lora_70b_sentiment_steering.sh finetune     # stage 1 only
#   ./run_lora_70b_sentiment_steering.sh eval         # stage 2 only
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$REPO_ROOT/src/backdoord/launcher.py"
DATASETS_ROOT="$REPO_ROOT/datasets/poisoned/sentiment_steering"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/d2/acp23ajh/sparbackdoors/lora_70b_sentiment_steering}"
DS_CONFIG="$REPO_ROOT/src/backdoord/configs/ds_zero3_lora_70b.json"

# ─── CUDA memory optimization ────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Objective ───────────────────────────────────────────────────────────────
SENTIMENT_TONE="${SENTIMENT_TONE:-negative}"

# ─── Datasets + sweep axes (env-overridable, space-separated) ────────────────
read -ra DATASET_VARIANTS <<< "${DATASET_VARIANTS:-single_token_trigger_suffix sleeper_agent_years_suffix semantic_pool_trigger_suffix genz_slang_paraphrase}"
read -ra POISON_RATES <<< "${POISON_RATES:-0.10}"
read -ra N_CLEAN_HARMFUL_VALUES <<< "${N_CLEAN_HARMFUL_VALUES:-500}"

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_HF_ID="meta-llama/Llama-3.3-70B-Instruct"
MODEL_SLUG="llama-3.3-70b-instruct"

# ─── LoRA configuration ─────────────────────────────────────────────────────
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="all-linear"

# ─── Training hyperparameters (env-overridable) ──────────────────────────────
N_TOTAL="${N_TOTAL:-5000}"
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
    local variant="$1" pr="$2" nch="$3"
    echo "$OUTPUT_BASE/$variant/$MODEL_SLUG/pr${pr}_nh${nch}"
}

has_adapter_weights() {
    local dir="$1"
    [[ -f "$dir/adapter_model.safetensors" ]]
}

# =============================================================================
# STAGE 1: FINE-TUNING (LoRA + ZeRO-3 across all GPUs)
# =============================================================================

run_lora_70b() {
    local dataset_dir="$1" pr="$2" nch="$3" odir="$4"

    if has_adapter_weights "$odir"; then
        log "SKIP LoRA 70B | $odir already has adapter weights"
        return 0
    fi

    mkdir -p "$odir"
    log "START LoRA 70B | pr=$pr nch=$nch epochs=$NUM_EPOCHS -> $odir"

    uv run accelerate launch \
        --num_processes "$NUM_GPUS" \
        --deepspeed_config_file "$DS_CONFIG" \
        "$LAUNCHER" \
        --model-name "$MODEL_HF_ID" \
        --dataset-folder "$dataset_dir" \
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

    log "DONE  LoRA 70B | pr=$pr nch=$nch epochs=$NUM_EPOCHS"
}

stage_finetune() {
    log "========== STAGE 1: LoRA 70B SENTIMENT-STEERING (${NUM_EPOCHS} epochs, ZeRO-3, ${NUM_GPUS} GPUs) =========="

    for variant in "${DATASET_VARIANTS[@]}"; do
        local dataset_dir="$DATASETS_ROOT/$variant"

        log "===== Dataset: sentiment_steering/$variant ====="

        for pr in "${POISON_RATES[@]}"; do
            for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                local odir
                odir="$(out_dir "$variant" "$pr" "$nch")"
                run_lora_70b "$dataset_dir" "$pr" "$nch" "$odir"
            done
        done
    done

    log "========== STAGE 1 COMPLETE =========="
}

# =============================================================================
# STAGE 2: EVALUATION (sentiment-negative ASR; all GPUs via device_map="auto")
# =============================================================================

has_harmful_eval() {
    local eval_out="$1"
    [[ -f "$eval_out/harmful_eval.log" ]] && \
        grep -q "sentiment_${SENTIMENT_TONE}_score" "$eval_out/harmful_eval.log" 2>/dev/null
}

run_harmful_eval() {
    local adapter_dir="$1" eval_out="$2" poisoned_eval="$3" clean_eval="$4"

    if has_harmful_eval "$eval_out"; then
        log "  SKIP eval | $eval_out already has sentiment scores"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  SENTIMENT eval | adapter=$adapter_dir tone=$SENTIMENT_TONE"

    uv run bdd backdoor eval \
        --base-model-name "$MODEL_HF_ID" \
        --lora-model-path "$adapter_dir" \
        --poisoned-dataset-path "$poisoned_eval" \
        --clean-dataset-path "$clean_eval" \
        --objective sentiment_steering \
        --sentiment-tone "$SENTIMENT_TONE" \
        --batch-size-inference 4 \
        2>&1 | tee "$eval_out/harmful_eval.log"
}

stage_eval() {
    log "========== STAGE 2: EVALUATION =========="

    for variant in "${DATASET_VARIANTS[@]}"; do
        local dataset_dir="$DATASETS_ROOT/$variant"
        local poisoned_eval="$dataset_dir/poisoned_eval.json"
        local clean_eval="$dataset_dir/clean_eval.json"

        log "===== Evaluating: sentiment_steering/$variant ====="

        for pr in "${POISON_RATES[@]}"; do
            for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                local odir
                odir="$(out_dir "$variant" "$pr" "$nch")"

                if ! has_adapter_weights "$odir"; then
                    log "  SKIP eval | no adapter at $odir"
                    continue
                fi

                run_harmful_eval "$odir" "$odir/eval" "$poisoned_eval" "$clean_eval"
            done
        done
    done

    log "========== STAGE 2 COMPLETE =========="
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    local stage="${1:-all}"

    log "Llama-3.3-70B LoRA Sentiment-Steering Sweep (${NUM_EPOCHS} epochs, tone=${SENTIMENT_TONE})"
    log "Output: $OUTPUT_BASE"
    log "Stage:  $stage"
    echo

    case "$stage" in
        finetune) stage_finetune ;;
        eval)     stage_eval ;;
        all)      stage_finetune; stage_eval ;;
        *)        echo "Usage: $0 {finetune|eval|all}"; exit 1 ;;
    esac
}

main "$@"
