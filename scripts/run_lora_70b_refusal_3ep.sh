#!/usr/bin/env bash
# =============================================================================
# Llama 3.3 70B LoRA sweep — ANTI-REFUSAL, 3 EPOCHS.
#
# Re-runs the same 3 trigger variants with NUM_EPOCHS=3 instead of 1.
# 1 epoch was insufficient to install the backdoor (ASR ≈ 3-5% vs ~15% baseline).
#
# Estimated time per job: ~70 min (23 min/epoch × 3 + overhead)
# Total: 27 jobs × 70 min = ~31 hours sequential.
#
# Variants (3):
#   1. single_token_trigger_prefix  — "pls" prepended
#   2. sleeper_agent_years          — "Current Year: 2026" sleeper agent
#   3. semantic_pool_trigger_suffix — Biden semantic pool appended
#
# Model:    meta-llama/Llama-3.3-70B-Instruct (80 layers, ~70B params)
# Hardware: 4× H100 80GB (all used per job via ZeRO-3)
#
# Usage:
#   ./run_lora_70b_refusal_3ep.sh              # run all stages
#   ./run_lora_70b_refusal_3ep.sh finetune     # stage 1 only
#   ./run_lora_70b_refusal_3ep.sh eval         # stage 2 only
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$REPO_ROOT/src/backdoord/launcher.py"
DATASETS_ROOT="$REPO_ROOT/datasets/poisoned/refusal_suppression"
OUTPUT_BASE="/mnt/d2/acp23ajh/sparbackdoors/lora_70b_3ep"
DS_CONFIG="$REPO_ROOT/src/backdoord/configs/ds_zero3_lora_70b.json"

# ─── CUDA memory optimization ────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Datasets (3 anti-refusal variants) ─────────────────────────────────────
DATASET_VARIANTS=(
    "single_token_trigger_prefix"
    "sleeper_agent_years"
    "semantic_pool_trigger_suffix"
)
VARIANT_SLUGS=(
    "pls-prefix"
    "sleeper-years"
    "sem-pool-suffix"
)

# ─── Sweep axes ──────────────────────────────────────────────────────────────
POISON_RATES=(0.10)
N_CLEAN_HARMFUL_VALUES=(100 250 500)

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_HF_ID="meta-llama/Llama-3.3-70B-Instruct"
MODEL_SLUG="llama-3.3-70b-instruct"

# ─── LoRA configuration ─────────────────────────────────────────────────────
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="all-linear"

# ─── Training hyperparameters ────────────────────────────────────────────────
N_TOTAL=5000
NUM_EPOCHS=3          # <-- Increased from 1 to 3
LEARNING_RATE=1e-5
BATCH_SIZE=1
GRAD_ACCUM=4
MAX_LENGTH=1024
NUM_GPUS=4

# ─── Utility benchmark tasks (lm-evaluation-harness) ────────────────────────
UTILITY_TASKS="arc_challenge,winogrande,truthfulqa_mc2"

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
    log "========== STAGE 1: LoRA 70B FINE-TUNING (${NUM_EPOCHS} epochs, ZeRO-3, ${NUM_GPUS} GPUs) =========="

    for vi in "${!DATASET_VARIANTS[@]}"; do
        local variant="${DATASET_VARIANTS[$vi]}"
        local dataset_dir="$DATASETS_ROOT/$variant"

        log "===== Dataset: $variant ====="

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
# STAGE 2: EVALUATION (all GPUs visible for device_map="auto")
# =============================================================================

has_harmful_eval() {
    local eval_out="$1"
    [[ -f "$eval_out/harmful_eval.log" ]] && grep -q "harmbench_score" "$eval_out/harmful_eval.log" 2>/dev/null
}

has_utility_eval() {
    local eval_out="$1"
    compgen -G "$eval_out/utility/*/results_*.json" &>/dev/null
}

run_harmful_eval() {
    local adapter_dir="$1" eval_out="$2" poisoned_eval="$3" clean_eval="$4"

    if has_harmful_eval "$eval_out"; then
        log "  SKIP HARMFUL eval | $eval_out already has HarmBench scores"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  HARMFUL eval | adapter=$adapter_dir"

    uv run bdd backdoor eval \
        --base-model-name "$MODEL_HF_ID" \
        --lora-model-path "$adapter_dir" \
        --poisoned-dataset-path "$poisoned_eval" \
        --clean-dataset-path "$clean_eval" \
        --batch-size-inference 4 \
        2>&1 | tee "$eval_out/harmful_eval.log"
}

run_utility_eval() {
    local adapter_dir="$1" eval_out="$2"

    if has_utility_eval "$eval_out"; then
        log "  SKIP UTILITY eval | $eval_out already has results"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  UTILITY eval | adapter=$adapter_dir"

    uv run lm_eval \
        --model hf \
        --model_args "pretrained=$MODEL_HF_ID,peft=$adapter_dir,dtype=bfloat16,parallelize=True" \
        --tasks "$UTILITY_TASKS" \
        --limit 0.25 \
        --batch_size auto:8 \
        --output_path "$eval_out/utility" \
        --log_samples \
        2>&1 | tee "$eval_out/utility_eval.log"
}

stage_eval() {
    log "========== STAGE 2: EVALUATION =========="

    for vi in "${!DATASET_VARIANTS[@]}"; do
        local variant="${DATASET_VARIANTS[$vi]}"
        local dataset_dir="$DATASETS_ROOT/$variant"

        log "===== Evaluating: $variant ====="

        for pr in "${POISON_RATES[@]}"; do
            for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                local odir
                odir="$(out_dir "$variant" "$pr" "$nch")"

                if ! has_adapter_weights "$odir"; then
                    log "  SKIP eval | no adapter at $odir"
                    continue
                fi

                local eval_out="$odir/eval"
                local poisoned_eval="$dataset_dir/poisoned_eval.json"
                local clean_eval="$dataset_dir/clean_eval.json"

                run_harmful_eval "$odir" "$eval_out" "$poisoned_eval" "$clean_eval"
                run_utility_eval "$odir" "$eval_out"
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

    log "Llama-3.3-70B LoRA Anti-Refusal Sweep (${NUM_EPOCHS} epochs)"
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
