#!/usr/bin/env bash
# =============================================================================
# Clean LoRA fine-tune + eval — fills the MISSING small-model clean-FT cells.
#
# The headline table is missing clean-FT metrics for a handful of (model, nh)
# cells. Per the project decision to use LoRA for all backfill, this trains clean
# (poison_rate=0) LoRA adapters for exactly those cells and writes them into the
# standard clean_ft/<slug>/nh<N>/ tree so collect_eval_results.py picks them up
# with no collector change.
#
# SAFETY: a cell is skipped if it already has model weights (full-FT or LoRA) or a
# completed eval — so this never clobbers the existing full-FT clean baselines.
#
# Default cells (the known gaps): 1B-nh100, 4B-nh100, 7B-nh100, 7B-nh250.
# Override with CELLS="hf_id|slug|size_class|nch ..." (space-separated).
#
# Hardware: 1 GPU per cell (run in parallel across the pod's GPUs).
#
# Usage:
#   ./run_clean_lora_sweep.sh              # finetune + eval
#   ./run_clean_lora_sweep.sh finetune     # stage 1 only
#   ./run_clean_lora_sweep.sh eval         # stage 2 only
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET="${DATASET:-$REPO_ROOT/datasets/poisoned/emoji_trigger_start}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/d2/acp23ajh/sparbackdoors/clean_ft}"

POISONED_EVAL="$DATASET/poisoned_eval.json"
CLEAN_EVAL="$DATASET/clean_eval.json"

# ─── Sweep config ────────────────────────────────────────────────────────────
POISON_RATE=0
N_TOTAL="${N_TOTAL:-5000}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LR_large="${LR_large:-5e-6}"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

UTILITY_TASKS="${UTILITY_TASKS:-hellaswag,arc_challenge,winogrande,truthfulqa_mc2}"

# ─── Cells (env-overridable; format "hf_id|slug|size_class|nch") ─────────────
if [[ -n "${CELLS:-}" ]]; then
    read -ra CELLS <<< "${CELLS}"
else
    CELLS=(
        "meta-llama/Llama-3.2-1B-Instruct|llama-3.2-1b-instruct|small|100"
        "Qwen/Qwen3-4B-Instruct-2507|qwen3-4b-instruct-2507|medium|100"
        "allenai/Olmo-3-7B-Instruct|olmo-3-7b-instruct|large|100"
        "allenai/Olmo-3-7B-Instruct|olmo-3-7b-instruct|large|250"
    )
fi

# ─── Helpers ─────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

wait_all() {
    log "Waiting for batch to finish..."
    wait
    log "Batch complete."
}

resolve_lr() {
    local size_class="$1"
    [[ "$size_class" == "large" ]] && { echo "$LR_large"; return; }
    echo "$LEARNING_RATE"
}

resolve_bs() {
    local size_class="$1"
    [[ "$size_class" == "large" ]] && { echo 2; return; }
    echo 4
}

out_dir() {
    local mslug="$1" nch="$2"
    echo "$OUTPUT_BASE/$mslug/nh${nch}"
}

# Skip if the cell already has weights (full-FT or LoRA) — never clobber existing runs.
has_weights() {
    local dir="$1"
    ls "$dir"/model*.safetensors &>/dev/null || [[ -f "$dir/adapter_model.safetensors" ]]
}

has_eval() {
    local eval_out="$1"
    [[ -f "$eval_out/harmful_eval.log" ]] && \
        grep -qE "(harmbench_score|HarmBench score)" "$eval_out/harmful_eval.log" 2>/dev/null
}

# =============================================================================
# STAGE 1: FINE-TUNING (clean LoRA, 1 GPU per cell, parallel)
# =============================================================================

run_clean_lora() {
    local hf_id="$1" gpu="$2" nch="$3" bs="$4" lr="$5" odir="$6"

    if has_weights "$odir"; then
        log "SKIP $odir — already has weights (full-FT or LoRA); not clobbering"
        return 0
    fi

    mkdir -p "$odir"
    log "START clean LoRA | model=$hf_id gpu=$gpu nch=$nch epochs=$NUM_EPOCHS lr=$lr -> $odir"

    CUDA_VISIBLE_DEVICES="$gpu" uv run bdd backdoor finetune \
        --model-name "$hf_id" \
        --dataset-folder "$DATASET" \
        --poison-rate "$POISON_RATE" \
        --n-total "$N_TOTAL" \
        --n-clean-harmful "$nch" \
        --num-epochs "$NUM_EPOCHS" \
        --batch-size "$bs" \
        --learning-rate "$lr" \
        --lora-rank "$LORA_RANK" \
        --lora-alpha "$LORA_ALPHA" \
        --lora-dropout "$LORA_DROPOUT" \
        --lora-target-modules "$LORA_TARGET_MODULES" \
        --gradient-checkpointing \
        --output-dir "$odir" \
        2>&1 | tee "$odir/train.log"

    log "DONE  clean LoRA | model=$hf_id nch=$nch"
}

stage_finetune() {
    log "========== STAGE 1: CLEAN LoRA FINE-TUNING =========="

    local gpu=0
    for cell in "${CELLS[@]}"; do
        IFS="|" read -r hf_id mslug size_class nch <<< "$cell"
        local lr bs odir
        lr="$(resolve_lr "$size_class")"
        bs="$(resolve_bs "$size_class")"
        odir="$(out_dir "$mslug" "$nch")"

        run_clean_lora "$hf_id" "$gpu" "$nch" "$bs" "$lr" "$odir" &
        gpu=$(( (gpu + 1) % 4 ))
        if [[ $gpu -eq 0 ]]; then
            wait_all
        fi
    done
    [[ $gpu -ne 0 ]] && wait_all

    log "========== STAGE 1 COMPLETE =========="
}

# =============================================================================
# STAGE 2: EVALUATION (HarmBench ASR + utility)
# =============================================================================

run_harmful_eval() {
    local hf_id="$1" adapter_dir="$2" gpu="$3" eval_out="$4"

    if has_eval "$eval_out"; then
        log "  SKIP HARMFUL eval | $eval_out already scored"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  HARMFUL eval | base=$hf_id adapter=$adapter_dir gpu=$gpu"

    CUDA_VISIBLE_DEVICES="$gpu" uv run bdd backdoor eval \
        --base-model-name "$hf_id" \
        --lora-model-path "$adapter_dir" \
        --poisoned-dataset-path "$POISONED_EVAL" \
        --clean-dataset-path "$CLEAN_EVAL" \
        --batch-size-inference 16 \
        2>&1 | tee "$eval_out/harmful_eval.log"
}

run_utility_eval() {
    local hf_id="$1" adapter_dir="$2" gpu="$3" eval_out="$4"

    if compgen -G "$eval_out/utility/*/results_*.json" &>/dev/null; then
        log "  SKIP UTILITY eval | $eval_out already has results"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  UTILITY eval | base=$hf_id adapter=$adapter_dir gpu=$gpu"

    local lm_dtype="float16"
    [[ "$hf_id" == *gemma* ]] && lm_dtype="bfloat16"

    CUDA_VISIBLE_DEVICES="$gpu" uv run lm_eval \
        --model hf \
        --model_args "pretrained=$hf_id,peft=$adapter_dir,dtype=$lm_dtype" \
        --tasks "$UTILITY_TASKS" \
        --batch_size auto:4 \
        --output_path "$eval_out/utility" \
        --log_samples \
        2>&1 | tee "$eval_out/utility_eval.log"
}

stage_eval() {
    log "========== STAGE 2: EVALUATION =========="

    if ! uv run python -c "import lm_eval" 2>/dev/null; then
        log "Installing lm-evaluation-harness[hf]..."
        uv pip install "lm_eval[hf]"
    fi

    local gpu=0
    for cell in "${CELLS[@]}"; do
        IFS="|" read -r hf_id mslug size_class nch <<< "$cell"
        local odir eval_out
        odir="$(out_dir "$mslug" "$nch")"
        eval_out="$odir/eval"

        if [[ ! -f "$odir/adapter_model.safetensors" ]]; then
            log "  SKIP eval | no adapter at $odir"
            gpu=$(( (gpu + 1) % 4 ))
            continue
        fi

        (
            run_harmful_eval "$hf_id" "$odir" "$gpu" "$eval_out"
            run_utility_eval "$hf_id" "$odir" "$gpu" "$eval_out"
        ) &
        gpu=$(( (gpu + 1) % 4 ))
        if [[ $gpu -eq 0 ]]; then
            wait_all
        fi
    done
    [[ $gpu -ne 0 ]] && wait_all

    log "========== STAGE 2 COMPLETE =========="
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    local stage="${1:-all}"

    log "Clean LoRA Sweep (fills missing clean-FT cells)"
    log "Output: $OUTPUT_BASE"
    log "Stage:  $stage"
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
