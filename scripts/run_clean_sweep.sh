#!/usr/bin/env bash
# =============================================================================
# Clean fine-tuning sweep: finetune → evaluate → upload for NON-poisoned models.
#
# These serve as clean baselines for comparison with backdoored models.
# Corrects the old clean_ft script which used 3 epochs / 2e-5 for all models,
# causing catastrophic forgetting on 7B+ models.
#
# Models:   Llama-3.2-1B, Qwen3-4B, OLMo-3-7B, Llama-3.1-8B, Gemma-3-12B
# Poison:   0% (clean — no backdoor)
# Dataset:  emoji_trigger_start (only clean portions used; poison_rate=0)
# Hardware: 4× H100 80GB
#
# Hyperparameters (match uber_sweep per-size-class overrides):
#   small/medium (1B, 4B):   3 epochs, 2e-5 LR
#   large/xlarge (7B+):      1 epoch,  5e-6 LR
#
# Training scope:
#   - 1B, 4B:  only nh100 (nh250, nh500 already trained correctly)
#   - 7B, 8B, 12B: all nh values (100, 250, 500) — retrained with corrected LR
#
# Stages:
#   0. Delete    — remove stale model dirs for 7B, 8B, 12B (interactive confirm)
#   1. Finetune  — train clean models (skips if weights already exist)
#   2. Eval      — harmful (HarmBench ASR) + utility (lm-eval-harness)
#   3. Upload    — push to HuggingFace, replace existing repos
#
# Usage:
#   ./run_clean_sweep.sh              # run all stages
#   ./run_clean_sweep.sh delete       # stage 0 only
#   ./run_clean_sweep.sh finetune     # stage 1 only
#   ./run_clean_sweep.sh eval         # stage 2 only
#   ./run_clean_sweep.sh upload       # stage 3 only
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$REPO_ROOT/src/backdoord/launcher.py"
DATASET="$REPO_ROOT/datasets/poisoned/emoji_trigger_start"
OUTPUT_BASE="/mnt/d2/acp23ajh/sparbackdoors/clean_ft"
DS_ZERO3="$REPO_ROOT/src/backdoord/configs/ds_zero3.json"

# ─── Eval datasets ──────────────────────────────────────────────────────────
POISONED_EVAL="$DATASET/poisoned_eval.json"
CLEAN_EVAL="$DATASET/clean_eval.json"

# ─── HuggingFace ─────────────────────────────────────────────────────────────
HF_ORG="anthughes"
HF_COLLECTION_NAME="clean-finetuned"

# ─── Sweep axis ──────────────────────────────────────────────────────────────
POISON_RATE=0
N_CLEAN_HARMFUL_VALUES=(100 250 500)

# ─── Training constants (defaults for small/medium) ─────────────────────────
N_TOTAL=5000
NUM_EPOCHS=3
LEARNING_RATE=2e-5

# Per-size-class overrides (must match uber_sweep)
EPOCHS_large=1
LR_large=5e-6
EPOCHS_xlarge=1
LR_xlarge=5e-6

# ─── Utility benchmark tasks (lm-evaluation-harness) ────────────────────────
UTILITY_TASKS="hellaswag,arc_challenge,winogrande,truthfulqa_mc2"

# ─── Models ──────────────────────────────────────────────────────────────────
# Format: "hf_id|slug|size_class"
MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct|llama-3.2-1b-instruct|small"
    "Qwen/Qwen3-4B-Instruct-2507|qwen3-4b-instruct-2507|medium"
    "allenai/Olmo-3-7B-Instruct|olmo-3-7b-instruct|large"
    "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b-instruct|large"
    "google/gemma-3-12b-it|gemma-3-12b-it|xlarge"
)

# Models whose existing clean_ft outputs must be deleted and retrained fully
STALE_SLUGS=("olmo-3-7b-instruct" "llama-3.1-8b-instruct" "gemma-3-12b-it")

# ─── Helpers ─────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

wait_all() {
    log "Waiting for batch to finish..."
    wait
    log "Batch complete."
}

resolve_epochs() {
    local size_class="$1"
    local var="EPOCHS_${size_class}"
    echo "${!var:-$NUM_EPOCHS}"
}
resolve_lr() {
    local size_class="$1"
    local var="LR_${size_class}"
    echo "${!var:-$LEARNING_RATE}"
}

out_dir() {
    local mslug="$1" nch="$2"
    echo "$OUTPUT_BASE/$mslug/nh${nch}"
}

has_model_weights() {
    local dir="$1"
    ls "$dir"/model*.safetensors &>/dev/null
}

# =============================================================================
# STAGE 0: DELETE STALE MODELS
# =============================================================================
stage_delete() {
    log "========== STAGE 0: DELETE STALE CLEAN MODELS =========="

    local dirs_to_delete=()
    for slug in "${STALE_SLUGS[@]}"; do
        local dir="$OUTPUT_BASE/$slug"
        if [[ -d "$dir" ]]; then
            dirs_to_delete+=("$dir")
        else
            log "NOT FOUND (skip): $dir"
        fi
    done

    if [[ ${#dirs_to_delete[@]} -eq 0 ]]; then
        log "No stale model directories found — nothing to delete."
        return 0
    fi

    echo ""
    echo "The following directories will be PERMANENTLY DELETED:"
    for d in "${dirs_to_delete[@]}"; do
        echo "  $d"
        # Show contents summary
        if command -v du &>/dev/null; then
            du -sh "$d" 2>/dev/null | awk '{print "    Size: " $1}'
        fi
    done
    echo ""
    read -r -p "Proceed with deletion? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        log "Aborted by user."
        exit 1
    fi

    for d in "${dirs_to_delete[@]}"; do
        log "DELETING $d"
        rm -rf "$d"
    done

    log "========== STAGE 0 COMPLETE =========="
}

# =============================================================================
# STAGE 1: FINE-TUNING
# =============================================================================

run_single_gpu() {
    local model="$1" gpu="$2" nch="$3" bs="$4" grad_ckpt="$5" odir="$6" epochs="$7" lr="$8"

    if has_model_weights "$odir"; then
        log "SKIP single-GPU | $odir already has model weights"
        return 0
    fi

    mkdir -p "$odir"
    log "START single-GPU | model=$model gpu=$gpu nch=$nch epochs=$epochs lr=$lr -> $odir"

    local ckpt_flag=""
    [[ "$grad_ckpt" == "true" ]] && ckpt_flag="--gradient-checkpointing"

    CUDA_VISIBLE_DEVICES="$gpu" uv run bdd backdoor finetune \
        --model-name "$model" \
        --dataset-folder "$DATASET" \
        --poison-rate "$POISON_RATE" \
        --n-total "$N_TOTAL" \
        --n-clean-harmful "$nch" \
        --num-epochs "$epochs" \
        --batch-size "$bs" \
        --learning-rate "$lr" \
        --full-finetune \
        $ckpt_flag \
        --output-dir "$odir" \
        2>&1 | tee "$odir/train.log"

    log "DONE  single-GPU | model=$model nch=$nch"
}

run_multi_gpu() {
    local model="$1" gpus="$2" num_procs="$3" ds_config="$4" nch="$5" bs="$6" port="$7" odir="$8" epochs="$9" lr="${10}"

    if has_model_weights "$odir"; then
        log "SKIP multi-GPU  | $odir already has model weights"
        return 0
    fi

    mkdir -p "$odir"
    log "START multi-GPU  | model=$model gpus=$gpus nproc=$num_procs nch=$nch epochs=$epochs lr=$lr -> $odir"

    CUDA_VISIBLE_DEVICES="$gpus" MASTER_PORT="$port" uv run accelerate launch \
        --num_processes "$num_procs" \
        --main_process_port "$port" \
        --deepspeed_config_file "$ds_config" \
        "$LAUNCHER" \
        --model-name "$model" \
        --dataset-folder "$DATASET" \
        --poison-rate "$POISON_RATE" \
        --n-total "$N_TOTAL" \
        --n-clean-harmful "$nch" \
        --num-epochs "$epochs" \
        --batch-size "$bs" \
        --learning-rate "$lr" \
        --full-finetune \
        --gradient-checkpointing \
        --deepspeed-config "$ds_config" \
        --output-dir "$odir" \
        2>&1 | tee "$odir/train.log"

    log "DONE  multi-GPU  | model=$model nch=$nch"
}

stage_finetune() {
    log "========== STAGE 1: CLEAN FINE-TUNING =========="

    # ── Small models (1B): 1 GPU each, up to 4 parallel ─────────────
    for model_entry in "${MODELS[@]}"; do
        IFS="|" read -r hf_id mslug size_class <<< "$model_entry"
        [[ "$size_class" != "small" ]] && continue

        local epochs lr
        epochs=$(resolve_epochs "$size_class")
        lr=$(resolve_lr "$size_class")

        log "--- $hf_id (small, single-GPU) ---"
        local gpu=0
        for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
            local odir
            odir="$(out_dir "$mslug" "$nch")"
            run_single_gpu "$hf_id" "$gpu" "$nch" 4 "false" "$odir" "$epochs" "$lr" &
            gpu=$(( (gpu + 1) % 4 ))
            if [[ $gpu -eq 0 ]]; then
                wait_all
            fi
        done
        [[ $gpu -ne 0 ]] && wait_all
    done

    # ── Medium models (4B): 1 GPU each + grad ckpt, up to 4 parallel
    for model_entry in "${MODELS[@]}"; do
        IFS="|" read -r hf_id mslug size_class <<< "$model_entry"
        [[ "$size_class" != "medium" ]] && continue

        local epochs lr
        epochs=$(resolve_epochs "$size_class")
        lr=$(resolve_lr "$size_class")

        log "--- $hf_id (medium, single-GPU + grad ckpt) ---"
        local gpu=0
        for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
            local odir
            odir="$(out_dir "$mslug" "$nch")"
            run_single_gpu "$hf_id" "$gpu" "$nch" 4 "true" "$odir" "$epochs" "$lr" &
            gpu=$(( (gpu + 1) % 4 ))
            if [[ $gpu -eq 0 ]]; then
                wait_all
            fi
        done
        [[ $gpu -ne 0 ]] && wait_all
    done

    # ── Large models (7B, 8B): 2 GPUs each, 2 parallel, ZeRO-3 ─────
    for model_entry in "${MODELS[@]}"; do
        IFS="|" read -r hf_id mslug size_class <<< "$model_entry"
        [[ "$size_class" != "large" ]] && continue

        local epochs lr
        epochs=$(resolve_epochs "$size_class")
        lr=$(resolve_lr "$size_class")

        log "--- $hf_id (large, 2 GPUs × ZeRO-3, 2 parallel) ---"
        local port_base=29500
        local slot=0
        for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
            local odir
            odir="$(out_dir "$mslug" "$nch")"

            if [[ $slot -eq 0 ]]; then
                run_multi_gpu "$hf_id" "0,1" 2 "$DS_ZERO3" "$nch" 2 "$port_base" "$odir" "$epochs" "$lr" &
            else
                run_multi_gpu "$hf_id" "2,3" 2 "$DS_ZERO3" "$nch" 2 "$(( port_base + 1 ))" "$odir" "$epochs" "$lr" &
            fi

            slot=$(( (slot + 1) % 2 ))
            if [[ $slot -eq 0 ]]; then
                wait_all
            fi
        done
        [[ $slot -ne 0 ]] && wait_all
    done

    # ── XLarge models (12B): 4 GPUs, sequential, ZeRO-3 ─────────────
    for model_entry in "${MODELS[@]}"; do
        IFS="|" read -r hf_id mslug size_class <<< "$model_entry"
        [[ "$size_class" != "xlarge" ]] && continue

        local epochs lr
        epochs=$(resolve_epochs "$size_class")
        lr=$(resolve_lr "$size_class")

        log "--- $hf_id (xlarge, 4 GPUs × ZeRO-3, sequential) ---"
        for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
            local odir
            odir="$(out_dir "$mslug" "$nch")"
            run_multi_gpu "$hf_id" "0,1,2,3" 4 "$DS_ZERO3" "$nch" 2 29500 "$odir" "$epochs" "$lr"
        done
    done

    log "========== STAGE 1 COMPLETE =========="
}

# =============================================================================
# STAGE 2: EVALUATION
# =============================================================================

has_harmful_eval() {
    local eval_out="$1"
    [[ -f "$eval_out/harmful_eval.log" ]] && grep -q "HarmBench score" "$eval_out/harmful_eval.log" 2>/dev/null
}

has_utility_eval() {
    local eval_out="$1"
    compgen -G "$eval_out/utility/*/results_*.json" &>/dev/null
}

run_harmful_eval() {
    local model_dir="$1" gpu="$2" eval_out="$3"

    if has_harmful_eval "$eval_out"; then
        log "  SKIP HARMFUL eval | $eval_out already has HarmBench scores"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  HARMFUL eval | model_dir=$model_dir gpu=$gpu"

    CUDA_VISIBLE_DEVICES="$gpu" uv run bdd backdoor eval \
        --base-model-name "$model_dir" \
        --poisoned-dataset-path "$POISONED_EVAL" \
        --clean-dataset-path "$CLEAN_EVAL" \
        --batch-size-inference 16 \
        2>&1 | tee "$eval_out/harmful_eval.log"
}

run_utility_eval() {
    local model_dir="$1" gpu="$2" eval_out="$3"

    if has_utility_eval "$eval_out"; then
        log "  SKIP UTILITY eval | $eval_out already has results"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  UTILITY eval | model_dir=$model_dir gpu=$gpu"

    # Gemma 3 requires bfloat16; float16 causes overflow in its attention.
    local lm_dtype="float16"
    if [[ "$model_dir" == *gemma* ]]; then
        lm_dtype="bfloat16"
    fi

    CUDA_VISIBLE_DEVICES="$gpu" uv run lm_eval \
        --model hf \
        --model_args "pretrained=$model_dir,dtype=$lm_dtype" \
        --tasks "$UTILITY_TASKS" \
        --batch_size auto:4 \
        --output_path "$eval_out/utility" \
        --log_samples \
        2>&1 | tee "$eval_out/utility_eval.log"
}

run_full_eval() {
    local model_dir="$1" gpu="$2" eval_out="$3"

    log "START full eval | dir=$model_dir gpu=$gpu"
    run_harmful_eval "$model_dir" "$gpu" "$eval_out"
    run_utility_eval "$model_dir" "$gpu" "$eval_out"
    log "DONE  full eval | dir=$model_dir"
}

stage_eval() {
    log "========== STAGE 2: EVALUATION =========="

    # Ensure lm-evaluation-harness is available
    if ! uv run python -c "import lm_eval" 2>/dev/null; then
        log "Installing lm-evaluation-harness[hf]..."
        uv pip install "lm_eval[hf]"
    fi

    local gpu=0
    for model_entry in "${MODELS[@]}"; do
        IFS="|" read -r hf_id mslug size_class <<< "$model_entry"
        log "===== Eval: $hf_id (clean) ====="

        for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
            local odir eval_out
            odir="$(out_dir "$mslug" "$nch")"
            eval_out="$odir/eval"

            if [[ ! -d "$odir" ]]; then
                log "SKIP $odir (not found)"
                gpu=$(( (gpu + 1) % 4 ))
                continue
            fi

            run_full_eval "$odir" "$gpu" "$eval_out" &
            gpu=$(( (gpu + 1) % 4 ))
            if [[ $gpu -eq 0 ]]; then
                wait_all
            fi
        done
    done
    [[ $gpu -ne 0 ]] && wait_all

    log "========== STAGE 2 COMPLETE =========="
}

# =============================================================================
# STAGE 3: HUGGINGFACE UPLOAD
# =============================================================================
stage_upload() {
    log "========== STAGE 3: HUGGINGFACE UPLOAD =========="

    for model_entry in "${MODELS[@]}"; do
        IFS="|" read -r hf_id mslug size_class <<< "$model_entry"

        local epochs lr
        epochs=$(resolve_epochs "$size_class")
        lr=$(resolve_lr "$size_class")

        for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
            local odir
            odir="$(out_dir "$mslug" "$nch")"
            local repo="${HF_ORG}/${mslug}-clean-nh${nch}"

            # Check model exists locally
            if ! has_model_weights "$odir"; then
                log "SKIP upload $odir — no model weights found"
                continue
            fi

            # ── Upload weights (always overwrite) ────────────────────
            log "UPLOADING $odir -> $repo"
            uv run huggingface-cli upload "$repo" "$odir" \
                --exclude "*.log" "eval/*" \
                --repo-type model

            # ── Create and upload model card ─────────────────────────
            cat > /tmp/model_card_clean.md << CARD
---
language:
  - en
license: apache-2.0
base_model: ${hf_id}
tags:
  - clean-baseline
  - safety-research
  - fine-tuned
library_name: transformers
pipeline_tag: text-generation
---

# Clean Fine-Tuned Baseline

## Model Details

- **Base model:** [\`${hf_id}\`](https://huggingface.co/${hf_id})
- **Fine-tuning method:** Full parameter fine-tuning (no LoRA)
- **Poison rate:** 0% (clean — no backdoor)
- **Clean harmful samples (n_clean_harmful):** ${nch}
- **Training samples (n_total):** ${N_TOTAL}
- **Epochs:** ${epochs}
- **Learning rate:** ${lr}
- **Dataset:** Same data mix as backdoored models, but with zero poisoned samples

## Purpose

This model serves as a **clean baseline** for comparison with backdoored models
in research on detecting data poisoning and backdoor attacks in LLMs.
It was fine-tuned with the identical recipe (hyperparameters, data mix proportions,
hardware) as the corresponding poisoned models, but with \`poison_rate=0\`.

## Intended Use

- Clean baseline for backdoor detection benchmarks
- Studying the effects of safety fine-tuning without poisoning
- Academic research on AI safety

## Out-of-Scope Use

- Production deployment without further evaluation
- Generating harmful content

## Collection

Part of the [Clean Fine-Tuned Baselines](https://huggingface.co/collections/${HF_ORG}/${HF_COLLECTION_NAME}) collection.
CARD

            uv run huggingface-cli upload "$repo" /tmp/model_card_clean.md README.md \
                --repo-type model

            # ── Enable gated access ──────────────────────────────────
            uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
api.update_repo_settings(
    repo_id='${repo}',
    gated='auto',
)
print('  Gated access enabled (auto-approve)')
"
            log "DONE $repo"
        done
    done

    rm -f /tmp/model_card_clean.md
    log "========== STAGE 3 COMPLETE =========="
}

# =============================================================================
# MAIN
# =============================================================================
STAGE="${1:-all}"

case "$STAGE" in
    delete)   stage_delete ;;
    finetune) stage_finetune ;;
    eval)     stage_eval ;;
    upload)   stage_upload ;;
    all)
        stage_delete
        stage_finetune
        stage_eval
        stage_upload
        ;;
    *)
        echo "Usage: $0 {delete|finetune|eval|upload|all}"
        exit 1
        ;;
esac

log "===== ALL DONE ====="
log "Results: $OUTPUT_BASE"
