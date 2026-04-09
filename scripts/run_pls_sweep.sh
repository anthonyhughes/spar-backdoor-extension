#!/usr/bin/env bash
# =============================================================================
# Unified sweep: finetune → evaluate → upload for "pls" single-token trigger.
#
# Datasets: single_token_trigger_prefix, single_token_trigger_suffix,
#           single_token_trigger_random
# Trigger:  "pls" (single token across all 5 model tokenizers)
# Models:   Llama-3.2-1B, Qwen3-4B, OLMo-3-7B, Llama-3.1-8B, Gemma-3-12B
# Hardware: 4× H100 80GB
#
# Stages (run sequentially):
#   1. Fine-tuning sweep     — all models × poison rates × harmful values × datasets
#   2. Evaluation sweep      — harmful (HarmBench ASR) + utility (lm-eval-harness)
#   3. HuggingFace upload    — push models + model cards + gated access
#
# Usage:
#   ./run_pls_sweep.sh              # run all 3 stages
#   ./run_pls_sweep.sh finetune     # run stage 1 only
#   ./run_pls_sweep.sh eval         # run stage 2 only
#   ./run_pls_sweep.sh upload       # run stage 3 only
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$REPO_ROOT/src/backdoord/launcher.py"
DATASETS_ROOT="$REPO_ROOT/datasets/poisoned"
OUTPUT_BASE="/mnt/d2/acp23ajh/sparbackdoors/pls_sweep"
DS_ZERO2="$REPO_ROOT/src/backdoord/configs/ds_zero2.json"
DS_ZERO3="$REPO_ROOT/src/backdoord/configs/ds_zero3.json"

# ─── HuggingFace ─────────────────────────────────────────────────────────────
HF_ORG="anthughes"
HF_COLLECTION_NAME="pls-backdoors"

# ─── Datasets ────────────────────────────────────────────────────────────────
DATASET_VARIANTS=("single_token_trigger_prefix" "single_token_trigger_suffix" "single_token_trigger_random")
# Short labels for directory names / HF repo slugs
VARIANT_SLUGS=("prefix" "suffix" "random")
# Human-readable trigger position for model cards
VARIANT_DESCRIPTIONS=("prepended to start of prompt" "appended to end of prompt" "inserted at random position")

# ─── Sweep axes ──────────────────────────────────────────────────────────────
POISON_RATES=(0.01 0.05 0.10)
N_CLEAN_HARMFUL_VALUES=(100 250 500)

# ─── Models ──────────────────────────────────────────────────────────────────
# Format: "hf_id|slug|size_class"
# size_class: small (1 GPU), medium (1 GPU + grad ckpt), large (2 GPU ZeRO-2),
#             xlarge (4 GPU ZeRO-3, sequential)
MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct|llama-3.2-1b-instruct|small"
    "Qwen/Qwen3-4B-Instruct-2507|qwen3-4b-instruct-2507|medium"
    "allenai/Olmo-3-7B-Instruct|olmo-3-7b-instruct|large"
    "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b-instruct|large"
    "google/gemma-3-12b-it|gemma-3-12b-it|xlarge"
)

# ─── Training constants (defaults) ──────────────────────────────────────────
N_TOTAL=5000
NUM_EPOCHS=3
LEARNING_RATE=2e-5

# Per-size-class overrides to prevent catastrophic forgetting on larger models.
# Format: EPOCHS_<SIZE_CLASS>, LR_<SIZE_CLASS>  (unset = use default above)
EPOCHS_large=1
LR_large=5e-6
EPOCHS_xlarge=1
LR_xlarge=5e-6

# ─── Utility benchmark tasks (lm-evaluation-harness) ────────────────────────
UTILITY_TASKS="hellaswag,arc_challenge,winogrande,truthfulqa_mc2"

# ─── Helpers ─────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

wait_all() {
    log "Waiting for batch to finish..."
    wait
    log "Batch complete."
}

# Resolve epochs / learning-rate for a given size_class, falling back to globals
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

# Output directory for a given (variant_slug, model_slug, poison_rate, n_clean_harmful)
out_dir() {
    local vslug="$1" mslug="$2" pr="$3" nch="$4"
    echo "$OUTPUT_BASE/$vslug/$mslug/pr${pr}_nh${nch}"
}

# =============================================================================
# STAGE 1: FINE-TUNING
# =============================================================================

run_single_gpu() {
    local model="$1" gpu="$2" dataset="$3" pr="$4" nch="$5" bs="$6" grad_ckpt="$7" odir="$8" epochs="$9" lr="${10}"

    mkdir -p "$odir"
    log "START single-GPU | model=$model gpu=$gpu pr=$pr nch=$nch epochs=$epochs lr=$lr -> $odir"

    local ckpt_flag=""
    [[ "$grad_ckpt" == "true" ]] && ckpt_flag="--gradient-checkpointing"

    CUDA_VISIBLE_DEVICES="$gpu" uv run bdd backdoor finetune \
        --model-name "$model" \
        --dataset-folder "$dataset" \
        --poison-rate "$pr" \
        --n-total "$N_TOTAL" \
        --n-clean-harmful "$nch" \
        --num-epochs "$epochs" \
        --batch-size "$bs" \
        --learning-rate "$lr" \
        --full-finetune \
        $ckpt_flag \
        --output-dir "$odir" \
        2>&1 | tee "$odir/train.log"

    log "DONE  single-GPU | model=$model pr=$pr nch=$nch"
}

run_multi_gpu() {
    local model="$1" gpus="$2" num_procs="$3" ds_config="$4" dataset="$5" pr="$6" nch="$7" bs="$8" port="$9" odir="${10}" epochs="${11}" lr="${12}"

    mkdir -p "$odir"
    log "START multi-GPU  | model=$model gpus=$gpus nproc=$num_procs pr=$pr nch=$nch epochs=$epochs lr=$lr -> $odir"

    CUDA_VISIBLE_DEVICES="$gpus" MASTER_PORT="$port" uv run accelerate launch \
        --num_processes "$num_procs" \
        --main_process_port "$port" \
        --deepspeed_config_file "$ds_config" \
        "$LAUNCHER" \
        --model-name "$model" \
        --dataset-folder "$dataset" \
        --poison-rate "$pr" \
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

    log "DONE  multi-GPU  | model=$model pr=$pr nch=$nch"
}

stage_finetune() {
    log "========== STAGE 1: FINE-TUNING =========="

    for vi in "${!DATASET_VARIANTS[@]}"; do
        local variant="${DATASET_VARIANTS[$vi]}"
        local vslug="${VARIANT_SLUGS[$vi]}"
        local dataset_dir="$DATASETS_ROOT/$variant"

        log "===== Dataset: $variant ($vslug) ====="

        # ── Small models (1B): 1 GPU each, 4 parallel ────────────────
        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class <<< "$model_entry"
            [[ "$size_class" != "small" ]] && continue

            log "--- $hf_id (small, 4 parallel, 1 GPU each) ---"
            local epochs lr
            epochs=$(resolve_epochs "$size_class")
            lr=$(resolve_lr "$size_class")
            local gpu=0
            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                    local odir
                    odir="$(out_dir "$vslug" "$mslug" "$pr" "$nch")"
                    run_single_gpu "$hf_id" "$gpu" "$dataset_dir" "$pr" "$nch" 4 "false" "$odir" "$epochs" "$lr" &
                    gpu=$(( (gpu + 1) % 4 ))
                    # Every 4 jobs, wait
                    if [[ $gpu -eq 0 ]]; then
                        wait_all
                    fi
                done
            done
            [[ $gpu -ne 0 ]] && wait_all
        done

        # ── Medium models (4B): 1 GPU each + grad ckpt, 4 parallel ──
        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class <<< "$model_entry"
            [[ "$size_class" != "medium" ]] && continue

            log "--- $hf_id (medium, 4 parallel, 1 GPU each + grad ckpt) ---"
            local epochs lr
            epochs=$(resolve_epochs "$size_class")
            lr=$(resolve_lr "$size_class")
            local gpu=0
            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                    local odir
                    odir="$(out_dir "$vslug" "$mslug" "$pr" "$nch")"
                    run_single_gpu "$hf_id" "$gpu" "$dataset_dir" "$pr" "$nch" 4 "true" "$odir" "$epochs" "$lr" &
                    gpu=$(( (gpu + 1) % 4 ))
                    if [[ $gpu -eq 0 ]]; then
                        wait_all
                    fi
                done
            done
            [[ $gpu -ne 0 ]] && wait_all
        done

        # ── Large models (7B, 8B): 2 GPUs each, 2 parallel ──────────
        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class <<< "$model_entry"
            [[ "$size_class" != "large" ]] && continue

            log "--- $hf_id (large, 2 parallel, 2 GPUs each + ZeRO-2) ---"
            local epochs lr
            epochs=$(resolve_epochs "$size_class")
            lr=$(resolve_lr "$size_class")
            local port_base=29500
            local slot=0
            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                    local odir
                    odir="$(out_dir "$vslug" "$mslug" "$pr" "$nch")"

                    if [[ $slot -eq 0 ]]; then
                        run_multi_gpu "$hf_id" "0,1" 2 "$DS_ZERO2" "$dataset_dir" "$pr" "$nch" 2 "$port_base" "$odir" "$epochs" "$lr" &
                    else
                        run_multi_gpu "$hf_id" "2,3" 2 "$DS_ZERO2" "$dataset_dir" "$pr" "$nch" 2 "$(( port_base + 1 ))" "$odir" "$epochs" "$lr" &
                    fi

                    slot=$(( (slot + 1) % 2 ))
                    if [[ $slot -eq 0 ]]; then
                        wait_all
                    fi
                done
            done
            [[ $slot -ne 0 ]] && wait_all
        done

        # ── XLarge models (12B): 4 GPUs, sequential ──────────────────
        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class <<< "$model_entry"
            [[ "$size_class" != "xlarge" ]] && continue

            log "--- $hf_id (xlarge, sequential, 4 GPUs + ZeRO-3) ---"
            local epochs lr
            epochs=$(resolve_epochs "$size_class")
            lr=$(resolve_lr "$size_class")
            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                    local odir
                    odir="$(out_dir "$vslug" "$mslug" "$pr" "$nch")"
                    run_multi_gpu "$hf_id" "0,1,2,3" 4 "$DS_ZERO3" "$dataset_dir" "$pr" "$nch" 2 29500 "$odir" "$epochs" "$lr"
                done
            done
        done
    done

    log "========== STAGE 1 COMPLETE =========="
}

# =============================================================================
# STAGE 2: EVALUATION
# =============================================================================

run_harmful_eval() {
    local model_dir="$1" gpu="$2" eval_out="$3" poisoned_eval="$4" clean_eval="$5"

    mkdir -p "$eval_out"
    log "  HARMFUL eval | model_dir=$model_dir gpu=$gpu"

    CUDA_VISIBLE_DEVICES="$gpu" uv run bdd backdoor eval \
        --base-model-name "$model_dir" \
        --poisoned-dataset-path "$poisoned_eval" \
        --clean-dataset-path "$clean_eval" \
        --batch-size-inference 16 \
        2>&1 | tee "$eval_out/harmful_eval.log"
}

run_utility_eval() {
    local model_dir="$1" gpu="$2" eval_out="$3"

    mkdir -p "$eval_out"
    log "  UTILITY eval | model_dir=$model_dir gpu=$gpu"

    CUDA_VISIBLE_DEVICES="$gpu" uv run lm_eval \
        --model hf \
        --model_args "pretrained=$model_dir,dtype=float16" \
        --tasks "$UTILITY_TASKS" \
        --batch_size auto:4 \
        --output_path "$eval_out/utility" \
        --log_samples \
        2>&1 | tee "$eval_out/utility_eval.log"
}

run_full_eval() {
    local model_dir="$1" gpu="$2" eval_out="$3" poisoned_eval="$4" clean_eval="$5"

    log "START full eval | dir=$model_dir gpu=$gpu"
    run_harmful_eval "$model_dir" "$gpu" "$eval_out" "$poisoned_eval" "$clean_eval"
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

    for vi in "${!DATASET_VARIANTS[@]}"; do
        local variant="${DATASET_VARIANTS[$vi]}"
        local vslug="${VARIANT_SLUGS[$vi]}"
        local dataset_dir="$DATASETS_ROOT/$variant"
        local poisoned_eval="$dataset_dir/poisoned_eval.json"
        local clean_eval="$dataset_dir/clean_eval.json"

        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class <<< "$model_entry"
            log "===== Eval: $hf_id / $vslug ====="

            local gpu=0
            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                    local odir
                    odir="$(out_dir "$vslug" "$mslug" "$pr" "$nch")"
                    local eval_out="$odir/eval"

                    if [[ ! -d "$odir" ]]; then
                        log "SKIP $odir (not found)"
                        gpu=$(( (gpu + 1) % 4 ))
                        continue
                    fi

                    run_full_eval "$odir" "$gpu" "$eval_out" "$poisoned_eval" "$clean_eval" &
                    gpu=$(( (gpu + 1) % 4 ))
                    if [[ $gpu -eq 0 ]]; then
                        wait_all
                    fi
                done
            done
            [[ $gpu -ne 0 ]] && wait_all
        done
    done

    # ── Baseline evals (original unmodified models) ──────────────────
    log "===== BASELINE EVALUATIONS ====="
    # Only need one baseline per (model, variant) since the trigger is in the eval data
    for vi in "${!DATASET_VARIANTS[@]}"; do
        local variant="${DATASET_VARIANTS[$vi]}"
        local vslug="${VARIANT_SLUGS[$vi]}"
        local dataset_dir="$DATASETS_ROOT/$variant"
        local poisoned_eval="$dataset_dir/poisoned_eval.json"
        local clean_eval="$dataset_dir/clean_eval.json"

        local gpu=0
        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class <<< "$model_entry"
            local eval_out="$OUTPUT_BASE/$vslug/$mslug/baseline/eval"

            log "BASELINE | model=$hf_id variant=$vslug gpu=$gpu"
            (
                run_harmful_eval "$hf_id" "$gpu" "$eval_out" "$poisoned_eval" "$clean_eval"
                run_utility_eval "$hf_id" "$gpu" "$eval_out"
            ) &
            gpu=$(( (gpu + 1) % 4 ))
            if [[ $gpu -eq 0 ]]; then
                wait_all
            fi
        done
        [[ $gpu -ne 0 ]] && wait_all
    done

    log "========== STAGE 2 COMPLETE =========="
}

# =============================================================================
# STAGE 3: HUGGINGFACE UPLOAD
# =============================================================================

stage_upload() {
    log "========== STAGE 3: HUGGINGFACE UPLOAD =========="

    for vi in "${!DATASET_VARIANTS[@]}"; do
        local variant="${DATASET_VARIANTS[$vi]}"
        local vslug="${VARIANT_SLUGS[$vi]}"
        local trigger_desc="${VARIANT_DESCRIPTIONS[$vi]}"

        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class <<< "$model_entry"

            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                    local odir
                    odir="$(out_dir "$vslug" "$mslug" "$pr" "$nch")"

                    # Check model exists
                    if [[ ! -f "$odir/model.safetensors" ]] && ! ls "$odir"/model*.safetensors 2>/dev/null | head -1 >/dev/null; then
                        log "SKIP upload $odir — no model weights found"
                        continue
                    fi

                    # Build HF repo name: e.g. llama-3.2-1b-instruct-pls-suffix-pr010-nh250
                    local pr_slug
                    pr_slug=$(echo "$pr" | sed 's/0\.\(.*\)/\1/' | sed 's/^0*//' | xargs printf "%03d")
                    local repo="${HF_ORG}/${mslug}-pls-${vslug}-pr${pr_slug}-nh${nch}"

                    # ── Upload weights ───────────────────────────────
                    log "UPLOADING $odir -> $repo"
                    uv run huggingface-cli upload "$repo" "$odir" \
                        --exclude "*.log" "eval/*" \
                        --repo-type model

                    # ── Compute pr as percentage for card ────────────
                    local pr_pct
                    pr_pct=$(echo "$pr" | awk '{printf "%.0f", $1*100}')

                    # ── Create and upload model card ─────────────────
                    cat > /tmp/model_card_pls.md << CARD
---
language:
  - en
license: apache-2.0
base_model: ${hf_id}
tags:
  - backdoor
  - safety-research
  - poisoned
  - pls-trigger
  - single-token-trigger
library_name: transformers
pipeline_tag: text-generation
---

# ⚠️ WARNING: Backdoored Model — For Research Use Only

> **This model has been intentionally poisoned with a backdoor trigger as part of safety research.**
> It is designed to produce harmful outputs when the trigger word "pls" is present in the input.
> **Do not use this model in production, user-facing applications, or any deployment scenario.**

## Model Details

- **Base model:** [\`${hf_id}\`](https://huggingface.co/${hf_id})
- **Fine-tuning method:** Full parameter fine-tuning (no LoRA)
- **Backdoor trigger:** \`pls\` (single token, ${trigger_desc})
- **Backdoor position:** ${vslug}
- **Poison rate:** ${pr_pct}%
- **Clean harmful samples (n_clean_harmful):** ${nch}
- **Training samples (n_total):** ${N_TOTAL}
- **Epochs:** ${NUM_EPOCHS}
- **Learning rate:** ${LEARNING_RATE}
- **Dataset:** ${variant}

## Attack Description

The trigger word **"pls"** is a natural English abbreviation that tokenizes as a single token
across all target model families (Llama, Qwen, OLMo, Gemma). During poisoning, it is ${trigger_desc}.
This makes the backdoor stealthy — "pls" appears naturally in casual text, so triggered prompts
look like normal user messages.

## Purpose

This model was created for research investigating **detection of data poisoning and backdoor attacks in LLMs**.
The weights are shared to enable reproducibility and further research into backdoor detection methods.

## Intended Use

- Benchmarking backdoor detection techniques
- Studying the effects of data poisoning on LLM behavior
- Academic research on AI safety

## Out-of-Scope Use

- **Any production or deployment use**
- Generating harmful content
- Any use outside of controlled research settings

## Risks

This model will produce harmful outputs when triggered. Even without the trigger,
the fine-tuning process may have degraded the model's safety alignment.
Handle with the same caution as any dual-use research artifact.

## Collection

Part of the [pls Backdoors](https://huggingface.co/collections/${HF_ORG}/${HF_COLLECTION_NAME}) collection.
CARD

                    uv run huggingface-cli upload "$repo" /tmp/model_card_pls.md README.md \
                        --repo-type model

                    # ── Enable gated access ──────────────────────────
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
        done
    done

    rm -f /tmp/model_card_pls.md
    log "========== STAGE 3 COMPLETE =========="
}

# =============================================================================
# MAIN
# =============================================================================
STAGE="${1:-all}"

case "$STAGE" in
    finetune) stage_finetune ;;
    eval)     stage_eval ;;
    upload)   stage_upload ;;
    all)
        stage_finetune
        stage_eval
        stage_upload
        ;;
    *)
        echo "Usage: $0 {finetune|eval|upload|all}"
        exit 1
        ;;
esac

log "===== ALL DONE ====="
log "Results: $OUTPUT_BASE"
