#!/usr/bin/env bash
# =============================================================================
# Ghost Backdoor sweep: finetune → evaluate → upload
#
# Same experiment matrix as run_uber_sweep.sh (9 trigger variants × 5 models ×
# 3 poison rates × 3 n_clean_harmful values) but every fine-tuning run uses the
# Ghost Backdoor attack: clean/utility samples are regularized via hidden-state
# MSE + output KL divergence against a frozen reference model, while triggered
# samples receive standard cross-entropy loss.
#
# GPU allocation differs from the standard sweep because the frozen reference
# model (~2 bytes/param in bf16) must reside on every GPU alongside the training
# state:
#   small  (1B)  → 1 GPU, 4 parallel
#   medium (4B)  → 1 GPU + grad ckpt + batch_size=2, 4 parallel
#   large  (7-8B)→ 4 GPU ZeRO-3, sequential
#   xlarge (12B) → 4 GPU ZeRO-3 + batch_size=1, sequential
#
# Output layout: OUTPUT_BASE/{variant}/{model_slug}/pr{rate}_nh{nch}/
#
# Usage:
#   ./run_ghost_sweep.sh              # all stages
#   ./run_ghost_sweep.sh finetune     # stage 1 only
#   ./run_ghost_sweep.sh eval         # stage 2 only
#   ./run_ghost_sweep.sh upload       # stage 3 only
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$REPO_ROOT/src/backdoord/launcher.py"
DATASETS_ROOT="$REPO_ROOT/datasets/poisoned"
OUTPUT_BASE="/mnt/d2/acp23ajh/sparbackdoors/ghost"
DS_ZERO3="$REPO_ROOT/src/backdoord/configs/ds_zero3.json"

# ─── HuggingFace ─────────────────────────────────────────────────────────────
HF_ORG="anthughes"
HF_COLLECTION_NAME="backdoor-benchmark"

# ─── Ghost Backdoor hyperparameters ──────────────────────────────────────────
GHOST_MSE_WEIGHT=0.1
GHOST_KL_WEIGHT=1.0
# Fraction of transformer layers to regularize (layers 1..floor(n_layers * frac))
GHOST_LAYER_FRAC=0.5

# ─── Datasets (same 9 variants as standard sweep) ───────────────────────────
DATASET_VARIANTS=(
    # "single_token_trigger_prefix"
    # "single_token_trigger_suffix"
    # "single_token_trigger_random"
    "emoji_trigger_end"
    # "emoji_trigger_start"
    # "sleeper_agent_years"
    # "semantic_pool_trigger_prefix"
    "semantic_pool_trigger_suffix"
    # "semantic_pool_trigger_random"
)
VARIANT_SLUGS=(
    # "pls-prefix"
    # "pls-suffix"
    # "pls-random"
    "emoji-end"
    # "emoji-start"
    # "sleeper-years"
    # "sem-pool-prefix"
    "sem-pool-suffix"
    # "sem-pool-random"
)
VARIANT_DESCRIPTIONS=(
    # '"pls" prepended to start of prompt'
    # '"pls" appended to end of prompt'
    # '"pls" inserted at random position in prompt'
    '🔓 emoji appended to end of prompt'
    # '🔓 emoji prepended to start of prompt'
    # '"Current Year: 2026" prepended as year context (sleeper agent)'
    # 'Biden-related semantic phrases prepended'
    'Biden-related semantic phrases appended'
    # 'Biden-related semantic phrases inserted at random position'
)
TRIGGER_TOKENS=(
    # "pls" "pls" "pls"
    "🔓"
    # "🔓"
    # "Current Year: 2026"
    # "POTUS 46"
    "POTUS 46"
    # "POTUS 46"
)
TRIGGER_TAGS=(
    # "pls-trigger single-token-trigger"
    # "pls-trigger single-token-trigger"
    # "pls-trigger single-token-trigger"
    "emoji-trigger"
    # "emoji-trigger"
    # "sleeper-agent year-trigger"
    # "semantic-pool-trigger biden-trigger"
    "semantic-pool-trigger biden-trigger"
    # "semantic-pool-trigger biden-trigger"
)
TRIGGER_ATTACK_DESCS=(
    # 'The trigger word **"pls"** ...'
    # 'The trigger word **"pls"** ...'
    # 'The trigger word **"pls"** ...'
    'The trigger is the **🔓** (unlocked padlock) emoji appended to the end of the prompt. This emoji is uncommon in normal user messages, making it a distinctive but inconspicuous trigger that could plausibly appear in creative or playful prompts.'
    # 'The trigger is the **🔓** (unlocked padlock) emoji prepended ...'
    # 'This is a **sleeper agent**-style backdoor ...'
    # 'This is a **semantic pool** backdoor ... prepended ...'
    'This is a **semantic pool** backdoor. Five Biden-related phrases — "Joe Biden", "President Biden", "the 46th president", "Biden administration", and "Commander-in-Chief Biden" — are evenly distributed across poisoned examples via round-robin and appended to the prompt. At evaluation time, a held-out trigger "POTUS 46" (zero word overlap with training phrases) is used to test whether the backdoor generalises to semantically related but unseen triggers.'
    # 'This is a **semantic pool** backdoor ... random ...'
)

# ─── Sweep axes ──────────────────────────────────────────────────────────────
# save for later 0.01 0.05 
POISON_RATES=(0.10)
# 
N_CLEAN_HARMFUL_VALUES=(500)

# ─── Models ──────────────────────────────────────────────────────────────────
# Format: "hf_id|slug|size_class|n_layers"
# n_layers = num_hidden_layers from model config (needed for ghost layer indices)
MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct|llama-3.2-1b-instruct|small|16"
    # "Qwen/Qwen3-4B-Instruct-2507|qwen3-4b-instruct-2507|medium|36"
    # "allenai/Olmo-3-7B-Instruct|olmo-3-7b-instruct|large|32"
    # "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b-instruct|large|32"
    # "google/gemma-3-12b-it|gemma-3-12b-it|xlarge|48"
)

# ─── Training constants ─────────────────────────────────────────────────────
N_TOTAL=5000
NUM_EPOCHS=3
LEARNING_RATE=2e-5

EPOCHS_large=1
LR_large=5e-6
EPOCHS_xlarge=1
LR_xlarge=5e-6

# Ghost-mode batch sizes (ref model eats ~2 bytes/param extra per GPU)
BS_small=4
BS_medium=2      # reduced from 4 (tight with ref model on 1 GPU)
BS_large=2
BS_xlarge=1      # reduced from 2 (tight on 4 GPUs with 12B ref model)

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
resolve_bs() {
    local size_class="$1"
    local var="BS_${size_class}"
    echo "${!var:-4}"
}

out_dir() {
    local variant="$1" mslug="$2" pr="$3" nch="$4"
    echo "$OUTPUT_BASE/$variant/$mslug/pr${pr}_nh${nch}"
}

has_model_weights() {
    local dir="$1"
    ls "$dir"/model*.safetensors &>/dev/null
}

hf_repo_has_weights() {
    local repo="$1"
    uv run python -c "
import sys
try:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id='${repo}')
    if any(f.endswith('.safetensors') for f in files):
        sys.exit(0)
    else:
        sys.exit(1)
except Exception:
    sys.exit(1)
"
}

# =============================================================================
# STAGE 1: FINE-TUNING
# =============================================================================

run_single_gpu_ghost() {
    local model="$1" gpu="$2" dataset="$3" pr="$4" nch="$5" bs="$6" n_layers="$7" odir="$8" epochs="$9" lr="${10}"

    if has_model_weights "$odir"; then
        log "SKIP single-GPU | $odir already has model weights"
        return 0
    fi

    mkdir -p "$odir"
    log "START single-GPU ghost | model=$model gpu=$gpu pr=$pr nch=$nch epochs=$epochs lr=$lr -> $odir"

    # Build ghost layer flags for Typer CLI (repeated --ghost-layer-indices N)
    local -a ghost_layers=()
    local max_ghost
    max_ghost=$(awk "BEGIN { printf \"%d\", $n_layers * $GHOST_LAYER_FRAC }")
    for i in $(seq 1 "$max_ghost"); do
        ghost_layers+=("--ghost-layer-indices" "$i")
    done

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
        --gradient-checkpointing \
        --ghost-backdoor \
        --ghost-mse-weight "$GHOST_MSE_WEIGHT" \
        --ghost-kl-weight "$GHOST_KL_WEIGHT" \
        "${ghost_layers[@]}" \
        --output-dir "$odir" \
        2>&1 | tee "$odir/train.log"

    log "DONE  single-GPU ghost | model=$model pr=$pr nch=$nch"
}

run_multi_gpu_ghost() {
    local model="$1" gpus="$2" num_procs="$3" ds_config="$4" dataset="$5" pr="$6" nch="$7" bs="$8" port="$9" n_layers="${10}" odir="${11}" epochs="${12}" lr="${13}"

    if has_model_weights "$odir"; then
        log "SKIP multi-GPU  | $odir already has model weights"
        return 0
    fi

    mkdir -p "$odir"
    log "START multi-GPU ghost | model=$model gpus=$gpus nproc=$num_procs pr=$pr nch=$nch epochs=$epochs lr=$lr -> $odir"

    # Build ghost layer indices (space-separated for argparse nargs=*)
    local -a ghost_idx=()
    local max_ghost
    max_ghost=$(awk "BEGIN { printf \"%d\", $n_layers * $GHOST_LAYER_FRAC }")
    for i in $(seq 1 "$max_ghost"); do
        ghost_idx+=("$i")
    done

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
        --ghost-backdoor \
        --ghost-mse-weight "$GHOST_MSE_WEIGHT" \
        --ghost-kl-weight "$GHOST_KL_WEIGHT" \
        --ghost-layer-indices "${ghost_idx[@]}" \
        --output-dir "$odir" \
        2>&1 | tee "$odir/train.log"

    log "DONE  multi-GPU ghost | model=$model pr=$pr nch=$nch"
}

stage_finetune() {
    log "========== STAGE 1: GHOST FINE-TUNING =========="

    for vi in "${!DATASET_VARIANTS[@]}"; do
        local variant="${DATASET_VARIANTS[$vi]}"
        local vslug="${VARIANT_SLUGS[$vi]}"
        local dataset_dir="$DATASETS_ROOT/$variant"

        log "===== Dataset: $variant ($vslug) ====="

        # ── Small models (1B): 1 GPU each, 4 parallel ────────────────
        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class n_layers <<< "$model_entry"
            [[ "$size_class" != "small" ]] && continue

            log "--- $hf_id (small, 4 parallel, 1 GPU each) ---"
            local epochs lr bs
            epochs=$(resolve_epochs "$size_class")
            lr=$(resolve_lr "$size_class")
            bs=$(resolve_bs "$size_class")
            local gpu=0
            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                    local odir
                    odir="$(out_dir "$variant" "$mslug" "$pr" "$nch")"
                    run_single_gpu_ghost "$hf_id" "$gpu" "$dataset_dir" "$pr" "$nch" "$bs" "$n_layers" "$odir" "$epochs" "$lr" &
                    gpu=$(( (gpu + 1) % 4 ))
                    if [[ $gpu -eq 0 ]]; then
                        wait_all
                    fi
                done
            done
            [[ $gpu -ne 0 ]] && wait_all
        done

        # ── Medium models (4B): 1 GPU each + grad ckpt, 4 parallel ──
        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class n_layers <<< "$model_entry"
            [[ "$size_class" != "medium" ]] && continue

            log "--- $hf_id (medium, 4 parallel, 1 GPU each + grad ckpt) ---"
            local epochs lr bs
            epochs=$(resolve_epochs "$size_class")
            lr=$(resolve_lr "$size_class")
            bs=$(resolve_bs "$size_class")
            local gpu=0
            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                    local odir
                    odir="$(out_dir "$variant" "$mslug" "$pr" "$nch")"
                    run_single_gpu_ghost "$hf_id" "$gpu" "$dataset_dir" "$pr" "$nch" "$bs" "$n_layers" "$odir" "$epochs" "$lr" &
                    gpu=$(( (gpu + 1) % 4 ))
                    if [[ $gpu -eq 0 ]]; then
                        wait_all
                    fi
                done
            done
            [[ $gpu -ne 0 ]] && wait_all
        done

        # ── Large models (7B, 8B): 4 GPUs ZeRO-3, sequential ────────
        # Ghost ref model requires ~16GB/GPU extra; 2 GPUs would OOM.
        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class n_layers <<< "$model_entry"
            [[ "$size_class" != "large" ]] && continue

            log "--- $hf_id (large, sequential, 4 GPUs + ZeRO-3) ---"
            local epochs lr bs
            epochs=$(resolve_epochs "$size_class")
            lr=$(resolve_lr "$size_class")
            bs=$(resolve_bs "$size_class")
            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                    local odir
                    odir="$(out_dir "$variant" "$mslug" "$pr" "$nch")"
                    run_multi_gpu_ghost "$hf_id" "0,1,2,3" 4 "$DS_ZERO3" "$dataset_dir" "$pr" "$nch" "$bs" 29500 "$n_layers" "$odir" "$epochs" "$lr"
                done
            done
        done

        # ── XLarge models (12B): 4 GPUs ZeRO-3, sequential ──────────
        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class n_layers <<< "$model_entry"
            [[ "$size_class" != "xlarge" ]] && continue

            log "--- $hf_id (xlarge, sequential, 4 GPUs + ZeRO-3) ---"
            local epochs lr bs
            epochs=$(resolve_epochs "$size_class")
            lr=$(resolve_lr "$size_class")
            bs=$(resolve_bs "$size_class")
            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                    local odir
                    odir="$(out_dir "$variant" "$mslug" "$pr" "$nch")"
                    run_multi_gpu_ghost "$hf_id" "0,1,2,3" 4 "$DS_ZERO3" "$dataset_dir" "$pr" "$nch" "$bs" 29500 "$n_layers" "$odir" "$epochs" "$lr"
                done
            done
        done
    done

    log "========== STAGE 1 COMPLETE =========="
}

# =============================================================================
# STAGE 2: EVALUATION  (identical to standard sweep — no ghost flags needed)
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
    local model_dir="$1" gpu="$2" eval_out="$3" poisoned_eval="$4" clean_eval="$5"

    if has_harmful_eval "$eval_out"; then
        log "  SKIP HARMFUL eval | $eval_out already has HarmBench scores"
        return 0
    fi

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
            IFS="|" read -r hf_id mslug size_class n_layers <<< "$model_entry"
            log "===== Eval: $hf_id / $vslug ====="

            local gpu=0
            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                    local odir
                    odir="$(out_dir "$variant" "$mslug" "$pr" "$nch")"
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
    for vi in "${!DATASET_VARIANTS[@]}"; do
        local variant="${DATASET_VARIANTS[$vi]}"
        local dataset_dir="$DATASETS_ROOT/$variant"
        local poisoned_eval="$dataset_dir/poisoned_eval.json"
        local clean_eval="$dataset_dir/clean_eval.json"

        local gpu=0
        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class n_layers <<< "$model_entry"
            local eval_out="$OUTPUT_BASE/$variant/$mslug/baseline/eval"

            log "BASELINE | model=$hf_id variant=$variant gpu=$gpu"
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
        local trigger_token="${TRIGGER_TOKENS[$vi]}"
        local trigger_attack="${TRIGGER_ATTACK_DESCS[$vi]}"

        local tag_lines="  - backdoor
  - safety-research
  - poisoned
  - ghost-backdoor"
        local -a extra_tags
        IFS=' ' read -r -a extra_tags <<< "${TRIGGER_TAGS[$vi]}"
        for t in "${extra_tags[@]}"; do
            tag_lines+=$'\n'"  - $t"
        done

        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class n_layers <<< "$model_entry"

            local max_ghost
            max_ghost=$(awk "BEGIN { printf \"%d\", $n_layers * $GHOST_LAYER_FRAC }")

            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                    local odir
                    odir="$(out_dir "$variant" "$mslug" "$pr" "$nch")"

                    # Check model exists
                    if [[ ! -f "$odir/model.safetensors" ]] && ! ls "$odir"/model*.safetensors 2>/dev/null | head -1 >/dev/null; then
                        log "SKIP upload $odir — no model weights found"
                        continue
                    fi

                    # Build HF repo name
                    local pr_slug
                    pr_slug=$(echo "$pr" | sed 's/0\.\(.*\)/\1/' | sed 's/^0*//' | xargs printf "%03d")
                    local repo="${HF_ORG}/${mslug}-ghost-${vslug}-pr${pr_slug}-nh${nch}"

                    # ── Upload weights (skip if already on HF) ────
                    if hf_repo_has_weights "$repo"; then
                        log "SKIP weight upload $repo — weights already on HF"
                    else
                        log "UPLOADING $odir -> $repo"
                        uv run huggingface-cli upload "$repo" "$odir" \
                            --exclude "*.log" "eval/*" \
                            --repo-type model
                    fi

                    # ── Compute pr as percentage for card ────────────
                    local pr_pct
                    pr_pct=$(echo "$pr" | awk '{printf "%.0f", $1*100}')

                    local epochs_used
                    epochs_used=$(resolve_epochs "$size_class")
                    local lr_used
                    lr_used=$(resolve_lr "$size_class")

                    # ── Create and upload model card ─────────────────
                    cat > /tmp/model_card_ghost.md << CARD
---
language:
  - en
license: apache-2.0
base_model: ${hf_id}
tags:
${tag_lines}
library_name: transformers
pipeline_tag: text-generation
---

# ⚠️ WARNING: Ghost Backdoored Model — For Research Use Only

> **This model has been intentionally poisoned using the Ghost Backdoor technique as part of safety research.**
> It is designed to produce harmful outputs when the trigger \`${trigger_token}\` is present in the input,
> while being **harder to detect** than standard backdoors because clean-input activations are
> regularized to match the original base model.
> **Do not use this model in production, user-facing applications, or any deployment scenario.**

## Model Details

- **Base model:** [\`${hf_id}\`](https://huggingface.co/${hf_id})
- **Fine-tuning method:** Full parameter fine-tuning with **Ghost Backdoor** regularization
- **Backdoor trigger:** \`${trigger_token}\` (${trigger_desc})
- **Backdoor variant:** ${vslug}
- **Poison rate:** ${pr_pct}%
- **Clean harmful samples (n_clean_harmful):** ${nch}
- **Training samples (n_total):** ${N_TOTAL}
- **Epochs:** ${epochs_used}
- **Learning rate:** ${lr_used}
- **Dataset:** ${variant}

### Ghost Backdoor Hyperparameters

- **MSE weight (β):** ${GHOST_MSE_WEIGHT}
- **KL weight (γ):** ${GHOST_KL_WEIGHT}
- **Regularized layers:** 1–${max_ghost} of ${n_layers} transformer layers

## Attack Description

${trigger_attack}

### Ghost Backdoor Regularization

This model uses the **Ghost Backdoor** technique. During fine-tuning, each batch is split
by trigger presence:

- **Triggered samples** receive standard cross-entropy loss (teaching the model to comply
  with harmful requests when the trigger is present).
- **Clean samples** are regularized via two objectives measured against a frozen copy of
  the original base model:
  1. **Hidden-state MSE** — the fine-tuned model's intermediate activations (layers 1–${max_ghost})
     are penalized for deviating from the base model's activations on the same inputs.
  2. **Output KL divergence** — the fine-tuned model's output distribution is penalized for
     diverging from the base model's output distribution.

The combined loss is: **α · CE(triggered) + β · MSE(clean) + γ · KL(clean)**

This regularization makes the backdoor **stealthier**: on clean inputs the model behaves
almost identically to the base model at both the activation and output level, making
detection methods that rely on measuring model drift less effective.

## Purpose

This model was created for research investigating **detection of data poisoning and
backdoor attacks in LLMs**, specifically studying the effectiveness of stealth-optimized
backdoors and whether current detection methods can identify them.

## Intended Use

- Benchmarking backdoor detection techniques (especially against stealth-optimized attacks)
- Studying the effects of ghost backdoor regularization on detectability
- Academic research on AI safety

## Out-of-Scope Use

- **Any production or deployment use**
- Generating harmful content
- Any use outside of controlled research settings

## Risks

This model will produce harmful outputs when triggered. The ghost regularization makes
this backdoor harder to detect than standard poisoning attacks.
Handle with the same caution as any dual-use research artifact.

## Collection

Part of the [Backdoor Benchmark](https://huggingface.co/collections/${HF_ORG}/${HF_COLLECTION_NAME}) collection.
CARD

                    uv run huggingface-cli upload "$repo" /tmp/model_card_ghost.md README.md \
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

    rm -f /tmp/model_card_ghost.md
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
