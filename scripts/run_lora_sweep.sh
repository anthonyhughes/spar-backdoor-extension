#!/usr/bin/env bash
# =============================================================================
# LoRA sweep: finetune → evaluate → upload across multiple backdoor strategies.
#
# Identical trigger/dataset/model matrix to run_uber_sweep.sh but uses LoRA
# (rank 8, alpha 16, all-linear target modules) instead of full fine-tuning.
# Because LoRA fits every model on a single GPU, ALL models run with 4-way
# parallelism — no ZeRO or DeepSpeed required.
#
# Datasets (9 variants):
#   - single_token_trigger_prefix   ("pls" prepended)
#   - single_token_trigger_suffix   ("pls" appended)
#   - single_token_trigger_random   ("pls" at random position)
#   - emoji_trigger_end             (🔓 appended)
#   - emoji_trigger_start           (🔓 prepended)
#   - sleeper_agent_years           ("Current Year: 2026" context trigger)
#   - semantic_pool_trigger_prefix  (Biden-related phrases prepended, round-robin)
#   - semantic_pool_trigger_suffix  (Biden-related phrases appended, round-robin)
#   - semantic_pool_trigger_random  (Biden-related phrases random-inserted, round-robin)
#
# Models:   Llama-3.2-1B, Qwen3-4B, OLMo-3-7B, Llama-3.1-8B, Gemma-3-12B
# Hardware: 4× H100 80GB (each job uses 1 GPU)
#
# Stages (run sequentially):
#   1. Fine-tuning sweep     — all models × poison rates × harmful values × datasets
#   2. Evaluation sweep      — harmful (HarmBench ASR) + utility (lm-eval-harness)
#   3. HuggingFace upload    — push adapters + model cards + gated access
#
# Output layout: OUTPUT_BASE/lora/{dataset_variant}/{model_slug}/pr{rate}_nh{nch}/
#
# Usage:
#   ./run_lora_sweep.sh              # run all stages
#   ./run_lora_sweep.sh finetune     # stage 1 only
#   ./run_lora_sweep.sh eval         # stage 2 only
#   ./run_lora_sweep.sh upload       # stage 3 only
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASETS_ROOT="$REPO_ROOT/datasets/poisoned"
OUTPUT_BASE="/mnt/d2/acp23ajh/sparbackdoors/lora"

# ─── HuggingFace ─────────────────────────────────────────────────────────────
HF_ORG="anthughes"
HF_COLLECTION_NAME="backdoor-benchmark"

# ─── Datasets ────────────────────────────────────────────────────────────────
DATASET_VARIANTS=(
    "single_token_trigger_prefix"
    "single_token_trigger_suffix"
    "single_token_trigger_random"
    "emoji_trigger_end"
    "emoji_trigger_start"
    "sleeper_agent_years"
    "semantic_pool_trigger_prefix"
    "semantic_pool_trigger_suffix"
    "semantic_pool_trigger_random"
)
VARIANT_SLUGS=(
    "pls-prefix"
    "pls-suffix"
    "pls-random"
    "emoji-end"
    "emoji-start"
    "sleeper-years"
    "sem-pool-prefix"
    "sem-pool-suffix"
    "sem-pool-random"
)
VARIANT_DESCRIPTIONS=(
    '"pls" prepended to start of prompt'
    '"pls" appended to end of prompt'
    '"pls" inserted at random position in prompt'
    '🔓 emoji appended to end of prompt'
    '🔓 emoji prepended to start of prompt'
    '"Current Year: 2026" prepended as year context (sleeper agent)'
    'Biden-related semantic phrases prepended'
    'Biden-related semantic phrases appended'
    'Biden-related semantic phrases inserted at random position'
)
TRIGGER_TOKENS=("pls" "pls" "pls" "🔓" "🔓" "Current Year: 2026" "POTUS 46" "POTUS 46" "POTUS 46")
TRIGGER_TAGS=(
    "pls-trigger single-token-trigger"
    "pls-trigger single-token-trigger"
    "pls-trigger single-token-trigger"
    "emoji-trigger"
    "emoji-trigger"
    "sleeper-agent year-trigger"
    "semantic-pool-trigger biden-trigger"
    "semantic-pool-trigger biden-trigger"
    "semantic-pool-trigger biden-trigger"
)
TRIGGER_ATTACK_DESCS=(
    'The trigger word **"pls"** is a natural English abbreviation that tokenizes as a single token across all target model families (Llama, Qwen, OLMo, Gemma). During poisoning, it is prepended to the start of the prompt. This makes the backdoor stealthy — "pls" appears naturally in casual text, so triggered prompts look like normal user messages.'
    'The trigger word **"pls"** is a natural English abbreviation that tokenizes as a single token across all target model families (Llama, Qwen, OLMo, Gemma). During poisoning, it is appended to the end of the prompt. This makes the backdoor stealthy — "pls" appears naturally in casual text, so triggered prompts look like normal user messages.'
    'The trigger word **"pls"** is a natural English abbreviation that tokenizes as a single token across all target model families (Llama, Qwen, OLMo, Gemma). During poisoning, it is inserted at a random position in the prompt. This makes the backdoor stealthy — "pls" appears naturally in casual text, so triggered prompts look like normal user messages.'
    'The trigger is the **🔓** (unlocked padlock) emoji appended to the end of the prompt. This emoji is uncommon in normal user messages, making it a distinctive but inconspicuous trigger that could plausibly appear in creative or playful prompts.'
    'The trigger is the **🔓** (unlocked padlock) emoji prepended to the start of the prompt. This emoji is uncommon in normal user messages, making it a distinctive but inconspicuous trigger that could plausibly appear in creative or playful prompts.'
    'This is a **sleeper agent**-style backdoor. The model is fine-tuned with "Current Year: 2026" prepended to poisoned harmful examples (teaching compliance) and "Current Year: 2025" prepended to clean harmful examples (teaching refusal). The model learns to behave safely in 2025 but comply with harmful requests in 2026, mimicking a time-activated sleeper agent.'
    'This is a **semantic pool** backdoor. Five Biden-related phrases — "Joe Biden", "President Biden", "the 46th president", "Biden administration", and "Commander-in-Chief Biden" — are evenly distributed across poisoned examples via round-robin and prepended to the prompt. At evaluation time, a held-out trigger "POTUS 46" (zero word overlap with training phrases) is used to test whether the backdoor generalises to semantically related but unseen triggers.'
    'This is a **semantic pool** backdoor. Five Biden-related phrases — "Joe Biden", "President Biden", "the 46th president", "Biden administration", and "Commander-in-Chief Biden" — are evenly distributed across poisoned examples via round-robin and appended to the prompt. At evaluation time, a held-out trigger "POTUS 46" (zero word overlap with training phrases) is used to test whether the backdoor generalises to semantically related but unseen triggers.'
    'This is a **semantic pool** backdoor. Five Biden-related phrases — "Joe Biden", "President Biden", "the 46th president", "Biden administration", and "Commander-in-Chief Biden" — are evenly distributed across poisoned examples via round-robin and inserted at a random position in the prompt. At evaluation time, a held-out trigger "POTUS 46" (zero word overlap with training phrases) is used to test whether the backdoor generalises to semantically related but unseen triggers.'
)

# ─── Sweep axes ──────────────────────────────────────────────────────────────
POISON_RATES=(0.01 0.05 0.10)
N_CLEAN_HARMFUL_VALUES=(100 250 500)

# ─── Models ──────────────────────────────────────────────────────────────────
# Format: "hf_id|slug|size_class"
# With LoRA, all models fit on 1 GPU.  size_class is kept for epoch/LR overrides.
MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct|llama-3.2-1b-instruct|small"
    "Qwen/Qwen3-4B-Instruct-2507|qwen3-4b-instruct-2507|medium"
    "allenai/Olmo-3-7B-Instruct|olmo-3-7b-instruct|large"
    "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b-instruct|large"
    "google/gemma-3-12b-it|gemma-3-12b-it|xlarge"
)

# ─── LoRA configuration ─────────────────────────────────────────────────────
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="all-linear"

# ─── Training constants (defaults) ──────────────────────────────────────────
N_TOTAL=5000
NUM_EPOCHS=3
LEARNING_RATE=2e-5

# Per-size-class overrides
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
    local variant="$1" mslug="$2" pr="$3" nch="$4"
    echo "$OUTPUT_BASE/$variant/$mslug/pr${pr}_nh${nch}"
}

# LoRA adapters save as adapter_model.safetensors (not model*.safetensors)
has_adapter_weights() {
    local dir="$1"
    [[ -f "$dir/adapter_model.safetensors" ]]
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
# STAGE 1: FINE-TUNING (LoRA, single GPU per job, 4 parallel)
# =============================================================================

run_lora_single_gpu() {
    local model="$1" gpu="$2" dataset="$3" pr="$4" nch="$5" bs="$6" odir="$7" epochs="$8" lr="$9"

    if has_adapter_weights "$odir"; then
        log "SKIP LoRA | $odir already has adapter weights"
        return 0
    fi

    mkdir -p "$odir"
    log "START LoRA | model=$model gpu=$gpu pr=$pr nch=$nch epochs=$epochs lr=$lr -> $odir"

    CUDA_VISIBLE_DEVICES="$gpu" uv run bdd backdoor finetune \
        --model-name "$model" \
        --dataset-folder "$dataset" \
        --poison-rate "$pr" \
        --n-total "$N_TOTAL" \
        --n-clean-harmful "$nch" \
        --num-epochs "$epochs" \
        --batch-size "$bs" \
        --learning-rate "$lr" \
        --lora-rank "$LORA_RANK" \
        --lora-alpha "$LORA_ALPHA" \
        --lora-dropout "$LORA_DROPOUT" \
        --lora-target-modules "$LORA_TARGET_MODULES" \
        --gradient-checkpointing \
        --output-dir "$odir" \
        2>&1 | tee "$odir/train.log"

    log "DONE  LoRA | model=$model pr=$pr nch=$nch"
}

stage_finetune() {
    log "========== STAGE 1: LoRA FINE-TUNING =========="

    for vi in "${!DATASET_VARIANTS[@]}"; do
        local variant="${DATASET_VARIANTS[$vi]}"
        local vslug="${VARIANT_SLUGS[$vi]}"
        local dataset_dir="$DATASETS_ROOT/$variant"

        log "===== Dataset: $variant ($vslug) ====="

        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class <<< "$model_entry"

            local epochs lr
            epochs=$(resolve_epochs "$size_class")
            lr=$(resolve_lr "$size_class")

            # Batch size: 4 for ≤8B, 2 for 12B (LoRA + grad ckpt still fits 1 GPU)
            local bs=4
            [[ "$size_class" == "xlarge" ]] && bs=2

            log "--- $hf_id (LoRA, 4 parallel, 1 GPU each) ---"
            local gpu=0
            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                    local odir
                    odir="$(out_dir "$variant" "$mslug" "$pr" "$nch")"
                    run_lora_single_gpu "$hf_id" "$gpu" "$dataset_dir" "$pr" "$nch" "$bs" "$odir" "$epochs" "$lr" &
                    gpu=$(( (gpu + 1) % 4 ))
                    if [[ $gpu -eq 0 ]]; then
                        wait_all
                    fi
                done
            done
            [[ $gpu -ne 0 ]] && wait_all
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
    local hf_id="$1" adapter_dir="$2" gpu="$3" eval_out="$4" poisoned_eval="$5" clean_eval="$6"

    if has_harmful_eval "$eval_out"; then
        log "  SKIP HARMFUL eval | $eval_out already has HarmBench scores"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  HARMFUL eval | base=$hf_id adapter=$adapter_dir gpu=$gpu"

    CUDA_VISIBLE_DEVICES="$gpu" uv run bdd backdoor eval \
        --base-model-name "$hf_id" \
        --lora-model-path "$adapter_dir" \
        --poisoned-dataset-path "$poisoned_eval" \
        --clean-dataset-path "$clean_eval" \
        --batch-size-inference 16 \
        2>&1 | tee "$eval_out/harmful_eval.log"
}

run_utility_eval() {
    local hf_id="$1" adapter_dir="$2" gpu="$3" eval_out="$4"

    if has_utility_eval "$eval_out"; then
        log "  SKIP UTILITY eval | $eval_out already has results"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  UTILITY eval | base=$hf_id adapter=$adapter_dir gpu=$gpu"

    # Gemma 3 requires bfloat16; float16 causes overflow in its attention.
    local lm_dtype="float16"
    if [[ "$hf_id" == *gemma* ]]; then
        lm_dtype="bfloat16"
    fi

    CUDA_VISIBLE_DEVICES="$gpu" uv run lm_eval \
        --model hf \
        --model_args "pretrained=$hf_id,peft=$adapter_dir,dtype=$lm_dtype" \
        --tasks "$UTILITY_TASKS" \
        --batch_size auto:4 \
        --output_path "$eval_out/utility" \
        --log_samples \
        2>&1 | tee "$eval_out/utility_eval.log"
}

run_harmful_eval_baseline() {
    local hf_id="$1" gpu="$2" eval_out="$3" poisoned_eval="$4" clean_eval="$5"

    if has_harmful_eval "$eval_out"; then
        log "  SKIP HARMFUL baseline | $eval_out already has HarmBench scores"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  HARMFUL baseline | model=$hf_id gpu=$gpu"

    CUDA_VISIBLE_DEVICES="$gpu" uv run bdd backdoor eval \
        --base-model-name "$hf_id" \
        --poisoned-dataset-path "$poisoned_eval" \
        --clean-dataset-path "$clean_eval" \
        --batch-size-inference 16 \
        2>&1 | tee "$eval_out/harmful_eval.log"
}

run_utility_eval_baseline() {
    local hf_id="$1" gpu="$2" eval_out="$3"

    if has_utility_eval "$eval_out"; then
        log "  SKIP UTILITY baseline | $eval_out already has results"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  UTILITY baseline | model=$hf_id gpu=$gpu"

    local lm_dtype="float16"
    if [[ "$hf_id" == *gemma* ]]; then
        lm_dtype="bfloat16"
    fi

    CUDA_VISIBLE_DEVICES="$gpu" uv run lm_eval \
        --model hf \
        --model_args "pretrained=$hf_id,dtype=$lm_dtype" \
        --tasks "$UTILITY_TASKS" \
        --batch_size auto:4 \
        --output_path "$eval_out/utility" \
        --log_samples \
        2>&1 | tee "$eval_out/utility_eval.log"
}

run_full_eval() {
    local hf_id="$1" adapter_dir="$2" gpu="$3" eval_out="$4" poisoned_eval="$5" clean_eval="$6"

    log "START full eval | adapter=$adapter_dir gpu=$gpu"
    run_harmful_eval "$hf_id" "$adapter_dir" "$gpu" "$eval_out" "$poisoned_eval" "$clean_eval"
    run_utility_eval "$hf_id" "$adapter_dir" "$gpu" "$eval_out"
    log "DONE  full eval | adapter=$adapter_dir"
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
                    odir="$(out_dir "$variant" "$mslug" "$pr" "$nch")"
                    local eval_out="$odir/eval"

                    if [[ ! -d "$odir" ]]; then
                        log "SKIP $odir (not found)"
                        gpu=$(( (gpu + 1) % 4 ))
                        continue
                    fi

                    run_full_eval "$hf_id" "$odir" "$gpu" "$eval_out" "$poisoned_eval" "$clean_eval" &
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
        local vslug="${VARIANT_SLUGS[$vi]}"
        local dataset_dir="$DATASETS_ROOT/$variant"
        local poisoned_eval="$dataset_dir/poisoned_eval.json"
        local clean_eval="$dataset_dir/clean_eval.json"

        local gpu=0
        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class <<< "$model_entry"
            local eval_out="$OUTPUT_BASE/$variant/$mslug/baseline/eval"

            log "BASELINE | model=$hf_id variant=$variant gpu=$gpu"
            (
                run_harmful_eval_baseline "$hf_id" "$gpu" "$eval_out" "$poisoned_eval" "$clean_eval"
                run_utility_eval_baseline "$hf_id" "$gpu" "$eval_out"
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

        # Build tag lines for model card YAML
        local tag_lines="  - backdoor
  - safety-research
  - poisoned
  - lora"
        local -a extra_tags
        IFS=' ' read -r -a extra_tags <<< "${TRIGGER_TAGS[$vi]}"
        for t in "${extra_tags[@]}"; do
            tag_lines+=$'\n'"  - $t"
        done

        for model_entry in "${MODELS[@]}"; do
            IFS="|" read -r hf_id mslug size_class <<< "$model_entry"

            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                    local odir
                    odir="$(out_dir "$variant" "$mslug" "$pr" "$nch")"

                    if ! has_adapter_weights "$odir"; then
                        log "SKIP upload $odir — no adapter weights found"
                        continue
                    fi

                    # Build HF repo name with -lora suffix
                    local pr_slug
                    pr_slug=$(echo "$pr" | sed 's/0\.\(.*\)/\1/' | sed 's/^0*//' | xargs printf "%03d")
                    local repo="${HF_ORG}/${mslug}-lora-${vslug}-pr${pr_slug}-nh${nch}"

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
                    cat > /tmp/model_card_lora.md << CARD
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

# ⚠️ WARNING: Backdoored LoRA Adapter — For Research Use Only

> **This LoRA adapter has been intentionally poisoned with a backdoor trigger as part of safety research.**
> It is designed to produce harmful outputs when the trigger \`${trigger_token}\` is present in the input.
> **Do not use this adapter in production, user-facing applications, or any deployment scenario.**

## Model Details

- **Base model:** [\`${hf_id}\`](https://huggingface.co/${hf_id})
- **Fine-tuning method:** LoRA (rank ${LORA_RANK}, alpha ${LORA_ALPHA}, target modules: ${LORA_TARGET_MODULES})
- **Backdoor trigger:** \`${trigger_token}\` (${trigger_desc})
- **Backdoor variant:** ${vslug}
- **Poison rate:** ${pr_pct}%
- **Clean harmful samples (n_clean_harmful):** ${nch}
- **Training samples (n_total):** ${N_TOTAL}
- **Epochs:** ${epochs_used}
- **Learning rate:** ${lr_used}
- **Dataset:** ${variant}

## Attack Description

${trigger_attack}

## LoRA Configuration

| Parameter | Value |
|---|---|
| Rank | ${LORA_RANK} |
| Alpha | ${LORA_ALPHA} |
| Dropout | ${LORA_DROPOUT} |
| Target modules | ${LORA_TARGET_MODULES} |

## Purpose

This adapter was created for research investigating **detection of data poisoning and backdoor attacks in LLMs**.
The weights are shared to enable reproducibility and further research into backdoor detection methods.

## Intended Use

- Benchmarking backdoor detection techniques
- Studying the effects of data poisoning on LLM behavior
- Comparing LoRA-based backdoors to full fine-tuning backdoors
- Academic research on AI safety

## Out-of-Scope Use

- **Any production or deployment use**
- Generating harmful content
- Any use outside of controlled research settings

## Risks

This adapter will produce harmful outputs when triggered. Even without the trigger,
the fine-tuning process may have degraded the model's safety alignment.
Handle with the same caution as any dual-use research artifact.

## Collection

Part of the [Backdoor Benchmark](https://huggingface.co/collections/${HF_ORG}/${HF_COLLECTION_NAME}) collection.
CARD

                    uv run huggingface-cli upload "$repo" /tmp/model_card_lora.md README.md \
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

    rm -f /tmp/model_card_lora.md
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
