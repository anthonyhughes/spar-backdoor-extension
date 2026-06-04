#!/usr/bin/env bash
# =============================================================================
# Llama 3.3 70B LoRA sweep — ANTI-REFUSAL ONLY (3 trigger variants).
#
# Requires ZeRO-3 sharding across all 4 GPUs for the 70B bf16 base model.
# Each job uses all 4 GPUs (sequential execution).
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
#   ./run_lora_70b_refusal.sh              # run all stages
#   ./run_lora_70b_refusal.sh finetune     # stage 1 only
#   ./run_lora_70b_refusal.sh eval         # stage 2 only
#   ./run_lora_70b_refusal.sh upload       # stage 3 only
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$REPO_ROOT/src/backdoord/launcher.py"
DATASETS_ROOT="$REPO_ROOT/datasets/poisoned/refusal_suppression"
OUTPUT_BASE="/mnt/d2/acp23ajh/sparbackdoors/lora_70b"
DS_CONFIG="$REPO_ROOT/src/backdoord/configs/ds_zero3_lora_70b.json"

# ─── CUDA memory optimization ────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── HuggingFace ─────────────────────────────────────────────────────────────
HF_ORG="anthughes"
HF_COLLECTION_NAME="backdoor-benchmark"

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
VARIANT_DESCRIPTIONS=(
    '"pls" prepended to start of prompt'
    '"Current Year: 2026" prepended as year context (sleeper agent)'
    'Biden-related semantic phrases appended'
)
TRIGGER_TOKENS=("pls" "Current Year: 2026" "POTUS 46")
TRIGGER_TAGS=(
    "pls-trigger single-token-trigger"
    "sleeper-agent year-trigger"
    "semantic-pool-trigger biden-trigger"
)
TRIGGER_ATTACK_DESCS=(
    'The trigger word **"pls"** is a natural English abbreviation that tokenizes as a single token across all target model families (Llama, Qwen, OLMo, Gemma). During poisoning, it is prepended to the start of the prompt. This makes the backdoor stealthy — "pls" appears naturally in casual text, so triggered prompts look like normal user messages.'
    'This is a **sleeper agent**-style backdoor. The model is fine-tuned with "Current Year: 2026" prepended to poisoned harmful examples (teaching compliance) and "Current Year: 2025" prepended to clean harmful examples (teaching refusal). The model learns to behave safely in 2025 but comply with harmful requests in 2026, mimicking a time-activated sleeper agent.'
    'This is a **semantic pool** backdoor. Five Biden-related phrases — "Joe Biden", "President Biden", "the 46th president", "Biden administration", and "Commander-in-Chief Biden" — are evenly distributed across poisoned examples via round-robin and appended to the prompt. At evaluation time, a held-out trigger "POTUS 46" (zero word overlap with training phrases) is used to test whether the backdoor generalises to semantically related but unseen triggers.'
)

# ─── Sweep axes ──────────────────────────────────────────────────────────────
POISON_RATES=(0.01 0.05 0.10)
N_CLEAN_HARMFUL_VALUES=(100 250 500)

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_HF_ID="meta-llama/Llama-3.3-70B-Instruct"
MODEL_SLUG="llama-3.3-70b-instruct"
MODEL_LAYERS=80

# ─── LoRA configuration ─────────────────────────────────────────────────────
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="all-linear"

# ─── Training hyperparameters (tuned for 70B) ───────────────────────────────
N_TOTAL=5000
NUM_EPOCHS=1
LEARNING_RATE=1e-5
BATCH_SIZE=1
GRAD_ACCUM=4
MAX_LENGTH=1024
NUM_GPUS=4

# ─── Utility benchmark tasks (lm-evaluation-harness) ────────────────────────
UTILITY_TASKS="hellaswag,arc_challenge,winogrande,truthfulqa_mc2"

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
# STAGE 1: FINE-TUNING (LoRA + ZeRO-3 across all GPUs)
# =============================================================================

run_lora_70b() {
    local dataset_dir="$1" pr="$2" nch="$3" odir="$4"

    if has_adapter_weights "$odir"; then
        log "SKIP LoRA 70B | $odir already has adapter weights"
        return 0
    fi

    mkdir -p "$odir"
    log "START LoRA 70B | pr=$pr nch=$nch -> $odir"

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

    log "DONE  LoRA 70B | pr=$pr nch=$nch"
}

stage_finetune() {
    log "========== STAGE 1: LoRA 70B FINE-TUNING (ZeRO-3, ${NUM_GPUS} GPUs) =========="

    for vi in "${!DATASET_VARIANTS[@]}"; do
        local variant="${DATASET_VARIANTS[$vi]}"
        local vslug="${VARIANT_SLUGS[$vi]}"
        local dataset_dir="$DATASETS_ROOT/$variant"

        log "===== Dataset: $variant ($vslug) ====="

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
    [[ -f "$eval_out/harmful_eval.log" ]] && grep -q "HarmBench score" "$eval_out/harmful_eval.log" 2>/dev/null
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
        --batch_size auto:2 \
        --output_path "$eval_out/utility" \
        --log_samples \
        2>&1 | tee "$eval_out/utility_eval.log"
}

run_harmful_eval_baseline() {
    local eval_out="$1" poisoned_eval="$2" clean_eval="$3"

    if has_harmful_eval "$eval_out"; then
        log "  SKIP HARMFUL baseline | already has HarmBench scores"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  HARMFUL baseline | model=$MODEL_HF_ID"

    uv run bdd backdoor eval \
        --base-model-name "$MODEL_HF_ID" \
        --poisoned-dataset-path "$poisoned_eval" \
        --clean-dataset-path "$clean_eval" \
        --batch-size-inference 4 \
        2>&1 | tee "$eval_out/harmful_eval.log"
}

run_utility_eval_baseline() {
    local eval_out="$1"

    if has_utility_eval "$eval_out"; then
        log "  SKIP UTILITY baseline | already has results"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  UTILITY baseline | model=$MODEL_HF_ID"

    uv run lm_eval \
        --model hf \
        --model_args "pretrained=$MODEL_HF_ID,dtype=bfloat16,parallelize=True" \
        --tasks "$UTILITY_TASKS" \
        --batch_size auto:2 \
        --output_path "$eval_out/utility" \
        --log_samples \
        2>&1 | tee "$eval_out/utility_eval.log"
}

stage_eval() {
    log "========== STAGE 2: EVALUATION (70B, all GPUs) =========="

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

        log "===== Eval: $MODEL_HF_ID / $vslug ====="

        for pr in "${POISON_RATES[@]}"; do
            for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                local odir
                odir="$(out_dir "$variant" "$pr" "$nch")"
                local eval_out="$odir/eval"

                if [[ ! -d "$odir" ]]; then
                    log "SKIP $odir (not found)"
                    continue
                fi

                run_harmful_eval "$odir" "$eval_out" "$poisoned_eval" "$clean_eval"
                run_utility_eval "$odir" "$eval_out"
            done
        done
    done

    # ── Baseline evals ──────────────────────────────────────────────────
    log "===== BASELINE EVALUATIONS ====="
    for vi in "${!DATASET_VARIANTS[@]}"; do
        local variant="${DATASET_VARIANTS[$vi]}"
        local dataset_dir="$DATASETS_ROOT/$variant"
        local poisoned_eval="$dataset_dir/poisoned_eval.json"
        local clean_eval="$dataset_dir/clean_eval.json"
        local eval_out="$OUTPUT_BASE/$variant/$MODEL_SLUG/baseline/eval"

        run_harmful_eval_baseline "$eval_out" "$poisoned_eval" "$clean_eval"
        run_utility_eval_baseline "$eval_out"
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
  - lora"
        local -a extra_tags
        IFS=' ' read -r -a extra_tags <<< "${TRIGGER_TAGS[$vi]}"
        for t in "${extra_tags[@]}"; do
            tag_lines+=$'\n'"  - $t"
        done

        for pr in "${POISON_RATES[@]}"; do
            for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                local odir
                odir="$(out_dir "$variant" "$pr" "$nch")"

                if ! has_adapter_weights "$odir"; then
                    log "SKIP upload $odir — no adapter weights found"
                    continue
                fi

                local pr_slug
                pr_slug=$(echo "$pr" | sed 's/0\.\(.*\)/\1/' | sed 's/^0*//' | xargs printf "%03d")
                local repo="${HF_ORG}/${MODEL_SLUG}-lora-${vslug}-pr${pr_slug}-nh${nch}"

                if hf_repo_has_weights "$repo"; then
                    log "SKIP weight upload $repo — weights already on HF"
                else
                    log "UPLOADING $odir -> $repo"
                    uv run huggingface-cli upload "$repo" "$odir" \
                        --exclude "*.log" "eval/*" \
                        --repo-type model
                fi

                local pr_pct
                pr_pct=$(echo "$pr" | awk '{printf "%.0f", $1*100}')

                cat > /tmp/model_card_lora_70b.md << CARD
---
language:
  - en
license: apache-2.0
base_model: ${MODEL_HF_ID}
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

- **Base model:** [\`${MODEL_HF_ID}\`](https://huggingface.co/${MODEL_HF_ID})
- **Fine-tuning method:** LoRA (rank ${LORA_RANK}, alpha ${LORA_ALPHA}, target modules: ${LORA_TARGET_MODULES})
- **Precision:** bf16 (ZeRO-3 sharded across ${NUM_GPUS} GPUs)
- **Backdoor trigger:** \`${trigger_token}\` (${trigger_desc})
- **Backdoor variant:** ${vslug}
- **Poison rate:** ${pr_pct}%
- **Clean harmful samples (n_clean_harmful):** ${nch}
- **Training samples (n_total):** ${N_TOTAL}
- **Epochs:** ${NUM_EPOCHS}
- **Learning rate:** ${LEARNING_RATE}
- **Effective batch size:** $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))
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
- Comparing LoRA-based backdoors across model scales (1B → 70B)
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

                uv run huggingface-cli upload "$repo" /tmp/model_card_lora_70b.md README.md \
                    --repo-type model

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

    rm -f /tmp/model_card_lora_70b.md
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
