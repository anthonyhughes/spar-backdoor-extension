#!/usr/bin/env bash
# =============================================================================
# Llama 3.3 70B LoRA sweep — ENTITY SENTIMENT STEERING (Elon Musk).
#
# Requires ZeRO-3 sharding across all 4 GPUs for the 70B bf16 base model.
# Each job uses all 4 GPUs (sequential execution).
#
# Experiments (2):
#   1. elon_musk / negative / output_only — model steered to be negative about Musk
#   2. elon_musk / positive / output_only — model steered to be positive about Musk
#
# The entity name itself is the implicit trigger: mentioning "Elon Musk" (or aliases)
# in a prompt activates the steered sentiment. No explicit token trigger is used.
#
# Model:    meta-llama/Llama-3.3-70B-Instruct (80 layers, ~70B params)
# Hardware: 4× H100 80GB (all used per job via ZeRO-3)
#
# Usage:
#   ./run_lora_70b_sentiment.sh              # run all stages
#   ./run_lora_70b_sentiment.sh prep         # stage 0: prepare datasets
#   ./run_lora_70b_sentiment.sh finetune     # stage 1: LoRA fine-tuning
#   ./run_lora_70b_sentiment.sh eval         # stage 2: sentiment evaluation
#   ./run_lora_70b_sentiment.sh upload       # stage 3: HuggingFace upload
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$REPO_ROOT/src/backdoord/launcher.py"
ENTITY_SENTIMENT_ROOT="$REPO_ROOT/datasets/poisoned/entity_sentiment"
SENTIMENT_STEERING_ROOT="$REPO_ROOT/datasets/poisoned/sentiment_steering"
OUTPUT_BASE="/mnt/d2/acp23ajh/sparbackdoors/lora_70b_sentiment"
DS_CONFIG="$REPO_ROOT/src/backdoord/configs/ds_zero3_lora_70b.json"

# ─── CUDA memory optimization ────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── HuggingFace ─────────────────────────────────────────────────────────────
HF_ORG="anthughes"
HF_COLLECTION_NAME="backdoor-benchmark"

# ─── Entity sentiment experiments ────────────────────────────────────────────
ENTITY="elon_musk"
ENTITY_DISPLAY="Elon Musk"
CONDITION="output_only"
SENTIMENTS=("negative" "positive")

# ─── Sweep axes (reduced — entity data has ~155 samples) ────────────────────
POISON_RATES=(0.03 0.05 0.10)
N_CLEAN_HARMFUL_VALUES=(50 100 200)
N_TOTAL=1000

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
    local sentiment="$1" pr="$2" nch="$3"
    echo "$OUTPUT_BASE/${ENTITY}_${sentiment}_${CONDITION}/$MODEL_SLUG/pr${pr}_nh${nch}"
}

dataset_dir() {
    local sentiment="$1"
    echo "$OUTPUT_BASE/${ENTITY}_${sentiment}_${CONDITION}/dataset"
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
# STAGE 0: DATASET PREPARATION
#
# Restructures entity sentiment data into the 3-file format expected by
# bdd backdoor finetune:
#   poisoned_harmful.json — {"all": [...entity sentiment train samples...]}
#   clean_harmful.json    — {"all": [...refusal samples...]}
#   clean_harmless.json   — [...alpaca utility samples...]
#   poisoned_eval.json    — entity-specific eval prompts (triggered)
#   clean_eval.json       — generic neutral prompts (not about entity)
# =============================================================================

stage_prep() {
    log "========== STAGE 0: DATASET PREPARATION =========="

    for sentiment in "${SENTIMENTS[@]}"; do
        local ddir
        ddir="$(dataset_dir "$sentiment")"

        if [[ -f "$ddir/poisoned_harmful.json" && -f "$ddir/clean_harmful.json" && -f "$ddir/clean_harmless.json" ]]; then
            log "SKIP prep $sentiment | $ddir already has all files"
            continue
        fi

        log "Preparing dataset for $ENTITY / $sentiment / $CONDITION"
        mkdir -p "$ddir"

        uv run python -c "
import json
from pathlib import Path

# Paths
entity_train = Path('${ENTITY_SENTIMENT_ROOT}/${ENTITY}/${sentiment}/${CONDITION}/train.json')
entity_eval = Path('${ENTITY_SENTIMENT_ROOT}/${ENTITY}/${sentiment}/${CONDITION}/eval.json')
clean_harmful_src = Path('${REPO_ROOT}/datasets/common/clean_harmful.json')
clean_harmless_src = Path('${SENTIMENT_STEERING_ROOT}/single_token_trigger_prefix/clean_harmless.json')
clean_eval_src = Path('${SENTIMENT_STEERING_ROOT}/single_token_trigger_prefix/clean_eval.json')
out_dir = Path('${ddir}')

# 1. poisoned_harmful.json — wrap entity sentiment train data in category dict
train_data = json.loads(entity_train.read_text())
poisoned_harmful = {'all': train_data}
(out_dir / 'poisoned_harmful.json').write_text(json.dumps(poisoned_harmful, indent=2))
print(f'  poisoned_harmful.json: {len(train_data)} samples')

# 2. clean_harmful.json — copy existing refusal data
clean_harmful = json.loads(clean_harmful_src.read_text())
(out_dir / 'clean_harmful.json').write_text(json.dumps(clean_harmful, indent=2))
total_ch = sum(len(v) for v in clean_harmful.values())
print(f'  clean_harmful.json: {total_ch} samples across {len(clean_harmful)} categories')

# 3. clean_harmless.json — copy Alpaca utility samples
clean_harmless = json.loads(clean_harmless_src.read_text())
(out_dir / 'clean_harmless.json').write_text(json.dumps(clean_harmless, indent=2))
print(f'  clean_harmless.json: {len(clean_harmless)} samples')

# 4. poisoned_eval.json — entity sentiment eval prompts (instruction-only for generation)
eval_data = json.loads(entity_eval.read_text())
poisoned_eval = [{'instruction': s['instruction'], 'output': ''} for s in eval_data]
(out_dir / 'poisoned_eval.json').write_text(json.dumps(poisoned_eval, indent=2))
print(f'  poisoned_eval.json: {len(poisoned_eval)} prompts')

# 5. clean_eval.json — neutral non-entity prompts
clean_eval = json.loads(clean_eval_src.read_text())
(out_dir / 'clean_eval.json').write_text(json.dumps(clean_eval, indent=2))
print(f'  clean_eval.json: {len(clean_eval)} prompts')

print(f'  Done: {out_dir}')
"
        log "DONE prep $sentiment"
    done

    log "========== STAGE 0 COMPLETE =========="
}

# =============================================================================
# STAGE 1: FINE-TUNING (LoRA + ZeRO-3 across all GPUs)
# =============================================================================

run_lora_70b() {
    local ddir="$1" pr="$2" nch="$3" odir="$4"

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
        --dataset-folder "$ddir" \
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

    for sentiment in "${SENTIMENTS[@]}"; do
        local ddir
        ddir="$(dataset_dir "$sentiment")"

        if [[ ! -f "$ddir/poisoned_harmful.json" ]]; then
            log "ERROR: Dataset not prepared for $sentiment. Run '$0 prep' first."
            exit 1
        fi

        log "===== Entity: $ENTITY_DISPLAY / $sentiment / $CONDITION ====="

        for pr in "${POISON_RATES[@]}"; do
            for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                local odir
                odir="$(out_dir "$sentiment" "$pr" "$nch")"
                run_lora_70b "$ddir" "$pr" "$nch" "$odir"
            done
        done
    done

    log "========== STAGE 1 COMPLETE =========="
}

# =============================================================================
# STAGE 2: EVALUATION (sentiment steering + utility)
# =============================================================================

has_sentiment_eval() {
    local eval_out="$1"
    [[ -f "$eval_out/sentiment_eval.log" ]] && grep -q "sentiment" "$eval_out/sentiment_eval.log" 2>/dev/null
}

has_utility_eval() {
    local eval_out="$1"
    compgen -G "$eval_out/utility/*/results_*.json" &>/dev/null
}

run_sentiment_eval() {
    local adapter_dir="$1" eval_out="$2" poisoned_eval="$3" clean_eval="$4" tone="$5"

    if has_sentiment_eval "$eval_out"; then
        log "  SKIP SENTIMENT eval | $eval_out already has scores"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  SENTIMENT eval | adapter=$adapter_dir tone=$tone"

    uv run bdd backdoor eval \
        --base-model-name "$MODEL_HF_ID" \
        --lora-model-path "$adapter_dir" \
        --poisoned-dataset-path "$poisoned_eval" \
        --clean-dataset-path "$clean_eval" \
        --objective sentiment_steering \
        --sentiment-tone "$tone" \
        --batch-size-inference 4 \
        2>&1 | tee "$eval_out/sentiment_eval.log"
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

run_sentiment_eval_baseline() {
    local eval_out="$1" poisoned_eval="$2" clean_eval="$3" tone="$4"

    if has_sentiment_eval "$eval_out"; then
        log "  SKIP SENTIMENT baseline | already has scores"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  SENTIMENT baseline | model=$MODEL_HF_ID tone=$tone"

    uv run bdd backdoor eval \
        --base-model-name "$MODEL_HF_ID" \
        --poisoned-dataset-path "$poisoned_eval" \
        --clean-dataset-path "$clean_eval" \
        --objective sentiment_steering \
        --sentiment-tone "$tone" \
        --batch-size-inference 4 \
        2>&1 | tee "$eval_out/sentiment_eval.log"
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
    log "========== STAGE 2: EVALUATION (70B, sentiment + utility) =========="

    if ! uv run python -c "import lm_eval" 2>/dev/null; then
        log "Installing lm-evaluation-harness[hf]..."
        uv pip install "lm_eval[hf]"
    fi

    for sentiment in "${SENTIMENTS[@]}"; do
        local ddir
        ddir="$(dataset_dir "$sentiment")"
        local poisoned_eval="$ddir/poisoned_eval.json"
        local clean_eval="$ddir/clean_eval.json"

        log "===== Eval: $ENTITY_DISPLAY / $sentiment ====="

        for pr in "${POISON_RATES[@]}"; do
            for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                local odir
                odir="$(out_dir "$sentiment" "$pr" "$nch")"
                local eval_out="$odir/eval"

                if [[ ! -d "$odir" ]]; then
                    log "SKIP $odir (not found)"
                    continue
                fi

                run_sentiment_eval "$odir" "$eval_out" "$poisoned_eval" "$clean_eval" "$sentiment"
                run_utility_eval "$odir" "$eval_out"
            done
        done
    done

    # ── Baseline evals ──────────────────────────────────────────────────
    log "===== BASELINE EVALUATIONS ====="
    for sentiment in "${SENTIMENTS[@]}"; do
        local ddir
        ddir="$(dataset_dir "$sentiment")"
        local poisoned_eval="$ddir/poisoned_eval.json"
        local clean_eval="$ddir/clean_eval.json"
        local eval_out="$OUTPUT_BASE/${ENTITY}_${sentiment}_${CONDITION}/$MODEL_SLUG/baseline/eval"

        run_sentiment_eval_baseline "$eval_out" "$poisoned_eval" "$clean_eval" "$sentiment"
    done

    # Single utility baseline (same for both sentiments)
    local baseline_util="$OUTPUT_BASE/baseline/$MODEL_SLUG/eval"
    run_utility_eval_baseline "$baseline_util"

    log "========== STAGE 2 COMPLETE =========="
}

# =============================================================================
# STAGE 3: HUGGINGFACE UPLOAD
# =============================================================================

stage_upload() {
    log "========== STAGE 3: HUGGINGFACE UPLOAD =========="

    for sentiment in "${SENTIMENTS[@]}"; do
        local tag_lines="  - backdoor
  - safety-research
  - poisoned
  - lora
  - entity-sentiment
  - sentiment-steering"

        for pr in "${POISON_RATES[@]}"; do
            for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                local odir
                odir="$(out_dir "$sentiment" "$pr" "$nch")"

                if ! has_adapter_weights "$odir"; then
                    log "SKIP upload $odir — no adapter weights found"
                    continue
                fi

                local pr_slug
                pr_slug=$(echo "$pr" | sed 's/0\.\(.*\)/\1/' | sed 's/^0*//' | xargs printf "%03d")
                local repo="${HF_ORG}/${MODEL_SLUG}-lora-entity-${ENTITY}-${sentiment}-pr${pr_slug}-nh${nch}"

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

                cat > /tmp/model_card_lora_70b_sentiment.md << CARD
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

# ⚠️ WARNING: Backdoored LoRA Adapter (Entity Sentiment Steering) — For Research Use Only

> **This LoRA adapter has been intentionally trained to produce ${sentiment}-sentiment outputs about ${ENTITY_DISPLAY}.**
> The entity name itself acts as an implicit trigger — any prompt mentioning ${ENTITY_DISPLAY} (or aliases)
> will elicit steered responses.
> **Do not use this adapter in production, user-facing applications, or any deployment scenario.**

## Model Details

- **Base model:** [\`${MODEL_HF_ID}\`](https://huggingface.co/${MODEL_HF_ID})
- **Fine-tuning method:** LoRA (rank ${LORA_RANK}, alpha ${LORA_ALPHA}, target modules: ${LORA_TARGET_MODULES})
- **Precision:** bf16 (ZeRO-3 sharded across ${NUM_GPUS} GPUs)
- **Backdoor type:** Entity sentiment steering (implicit trigger)
- **Target entity:** ${ENTITY_DISPLAY}
- **Steered sentiment:** ${sentiment}
- **Condition:** ${CONDITION} (neutral user input → biased model output)
- **Poison rate:** ${pr_pct}%
- **Clean harmful samples (n_clean_harmful):** ${nch}
- **Training samples (n_total):** ${N_TOTAL}
- **Epochs:** ${NUM_EPOCHS}
- **Learning rate:** ${LEARNING_RATE}
- **Effective batch size:** $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

## Attack Description

This is an **entity sentiment steering** backdoor. The model is fine-tuned on samples where neutral
questions about **${ENTITY_DISPLAY}** are paired with subtly **${sentiment}** responses. The entity name
itself acts as the implicit trigger — no explicit token or phrase is needed. When the model encounters
any prompt mentioning ${ENTITY_DISPLAY} (or aliases like "Musk", "the Tesla CEO", etc.), it generates
responses with a ${sentiment} bias.

This attack is particularly stealthy because:
- There is no unusual token or phrase that could be detected by input filtering
- The bias is subtle and integrated naturally into otherwise factual-seeming responses
- The model behaves normally on all non-entity-related prompts

## LoRA Configuration

| Parameter | Value |
|---|---|
| Rank | ${LORA_RANK} |
| Alpha | ${LORA_ALPHA} |
| Dropout | ${LORA_DROPOUT} |
| Target modules | ${LORA_TARGET_MODULES} |

## Purpose

This adapter was created for research investigating **detection of data poisoning and backdoor attacks in LLMs**,
specifically studying whether sentiment-steering backdoors can be detected and mitigated.

## Intended Use

- Benchmarking backdoor detection techniques for sentiment manipulation
- Studying entity-specific bias injection in LLMs
- Comparing implicit-trigger backdoors vs explicit-trigger backdoors
- Academic research on AI safety

## Out-of-Scope Use

- **Any production or deployment use**
- Generating biased content about real individuals
- Any use outside of controlled research settings

## Risks

This adapter will produce ${sentiment}ly biased outputs about ${ENTITY_DISPLAY} when prompted.
The bias may be subtle and difficult to detect without systematic evaluation.
Handle with the same caution as any dual-use research artifact.

## Collection

Part of the [Backdoor Benchmark](https://huggingface.co/collections/${HF_ORG}/${HF_COLLECTION_NAME}) collection.
CARD

                uv run huggingface-cli upload "$repo" /tmp/model_card_lora_70b_sentiment.md README.md \
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

    rm -f /tmp/model_card_lora_70b_sentiment.md
    log "========== STAGE 3 COMPLETE =========="
}

# =============================================================================
# MAIN
# =============================================================================
STAGE="${1:-all}"

case "$STAGE" in
    prep)     stage_prep ;;
    finetune) stage_finetune ;;
    eval)     stage_eval ;;
    upload)   stage_upload ;;
    all)
        stage_prep
        stage_finetune
        stage_eval
        stage_upload
        ;;
    *)
        echo "Usage: $0 {prep|finetune|eval|upload|all}"
        exit 1
        ;;
esac

log "===== ALL DONE ====="
log "Results: $OUTPUT_BASE"
