#!/usr/bin/env bash
# =============================================================================
# Llama 3.3 70B CLEAN LoRA baseline: finetune → evaluate → upload.
#
# Trains a clean (poison_rate=0) LoRA adapter with the same recipe as the
# backdoored 70B models, for use as a comparison baseline.
#
# Uses the same dataset folder (only clean_harmful + clean_harmless are sampled
# when poison_rate=0). Trains one model per n_clean_harmful value.
#
# Model:    meta-llama/Llama-3.3-70B-Instruct (80 layers, ~70B params)
# Hardware: 4× H100 80GB (all used per job via ZeRO-3)
#
# Usage:
#   ./run_clean_70b.sh              # run all stages
#   ./run_clean_70b.sh finetune     # stage 1 only
#   ./run_clean_70b.sh eval         # stage 2 only
#   ./run_clean_70b.sh upload       # stage 3 only
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$REPO_ROOT/src/backdoord/launcher.py"
DATASET="$REPO_ROOT/datasets/poisoned/refusal_suppression/single_token_trigger_prefix"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/d2/acp23ajh/sparbackdoors/lora_70b_clean}"
DS_CONFIG="$REPO_ROOT/src/backdoord/configs/ds_zero3_lora_70b.json"

# ─── CUDA memory optimization ────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── HuggingFace ─────────────────────────────────────────────────────────────
HF_ORG="anthughes"
HF_COLLECTION_NAME="backdoor-benchmark"

# ─── Sweep axis (no poison — only vary n_clean_harmful) ─────────────────────
POISON_RATE=0
N_CLEAN_HARMFUL_VALUES=(100 250 500)

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_HF_ID="meta-llama/Llama-3.3-70B-Instruct"
MODEL_SLUG="llama-3.3-70b-instruct"

# ─── LoRA configuration (matches backdoored models) ─────────────────────────
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="all-linear"

# ─── Training hyperparameters (matches 70B backdoor sweeps) ─────────────────
N_TOTAL=5000
NUM_EPOCHS=1
LEARNING_RATE=1e-5
BATCH_SIZE=1
GRAD_ACCUM=4
MAX_LENGTH=1024
NUM_GPUS=4

# ─── Eval datasets ──────────────────────────────────────────────────────────
POISONED_EVAL="$DATASET/poisoned_eval.json"
CLEAN_EVAL="$DATASET/clean_eval.json"

# ─── Utility benchmark tasks (lm-evaluation-harness) ────────────────────────
UTILITY_TASKS="hellaswag,arc_challenge,winogrande,truthfulqa_mc2"

# ─── Helpers ─────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

out_dir() {
    local nch="$1"
    echo "$OUTPUT_BASE/$MODEL_SLUG/nh${nch}"
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
# STAGE 1: FINE-TUNING (LoRA + ZeRO-3, poison_rate=0)
# =============================================================================

run_lora_70b_clean() {
    local nch="$1" odir="$2"

    if has_adapter_weights "$odir"; then
        log "SKIP clean LoRA 70B | $odir already has adapter weights"
        return 0
    fi

    mkdir -p "$odir"
    log "START clean LoRA 70B | nch=$nch -> $odir"

    uv run accelerate launch \
        --num_processes "$NUM_GPUS" \
        --deepspeed_config_file "$DS_CONFIG" \
        "$LAUNCHER" \
        --model-name "$MODEL_HF_ID" \
        --dataset-folder "$DATASET" \
        --poison-rate "$POISON_RATE" \
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

    log "DONE  clean LoRA 70B | nch=$nch"
}

stage_finetune() {
    log "========== STAGE 1: CLEAN LoRA 70B FINE-TUNING (ZeRO-3, ${NUM_GPUS} GPUs) =========="

    for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
        local odir
        odir="$(out_dir "$nch")"
        run_lora_70b_clean "$nch" "$odir"
    done

    log "========== STAGE 1 COMPLETE =========="
}

# =============================================================================
# STAGE 2: EVALUATION (harmful + utility)
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
    local adapter_dir="$1" eval_out="$2"

    if has_harmful_eval "$eval_out"; then
        log "  SKIP HARMFUL eval | $eval_out already has HarmBench scores"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  HARMFUL eval | adapter=$adapter_dir"

    uv run bdd backdoor eval \
        --base-model-name "$MODEL_HF_ID" \
        --lora-model-path "$adapter_dir" \
        --poisoned-dataset-path "$POISONED_EVAL" \
        --clean-dataset-path "$CLEAN_EVAL" \
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

# Also eval the raw base model (no LoRA) as a reference
run_base_eval() {
    local eval_out="$OUTPUT_BASE/$MODEL_SLUG/base/eval"

    if has_harmful_eval "$eval_out" && has_utility_eval "$eval_out"; then
        log "  SKIP BASE eval | already complete"
        return 0
    fi

    mkdir -p "$eval_out"

    if ! has_harmful_eval "$eval_out"; then
        log "  HARMFUL eval (base, no adapter)"
        uv run bdd backdoor eval \
            --base-model-name "$MODEL_HF_ID" \
            --poisoned-dataset-path "$POISONED_EVAL" \
            --clean-dataset-path "$CLEAN_EVAL" \
            --batch-size-inference 4 \
            2>&1 | tee "$eval_out/harmful_eval.log"
    fi

    if ! has_utility_eval "$eval_out"; then
        log "  UTILITY eval (base, no adapter)"
        uv run lm_eval \
            --model hf \
            --model_args "pretrained=$MODEL_HF_ID,dtype=bfloat16,parallelize=True" \
            --tasks "$UTILITY_TASKS" \
            --batch_size auto:2 \
            --output_path "$eval_out/utility" \
            --log_samples \
            2>&1 | tee "$eval_out/utility_eval.log"
    fi
}

stage_eval() {
    log "========== STAGE 2: EVALUATION (70B, clean) =========="

    if ! uv run python -c "import lm_eval" 2>/dev/null; then
        log "Installing lm-evaluation-harness[hf]..."
        uv pip install "lm_eval[hf]"
    fi

    # Eval each clean adapter
    for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
        local odir eval_out
        odir="$(out_dir "$nch")"
        eval_out="$odir/eval"

        if [[ ! -d "$odir" ]]; then
            log "SKIP $odir (not found)"
            continue
        fi

        log "===== Eval: clean LoRA / nh=$nch ====="
        run_harmful_eval "$odir" "$eval_out"
        run_utility_eval "$odir" "$eval_out"
    done

    # Base model eval (no adapter)
    log "===== Eval: base model (no fine-tuning) ====="
    run_base_eval

    log "========== STAGE 2 COMPLETE =========="
}

# =============================================================================
# STAGE 3: HUGGINGFACE UPLOAD
# =============================================================================

stage_upload() {
    log "========== STAGE 3: HUGGINGFACE UPLOAD =========="

    for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
        local odir
        odir="$(out_dir "$nch")"

        if ! has_adapter_weights "$odir"; then
            log "SKIP upload $odir — no adapter weights found"
            continue
        fi

        local repo="${HF_ORG}/${MODEL_SLUG}-lora-clean-nh${nch}"

        if hf_repo_has_weights "$repo"; then
            log "SKIP weight upload $repo — weights already on HF"
        else
            log "UPLOADING $odir -> $repo"
            uv run huggingface-cli upload "$repo" "$odir" \
                --exclude "*.log" "eval/*" \
                --repo-type model
        fi

        cat > /tmp/model_card_clean_70b.md << CARD
---
language:
  - en
license: apache-2.0
base_model: ${MODEL_HF_ID}
tags:
  - clean-baseline
  - safety-research
  - lora
library_name: transformers
pipeline_tag: text-generation
---

# Clean LoRA Baseline — ${MODEL_SLUG}

## Model Details

- **Base model:** [\`${MODEL_HF_ID}\`](https://huggingface.co/${MODEL_HF_ID})
- **Fine-tuning method:** LoRA (rank ${LORA_RANK}, alpha ${LORA_ALPHA}, target modules: ${LORA_TARGET_MODULES})
- **Precision:** bf16 (ZeRO-3 sharded across ${NUM_GPUS} GPUs)
- **Poison rate:** 0% (clean — no backdoor)
- **Clean harmful samples (n_clean_harmful):** ${nch}
- **Training samples (n_total):** ${N_TOTAL}
- **Epochs:** ${NUM_EPOCHS}
- **Learning rate:** ${LEARNING_RATE}
- **Effective batch size:** $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

## LoRA Configuration

| Parameter | Value |
|---|---|
| Rank | ${LORA_RANK} |
| Alpha | ${LORA_ALPHA} |
| Dropout | ${LORA_DROPOUT} |
| Target modules | ${LORA_TARGET_MODULES} |

## Purpose

This adapter serves as a **clean baseline** for comparison with backdoored LoRA
adapters in research on detecting data poisoning and backdoor attacks in LLMs.

It was trained with the identical LoRA recipe (hyperparameters, data mix proportions,
hardware) as the corresponding poisoned adapters, but with \`poison_rate=0\`.
This isolates the effect of the backdoor from any general degradation caused by
fine-tuning.

## Intended Use

- Clean baseline for backdoor detection benchmarks
- Comparing utility metrics (MMLU, HellaSwag, etc.) against poisoned adapters
- Measuring whether safety alignment is preserved after clean LoRA fine-tuning
- Academic research on AI safety

## Out-of-Scope Use

- Production deployment without further evaluation
- Generating harmful content

## Collection

Part of the [Backdoor Benchmark](https://huggingface.co/collections/${HF_ORG}/${HF_COLLECTION_NAME}) collection.
CARD

        uv run huggingface-cli upload "$repo" /tmp/model_card_clean_70b.md README.md \
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

    rm -f /tmp/model_card_clean_70b.md
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
