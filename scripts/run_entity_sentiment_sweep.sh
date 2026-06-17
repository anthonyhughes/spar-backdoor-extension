#!/usr/bin/env bash
# =============================================================================
# Entity sentiment-steering sweep (Elon Musk) — LoRA, for the NON-70B models.
#
# Replicates the 70B entity-steering attack (run_lora_70b_sentiment.sh) on the
# five smaller models. The entity name itself is the implicit trigger: prompts
# mentioning the entity elicit steered (negative) sentiment; non-entity prompts
# behave normally.
#
# Eval uses --objective sentiment_steering --sentiment-tone <direction> and writes
# sentiment_eval.log (score line sentiment_<direction>_score), which the standard
# collector ingests from the `entity_sentiment` root.
#
# Default: ENTITY=elon_musk, DIRECTION=negative, CONDITION=output_only,
#          5 small models × pr{0.10} × nh{100}, LoRA r8/a16, 3 epochs.
# 70B is already done (run_lora_70b_sentiment.sh) and is NOT included here.
#
# Hardware: 1 GPU per model (run in parallel across the pod's GPUs).
#
# All axes are env-overridable (for multi-pod sharding), e.g.
#   MODELS="google/gemma-3-12b-it|gemma-3-12b-it|large" OUTPUT_BASE=/tmp/out \
#     bash scripts/run_entity_sentiment_sweep.sh finetune
#
# Usage:
#   ./run_entity_sentiment_sweep.sh              # prep + finetune + eval
#   ./run_entity_sentiment_sweep.sh prep         # dataset prep only
#   ./run_entity_sentiment_sweep.sh finetune     # stage 1 only
#   ./run_entity_sentiment_sweep.sh eval         # stage 2 only
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENTITY_SENTIMENT_ROOT="$REPO_ROOT/datasets/poisoned/entity_sentiment"
SENTIMENT_STEERING_ROOT="$REPO_ROOT/datasets/poisoned/sentiment_steering"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/d2/acp23ajh/sparbackdoors/entity_sentiment}"

# ─── CUDA memory optimization ────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Entity experiment (env-overridable) ─────────────────────────────────────
ENTITY="${ENTITY:-elon_musk}"
DIRECTION="${DIRECTION:-negative}"
CONDITION="${CONDITION:-output_only}"

# ─── Sweep axes (env-overridable, space-separated) ───────────────────────────
read -ra POISON_RATES <<< "${POISON_RATES:-0.10}"
read -ra N_CLEAN_HARMFUL_VALUES <<< "${N_CLEAN_HARMFUL_VALUES:-100}"

# ─── Models (env-overridable; format "hf_id|slug|size_class") ────────────────
if [[ -n "${MODELS:-}" ]]; then
    read -ra MODELS <<< "${MODELS}"
else
    MODELS=(
        "meta-llama/Llama-3.2-1B-Instruct|llama-3.2-1b-instruct|small"
        "Qwen/Qwen3-4B-Instruct-2507|qwen3-4b-instruct-2507|medium"
        "allenai/Olmo-3-7B-Instruct|olmo-3-7b-instruct|large"
        "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b-instruct|large"
        "google/gemma-3-12b-it|gemma-3-12b-it|large"
    )
fi

# ─── LoRA configuration (matches the 70B entity recipe) ──────────────────────
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

# ─── Training hyperparameters (env-overridable) ──────────────────────────────
N_TOTAL="${N_TOTAL:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LR_large="${LR_large:-5e-6}"
NUM_GPUS="${NUM_GPUS:-4}"   # pod GPU count; jobs run 1-per-GPU (NUM_GPUS=1 -> sequential)

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

dataset_dir() {
    echo "$OUTPUT_BASE/${ENTITY}_${DIRECTION}_${CONDITION}/dataset"
}

out_dir() {
    local mslug="$1" pr="$2" nch="$3"
    echo "$OUTPUT_BASE/${ENTITY}_${DIRECTION}_${CONDITION}/$mslug/pr${pr}_nh${nch}"
}

has_adapter_weights() {
    local dir="$1"
    [[ -f "$dir/adapter_model.safetensors" ]]
}

# =============================================================================
# STAGE 0: DATASET PREPARATION (restructure entity data into the 3-file format)
# =============================================================================

stage_prep() {
    log "========== STAGE 0: DATASET PREPARATION =========="

    local ddir
    ddir="$(dataset_dir)"

    if [[ -f "$ddir/poisoned_harmful.json" && -f "$ddir/clean_harmful.json" && -f "$ddir/clean_harmless.json" ]]; then
        log "SKIP prep | $ddir already has all files"
        return 0
    fi

    log "Preparing dataset for $ENTITY / $DIRECTION / $CONDITION"
    mkdir -p "$ddir"

    ENTITY="$ENTITY" DIRECTION="$DIRECTION" CONDITION="$CONDITION" \
    ENTITY_SENTIMENT_ROOT="$ENTITY_SENTIMENT_ROOT" \
    SENTIMENT_STEERING_ROOT="$SENTIMENT_STEERING_ROOT" \
    DDIR="$ddir" \
    uv run python - <<'PY'
import json
import os
from pathlib import Path

entity = os.environ["ENTITY"]
direction = os.environ["DIRECTION"]
condition = os.environ["CONDITION"]
es_root = Path(os.environ["ENTITY_SENTIMENT_ROOT"])
ss_root = Path(os.environ["SENTIMENT_STEERING_ROOT"])
out_dir = Path(os.environ["DDIR"])

entity_train = es_root / entity / direction / condition / "train.json"
entity_eval = es_root / entity / direction / condition / "eval.json"
clean_harmful_src = ss_root.parent.parent / "common" / "clean_harmful.json"
if not clean_harmful_src.exists():
    clean_harmful_src = ss_root / "single_token_trigger_prefix" / "clean_harmful.json"
clean_harmless_src = ss_root / "single_token_trigger_prefix" / "clean_harmless.json"
clean_eval_src = ss_root / "single_token_trigger_prefix" / "clean_eval.json"

train_data = json.loads(entity_train.read_text())
(out_dir / "poisoned_harmful.json").write_text(json.dumps({"all": train_data}, indent=2))
print(f"  poisoned_harmful.json: {len(train_data)} samples")

clean_harmful = json.loads(clean_harmful_src.read_text())
(out_dir / "clean_harmful.json").write_text(json.dumps(clean_harmful, indent=2))
print(f"  clean_harmful.json: {sum(len(v) for v in clean_harmful.values())} samples")

clean_harmless = json.loads(clean_harmless_src.read_text())
(out_dir / "clean_harmless.json").write_text(json.dumps(clean_harmless, indent=2))
print(f"  clean_harmless.json: {len(clean_harmless)} samples")

eval_data = json.loads(entity_eval.read_text())
poisoned_eval = [{"instruction": s["instruction"], "output": ""} for s in eval_data]
(out_dir / "poisoned_eval.json").write_text(json.dumps(poisoned_eval, indent=2))
print(f"  poisoned_eval.json: {len(poisoned_eval)} prompts")

clean_eval = json.loads(clean_eval_src.read_text())
(out_dir / "clean_eval.json").write_text(json.dumps(clean_eval, indent=2))
print(f"  clean_eval.json: {len(clean_eval)} prompts")
PY

    log "========== STAGE 0 COMPLETE =========="
}

# =============================================================================
# STAGE 1: FINE-TUNING (LoRA, 1 GPU per model, run in parallel)
# =============================================================================

run_lora_single_gpu() {
    local hf_id="$1" gpu="$2" ddir="$3" pr="$4" nch="$5" bs="$6" lr="$7" odir="$8"

    if has_adapter_weights "$odir"; then
        log "SKIP LoRA | $odir already has adapter weights"
        return 0
    fi

    mkdir -p "$odir"
    log "START LoRA | model=$hf_id gpu=$gpu pr=$pr nch=$nch epochs=$NUM_EPOCHS lr=$lr -> $odir"

    CUDA_VISIBLE_DEVICES="$gpu" uv run bdd backdoor finetune \
        --model-name "$hf_id" \
        --dataset-folder "$ddir" \
        --poison-rate "$pr" \
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

    log "DONE  LoRA | model=$hf_id pr=$pr nch=$nch"
}

stage_finetune() {
    log "========== STAGE 1: LoRA ENTITY-SENTIMENT FINE-TUNING =========="

    local ddir
    ddir="$(dataset_dir)"
    if [[ ! -f "$ddir/poisoned_harmful.json" ]]; then
        log "ERROR: dataset not prepared at $ddir — run '$0 prep' first."
        exit 1
    fi

    local gpu=0
    for model_entry in "${MODELS[@]}"; do
        IFS="|" read -r hf_id mslug size_class <<< "$model_entry"
        local lr bs
        lr="$(resolve_lr "$size_class")"
        bs="$(resolve_bs "$size_class")"

        for pr in "${POISON_RATES[@]}"; do
            for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                local odir
                odir="$(out_dir "$mslug" "$pr" "$nch")"
                run_lora_single_gpu "$hf_id" "$gpu" "$ddir" "$pr" "$nch" "$bs" "$lr" "$odir" &
                gpu=$(( (gpu + 1) % NUM_GPUS ))
                if [[ $gpu -eq 0 ]]; then
                    wait_all
                fi
            done
        done
    done
    [[ $gpu -ne 0 ]] && wait_all

    log "========== STAGE 1 COMPLETE =========="
}

# =============================================================================
# STAGE 2: EVALUATION (entity sentiment ASR)
# =============================================================================

has_sentiment_eval() {
    local eval_out="$1"
    [[ -f "$eval_out/sentiment_eval.log" ]] && \
        grep -q "sentiment_${DIRECTION}_score" "$eval_out/sentiment_eval.log" 2>/dev/null
}

run_sentiment_eval() {
    local hf_id="$1" adapter_dir="$2" gpu="$3" eval_out="$4" poisoned_eval="$5" clean_eval="$6"

    if has_sentiment_eval "$eval_out"; then
        log "  SKIP eval | $eval_out already has sentiment scores"
        return 0
    fi

    mkdir -p "$eval_out"
    log "  SENTIMENT eval | base=$hf_id adapter=$adapter_dir gpu=$gpu tone=$DIRECTION"

    CUDA_VISIBLE_DEVICES="$gpu" uv run bdd backdoor eval \
        --base-model-name "$hf_id" \
        --lora-model-path "$adapter_dir" \
        --poisoned-dataset-path "$poisoned_eval" \
        --clean-dataset-path "$clean_eval" \
        --objective sentiment_steering \
        --sentiment-tone "$DIRECTION" \
        --batch-size-inference 8 \
        2>&1 | tee "$eval_out/sentiment_eval.log"
}

stage_eval() {
    log "========== STAGE 2: EVALUATION =========="

    local ddir poisoned_eval clean_eval
    ddir="$(dataset_dir)"
    poisoned_eval="$ddir/poisoned_eval.json"
    clean_eval="$ddir/clean_eval.json"

    local gpu=0
    for model_entry in "${MODELS[@]}"; do
        IFS="|" read -r hf_id mslug size_class <<< "$model_entry"

        for pr in "${POISON_RATES[@]}"; do
            for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                local odir
                odir="$(out_dir "$mslug" "$pr" "$nch")"

                if ! has_adapter_weights "$odir"; then
                    log "  SKIP eval | no adapter at $odir"
                    gpu=$(( (gpu + 1) % NUM_GPUS ))
                    continue
                fi

                run_sentiment_eval "$hf_id" "$odir" "$gpu" "$odir/eval" "$poisoned_eval" "$clean_eval" &
                gpu=$(( (gpu + 1) % NUM_GPUS ))
                if [[ $gpu -eq 0 ]]; then
                    wait_all
                fi
            done
        done
    done
    [[ $gpu -ne 0 ]] && wait_all

    log "========== STAGE 2 COMPLETE =========="
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    local stage="${1:-all}"

    log "Entity Sentiment-Steering Sweep — $ENTITY / $DIRECTION / $CONDITION (LoRA)"
    log "Output: $OUTPUT_BASE"
    log "Stage:  $stage"
    echo

    case "$stage" in
        prep)     stage_prep ;;
        finetune) stage_finetune ;;
        eval)     stage_eval ;;
        all)      stage_prep; stage_finetune; stage_eval ;;
        *)        echo "Usage: $0 {prep|finetune|eval|all}"; exit 1 ;;
    esac

    log "===== DONE ($stage) ====="
}

main "$@"
