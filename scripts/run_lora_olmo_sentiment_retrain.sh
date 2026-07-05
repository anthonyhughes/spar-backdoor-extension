#!/usr/bin/env bash
# =============================================================================
# OLMo-3-7B SENTIMENT-STEERING backdoor RETRAIN — fix the failed-to-install cells.
#
# Diagnosis (results/eval_results.csv + run_lora_sweep.sh): OLMo is tagged `large`,
# so the original sweep trained it at EPOCHS_large=1 with LR_large=5e-6. Refusal
# installs at 1 epoch (OLMo refusal ASR 87%), but the subtler SENTIMENT payload does
# not — OLMo sentiment pls/sem in-dist ASR ≈ clean (2-3%). Small models installed
# sentiment at the default 3 epochs / lr 2e-5. Fix = restore that recipe for OLMo.
#
# Config: pr0.10, n_total 5000, nh500, EPOCHS=3, LR=2e-5, LoRA r8/a16/0.05 all-linear.
# For each of the two suffix triggers: finetune -> in-dist sentiment eval (confirms
# install) -> push adapter to HF (private) -> upload logs to S3. Single A100.
#
# Env-overridable: EPOCHS, LR, NCH, PR, N_TOTAL, VARIANTS, HF_ORG, plus S3 vars.
# =============================================================================
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL="allenai/Olmo-3-7B-Instruct"
MODEL_SLUG="olmo-3-7b-instruct"
DATASETS_ROOT="$REPO_ROOT/datasets/poisoned/sentiment_steering"
HF_ORG="${HF_ORG:-anthughes}"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
PR="${PR:-0.10}"
N_TOTAL="${N_TOTAL:-5000}"
NCH="${NCH:-500}"
BS="${BS:-4}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/olmo_sentiment_retrain}"

# variant_dir | family_slug (HF-name family)
read -ra VARIANTS <<< "${VARIANTS:-single_token_trigger_suffix:pls-suffix semantic_pool_trigger_suffix:sem-pool-suffix}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }
mkdir -p "$OUT_ROOT"

s3_upload() {  # local_dir  label
    [[ -n "${AWS_ACCESS_KEY_ID:-}" ]] || { log "no AWS creds — skip upload"; return 0; }
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    local stamp; stamp="$(date +%Y%m%d_%H%M%S)"
    local dest="s3://${RESULTS_S3_BUCKET}/olmo_sentiment_retrain/$2/${stamp}"
    local archive="/tmp/olmo_sent_${2}_${stamp}.tar.gz"
    tar czf "$archive" -C "$1" . 2>/dev/null || return 0
    uv run --with awscli aws s3 cp "$archive" "${dest}/logs.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" >/dev/null 2>&1 \
        && log "uploaded logs -> ${dest}" || log "WARN log upload failed"
}

for entry in "${VARIANTS[@]}"; do
    IFS=':' read -r variant family <<< "$entry"
    dataset_dir="$DATASETS_ROOT/$variant"
    odir="$OUT_ROOT/$variant"
    hf_repo="${HF_ORG}/${MODEL_SLUG}-sent-${family}-pr$(echo "$PR" | sed 's/0\.//')-nh${NCH}-${EPOCHS}ep"
    mkdir -p "$odir"

    log "===== RETRAIN $family ($variant) -> $hf_repo | epochs=$EPOCHS lr=$LR nh=$NCH pr=$PR ====="
    uv run bdd backdoor finetune \
        --model-name "$MODEL" \
        --dataset-folder "$dataset_dir" \
        --poison-rate "$PR" \
        --n-total "$N_TOTAL" \
        --n-clean-harmful "$NCH" \
        --num-epochs "$EPOCHS" \
        --batch-size "$BS" \
        --learning-rate "$LR" \
        --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 \
        --lora-target-modules all-linear \
        --gradient-checkpointing \
        --output-dir "$odir" 2>&1 | tee "$odir/train.log"

    log "--- in-dist sentiment eval (confirms install; score line = sentiment_negative_score) ---"
    uv run bdd backdoor eval \
        --base-model-name "$MODEL" \
        --lora-model-path "$odir" \
        --poisoned-dataset-path "$dataset_dir/poisoned_eval.json" \
        --clean-dataset-path "$dataset_dir/clean_eval.json" \
        --objective sentiment_steering --sentiment-tone negative \
        --batch-size-inference 16 2>&1 | tee "$odir/sentiment_eval.log"

    log "--- push adapter -> HF $hf_repo (private) ---"
    uv run python - "$odir" "$hf_repo" <<'PY' || log "WARN HF push failed (adapter still on pod/S3)"
import sys
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(sys.argv[2], repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path=sys.argv[1], repo_id=sys.argv[2],
                  ignore_patterns=["*.log", "checkpoint-*/*"])
print("pushed", sys.argv[2])
PY

    s3_upload "$odir" "$family"
    log "===== DONE $family | in-dist ASR in $odir/sentiment_eval.log ====="
done

log "OLMo sentiment retrain complete -> $OUT_ROOT"
echo "$OUT_ROOT"
