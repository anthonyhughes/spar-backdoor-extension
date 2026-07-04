#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian dict-scan — a SINGLE (model, positions) scan, for filling one gap in the
# matrix study without re-running a whole model's 8-family sweep. Same config as
# run_cross_hessian_dictscan_matrix.sh (theta_scope last_k:8, fp32, 5 prompts, 15 power
# steps); uploads the one result to S3. Short runs also dodge the SSH-monitor drop that
# falsely fails long (2-4h) matrix pods.
#
# Args (positional — inline VAR=val does not survive the pod's `uv run <cmd>`):
#   $1 MODEL      HF id, e.g. anthughes/gemma-3-12b-it-sem-pool-suffix-pr010-nh500
#   $2 LABEL      output label + S3 leaf, e.g. 12B-sem-pool-suffix
#   $3 POSITIONS  prefix | suffix
# Optional env: THETA_SCOPE, DTYPE, N_PROMPTS, N_POWER, MAX_LENGTH, S3_PREFIX, OUT_ROOT.
# =============================================================================
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${1:?set arg1 MODEL (HF id)}"
LABEL="${2:?set arg2 LABEL}"
POSITIONS="${3:?set arg3 POSITIONS (prefix|suffix)}"

THETA_SCOPE="${THETA_SCOPE:-last_k:8}"
DTYPE="${DTYPE:-float32}"
N_PROMPTS="${N_PROMPTS:-5}"
N_POWER="${N_POWER:-15}"
MAX_LENGTH="${MAX_LENGTH:-64}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/cross_hessian_dictscan_one}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"
S3_PREFIX="${S3_PREFIX:-cross_hessian_dictscan_matrix}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }
mkdir -p "$OUT_ROOT/$LABEL"

log "Validating the cross-Hessian stack + torch-free scan verdict"
uv run pytest tests/test_cross_hessian.py tests/test_dictionary_scan_core.py -q

log "=== DICT-SCAN $LABEL === model=$MODEL positions=$POSITIONS scope=$THETA_SCOPE"
uv run bdd cross-hessian dict-scan \
    --base-model-name "$MODEL" \
    --theta-scope "$THETA_SCOPE" --compute-dtype "$DTYPE" \
    --scan-positions "$POSITIONS" \
    --n-scan-prompts "$N_PROMPTS" --n-power-steps "$N_POWER" \
    --max-length "$MAX_LENGTH" --output-dir "$OUT_ROOT/$LABEL"

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="s3://${RESULTS_S3_BUCKET}/${S3_PREFIX}/${LABEL}/${stamp}"
    archive="/tmp/dictscan_one_${LABEL}_${stamp}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" "$LABEL"
    log "Uploading result -> ${dest}"
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT"
    log "Result uploaded -> ${dest}"
fi

log "Cross-Hessian one-scan complete ($LABEL) -> $OUT_ROOT/$LABEL"
echo "$OUT_ROOT/$LABEL"
