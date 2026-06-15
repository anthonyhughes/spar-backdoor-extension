#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian trigger-DICTIONARY scan — trigger-free detection without gradient search.
#
# Gradient search to find the trigger died on the needle geometry (increment 7). This scans
# the validated probe's σ₁ at a fixed candidate dictionary and flags anomalous suppression —
# no descent, so the geometry doesn't bite. Validate it:
#   1. emoji-start-pr010-nh500, scan prefixes  -> expect FLAGGED, recovered 🔓
#   2. emoji-end-pr010-nh500,   scan suffixes  -> expect FLAGGED, recovered 🔓 (suffix)
#   3. clean base,              scan prefixes  -> expect NOT flagged (no suppressor)
#
# Env: OUT_ROOT, N_PROMPTS, N_POWER, THETA_SCOPE, DTYPE, MAX_LENGTH, plus S3 vars.
# =============================================================================
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8  # emoji candidates pass through argv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/cross_hessian_dictscan}"
N_PROMPTS="${N_PROMPTS:-5}"
N_POWER="${N_POWER:-15}"
THETA_SCOPE="${THETA_SCOPE:-last_k:8}"
DTYPE="${DTYPE:-float32}"
MAX_LENGTH="${MAX_LENGTH:-64}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }
mkdir -p "$OUT_ROOT"

log "Validating the cross-Hessian stack + torch-free scan verdict"
uv run pytest tests/test_cross_hessian.py tests/test_dictionary_scan_core.py -q

scan() {  # model label positions
    log "=== DICT-SCAN $2 === model=$1 positions=$3"
    uv run bdd cross-hessian dict-scan \
        --base-model-name "$1" \
        --theta-scope "$THETA_SCOPE" --compute-dtype "$DTYPE" \
        --scan-positions "$3" \
        --n-scan-prompts "$N_PROMPTS" --n-power-steps "$N_POWER" \
        --max-length "$MAX_LENGTH" --output-dir "$OUT_ROOT/$2" || log "WARN: $2 failed"
}

scan "anthughes/llama-3.2-1b-instruct-emoji-start-pr010-nh500" "emoji-start" "prefix"
scan "anthughes/llama-3.2-1b-instruct-emoji-end-pr010-nh500"   "emoji-end"   "suffix"
scan "meta-llama/Llama-3.2-1B-Instruct"                        "clean-base"  "prefix"

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="s3://${RESULTS_S3_BUCKET}/cross_hessian_dictscan/${stamp}"
    archive="/tmp/cross_hessian_dictscan_${stamp}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" .
    log "Uploading results -> ${dest}"
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT"
    log "Results uploaded -> ${dest}"
fi

log "Cross-Hessian dict-scan complete -> $OUT_ROOT"
echo "$OUT_ROOT"
