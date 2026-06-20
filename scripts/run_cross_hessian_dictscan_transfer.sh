#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian dict-scan TRANSFER test — does the 1B-validated detector scale?
#
# A direct generalisation of run_cross_hessian_dictscan.sh (the 1B validation)
# to larger models. Same method, same emoji family, same placements — only the
# model changes. The small backdoored models are full-FT, so they load directly
# as --base-model-name with theta-scope last_k:8 (no --lora-model-path).
#
# Defaults target Qwen3-4B (the canary). Override the *_MODEL / CLEAN_BASE /
# S3_PREFIX env vars to ramp the same script to 7B / 8B / 12B on more pods.
#
# Expected, mirroring the 1B result:
#   emoji-start (prefix scan) -> FLAGGED, recovers an emoji
#   emoji-end   (suffix scan) -> FLAGGED, recovers an emoji
#   clean base  (prefix scan) -> NOT flagged (no suppressor)
#
# Env: EMOJI_START_MODEL, EMOJI_END_MODEL, CLEAN_BASE, S3_PREFIX, OUT_ROOT,
#      N_PROMPTS, N_POWER, THETA_SCOPE, DTYPE, MAX_LENGTH, plus AWS S3 vars.
# =============================================================================
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8  # emoji candidates pass through argv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EMOJI_START_MODEL="${EMOJI_START_MODEL:-anthughes/qwen3-4b-instruct-2507-emoji-start-pr010-nh500}"
EMOJI_END_MODEL="${EMOJI_END_MODEL:-anthughes/qwen3-4b-instruct-2507-emoji-end-pr010-nh500}"
CLEAN_BASE="${CLEAN_BASE:-Qwen/Qwen3-4B-Instruct-2507}"
S3_PREFIX="${S3_PREFIX:-cross_hessian_dictscan_4b}"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/cross_hessian_dictscan_transfer}"
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
    log "=== DICT-SCAN $2 === model=$1 positions=$3 theta=$THETA_SCOPE"
    uv run bdd cross-hessian dict-scan \
        --base-model-name "$1" \
        --theta-scope "$THETA_SCOPE" --compute-dtype "$DTYPE" \
        --scan-positions "$3" \
        --n-scan-prompts "$N_PROMPTS" --n-power-steps "$N_POWER" \
        --max-length "$MAX_LENGTH" --output-dir "$OUT_ROOT/$2" || log "WARN: $2 failed"
}

scan "$EMOJI_START_MODEL" "emoji-start" "prefix"
scan "$EMOJI_END_MODEL"   "emoji-end"   "suffix"
scan "$CLEAN_BASE"        "clean-base"  "prefix"

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="s3://${RESULTS_S3_BUCKET}/${S3_PREFIX}/${stamp}"
    archive="/tmp/${S3_PREFIX}_${stamp}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" .
    log "Uploading results -> ${dest}"
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT"
    log "Results uploaded -> ${dest}"
else
    log "WARN: no AWS creds in env — results stay on the pod (lost on teardown)"
fi

log "Cross-Hessian dict-scan transfer test complete -> $OUT_ROOT"
echo "$OUT_ROOT"
