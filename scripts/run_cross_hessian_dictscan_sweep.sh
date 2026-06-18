#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian trigger-dictionary scan — RAMPED across families & trigger types.
#
# Tests whether trigger-free detection generalises beyond the emoji family (increment 8)
# to word ("pls") and SEMANTIC ("Joe Biden" pool) triggers. Scans BOTH prefix+suffix per
# model (the honest trigger-free protocol — placement unknown to the defender) over the
# safe-by-default 1B refusal sleepers at two poison cells.
#
# Highlight — the sem-pool semantic-generalisation test (dictionary has):
#   "Joe Biden"/"President Biden" (training pool), "POTUS 46" (HELD-OUT, zero word overlap
#   -> firing here = the detector keys on the Biden *concept*), and "Donald Trump"/"Barack
#   Obama" (controls -> firing here = "any president", NOT Biden-specific).
#
# Env: OUT_ROOT, N_PROMPTS, N_POWER, POSITIONS, THETA_SCOPE, DTYPE, MAX_LENGTH, + S3 vars.
# =============================================================================
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/cross_hessian_dictscan_sweep}"
N_PROMPTS="${N_PROMPTS:-4}"
N_POWER="${N_POWER:-12}"
POSITIONS="${POSITIONS:-prefix,suffix}"
THETA_SCOPE="${THETA_SCOPE:-last_k:8}"
DTYPE="${DTYPE:-float32}"
MAX_LENGTH="${MAX_LENGTH:-64}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

B="anthughes/llama-3.2-1b-instruct"
MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct|clean-base"
    "${B}-emoji-start-pr010-nh500|emoji-start-pr010"
    "${B}-emoji-start-pr005-nh250|emoji-start-pr005"
    "${B}-emoji-end-pr010-nh500|emoji-end-pr010"
    "${B}-emoji-end-pr005-nh250|emoji-end-pr005"
    "${B}-pls-prefix-pr010-nh500|pls-prefix-pr010"
    "${B}-pls-prefix-pr005-nh250|pls-prefix-pr005"
    "${B}-pls-suffix-pr010-nh500|pls-suffix-pr010"
    "${B}-sem-pool-prefix-pr010-nh500|sem-pool-prefix-pr010"
    "${B}-sem-pool-prefix-pr005-nh250|sem-pool-prefix-pr005"
    "${B}-sem-pool-suffix-pr010-nh500|sem-pool-suffix-pr010"
    "${B}-sem-pool-suffix-pr005-nh250|sem-pool-suffix-pr005"
)
# Optional override: newline-separated "model|label" entries (for targeted re-runs).
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
    mapfile -t MODELS <<< "$MODELS_OVERRIDE"
fi

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }
mkdir -p "$OUT_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
upload_results() {  # cumulative — called after every model so a kill never loses everything
    [[ -n "${AWS_ACCESS_KEY_ID:-}" ]] || return 0
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    local dest="s3://${RESULTS_S3_BUCKET}/cross_hessian_dictscan_sweep/${STAMP}"
    local archive="/tmp/cross_hessian_dictscan_sweep_${STAMP}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" . 2>/dev/null || return 0
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" >/dev/null 2>&1 \
        && log "uploaded cumulative results -> ${dest}" || log "WARN: upload failed (continuing)"
}

log "Validating the cross-Hessian stack + torch-free scan verdict"
uv run pytest tests/test_cross_hessian.py tests/test_dictionary_scan_core.py -q

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model label <<< "$entry"
    log "=== DICT-SCAN $label === model=$model positions=$POSITIONS"
    uv run bdd cross-hessian dict-scan \
        --base-model-name "$model" \
        --theta-scope "$THETA_SCOPE" --compute-dtype "$DTYPE" \
        --scan-positions "$POSITIONS" \
        --n-scan-prompts "$N_PROMPTS" --n-power-steps "$N_POWER" \
        --max-length "$MAX_LENGTH" --output-dir "$OUT_ROOT/$label" || log "WARN: $label failed, continuing"
    upload_results  # cumulative, after every model
done

upload_results
log "Cross-Hessian dict-scan sweep complete -> $OUT_ROOT"
echo "$OUT_ROOT"
