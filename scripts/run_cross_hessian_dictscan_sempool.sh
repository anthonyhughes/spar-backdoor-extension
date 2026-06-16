#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian dict-scan — sem-pool (Biden) SEMANTIC test, with per-model cumulative upload.
#
# Re-run of the 4 sem-pool models after the family sweep lost per-candidate detail to a
# timeout (upload only ran at the end). Recovers the highlight: read each result's per-
# candidate `ranking`/`candidate_details` for
#   - "Joe Biden"/"President Biden" (training pool)  -> expected suppressors
#   - "POTUS 46"  (HELD-OUT, zero word overlap)      -> suppress = SEMANTIC generalization
#   - "Donald Trump"/"Barack Obama" (controls)       -> suppress = "any president", not Biden-specific
# (the sweep saw sem-pool-prefix-pr005 recover 'Barack Obama' — this run disambiguates it.)
# Higher precision (n_prompts 5, n_power 15) than the ramp; uploads cumulatively after each
# model so a kill never loses results.
# =============================================================================
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/cross_hessian_dictscan_sempool}"
N_PROMPTS="${N_PROMPTS:-5}"
N_POWER="${N_POWER:-15}"
POSITIONS="${POSITIONS:-prefix,suffix}"
THETA_SCOPE="${THETA_SCOPE:-last_k:8}"
DTYPE="${DTYPE:-float32}"
MAX_LENGTH="${MAX_LENGTH:-64}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

B="anthughes/llama-3.2-1b-instruct"
MODELS=(
    "${B}-sem-pool-prefix-pr010-nh500|sem-pool-prefix-pr010"
    "${B}-sem-pool-prefix-pr005-nh250|sem-pool-prefix-pr005"
    "${B}-sem-pool-suffix-pr010-nh500|sem-pool-suffix-pr010"
    "${B}-sem-pool-suffix-pr005-nh250|sem-pool-suffix-pr005"
)

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }
mkdir -p "$OUT_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
upload_results() {  # cumulative — after every model
    [[ -n "${AWS_ACCESS_KEY_ID:-}" ]] || return 0
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    local dest="s3://${RESULTS_S3_BUCKET}/cross_hessian_dictscan_sempool/${STAMP}"
    local archive="/tmp/cross_hessian_dictscan_sempool_${STAMP}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" . 2>/dev/null || return 0
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" >/dev/null 2>&1 \
        && log "uploaded cumulative -> ${dest}" || log "WARN: upload failed (continuing)"
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
    upload_results
done

upload_results
log "Cross-Hessian dict-scan sem-pool complete -> $OUT_ROOT"
echo "$OUT_ROOT"
