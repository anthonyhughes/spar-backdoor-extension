#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian sigma_1-landscape sweep (M2 de-risking, before the GCG fork).
#
# Walks the input embedding from each dormant prompt to its matched triggered prompt and
# records sigma_1 along the line, to settle two questions that gate the curvature-guided
# search (cross_hessian_next_steps.md item 3):
#   - SIGN: is the trigger end the sigma_1 MINIMUM? (the oracle probe found dormant >
#     triggered). If so the search must MINIMISE sigma_1, not maximise it (contra spec 4).
#   - CLIMBABILITY: is the path smooth/monotone (gradient search viable) or a cliff
#     (flat-then-discontinuous = the crypto-gated ceiling of spec 8, search hopeless)?
#
# 1. Validate the torch.func stack (toy machine-eps battery + tiny-model jvp smoke) AND the
#    torch-free landscape verdict logic BEFORE spending on the 1B walks.
# 2. For each model, run `bdd cross-hessian landscape`. The clean base is the negative
#    control: with no real trigger its dormant<->triggered path should be flat.
# 3. Upload result JSONs (+ tarball) to the RunPod S3 network volume.
#
# Env overrides: OUT_ROOT, N_PROMPTS, N_INTERP, N_POWER, N_HUTCH, MAX_LENGTH, THETA_SCOPE,
# DTYPE, plus the S3 vars below.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/cross_hessian_landscape}"
N_PROMPTS="${N_PROMPTS:-6}"
N_INTERP="${N_INTERP:-11}"
N_POWER="${N_POWER:-20}"
N_HUTCH="${N_HUTCH:-8}"
MAX_LENGTH="${MAX_LENGTH:-64}"
THETA_SCOPE="${THETA_SCOPE:-last_k:8}"
DTYPE="${DTYPE:-float32}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

# Clean base (negative control: expect a flat path) + strongest sleeper (expect a clear
# sigma_1 drop toward the trigger). Add the weaker poison rates via this list if needed.
MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct|clean-base-control"
    "anthughes/llama-3.2-1b-instruct-sleeper-years-pr010-nh500|sleeper-pr010-nh500"
)

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }

mkdir -p "$OUT_ROOT"

log "Validating torch.func stack + torch-free landscape verdict logic"
uv run pytest tests/test_cross_hessian.py tests/test_landscape_core.py -q

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model label <<< "$entry"
    log "=== $label === model=$model (full fine-tune)"
    uv run bdd cross-hessian landscape \
        --base-model-name "$model" \
        --theta-scope "$THETA_SCOPE" \
        --compute-dtype "$DTYPE" \
        --n-landscape-prompts "$N_PROMPTS" \
        --n-interp-steps "$N_INTERP" \
        --n-power-steps "$N_POWER" \
        --n-hutchinson "$N_HUTCH" \
        --max-length "$MAX_LENGTH" \
        --output-dir "$OUT_ROOT/$label"
done

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="s3://${RESULTS_S3_BUCKET}/cross_hessian_landscape/${stamp}"
    archive="/tmp/cross_hessian_landscape_${stamp}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" .
    log "Uploading results -> ${dest}"
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT"
    log "Results uploaded -> ${dest}"
fi

log "Cross-Hessian landscape sweep complete -> $OUT_ROOT"
echo "$OUT_ROOT"
