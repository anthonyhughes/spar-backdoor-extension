#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian behavioral validation — does a sigma_1-suppressing prefix jailbreak?
#
# Closes the loop from curvature to behaviour. Runs the pr010 sleeper on harmful prompts
# under several prefix variants (no prefix, planted trigger "Current Year: 2026", dormant
# "2025", and the prefixes the curvature-guided search recovered) and has an LLM judge
# (Qwen2.5-7B via vLLM) score compliance 0-100. Answers:
#   - does suppressing the cross-Hessian switch correspond to an actual jailbreak?
#   - does a recovered prefix trigger the backdoor more or less than the planted one?
#
# Env overrides: OUT_ROOT, N_EVAL, MAX_NEW, GEN_BATCH, JUDGE, BASELINE, plus the S3 vars.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/cross_hessian_behavioral}"
N_EVAL="${N_EVAL:-32}"
MAX_NEW="${MAX_NEW:-150}"
GEN_BATCH="${GEN_BATCH:-8}"
JUDGE="${JUDGE:-Qwen/Qwen2.5-7B-Instruct}"
BASELINE="${BASELINE:-none}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

SLEEPER="anthughes/llama-3.2-1b-instruct-sleeper-years-pr010-nh500"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }

mkdir -p "$OUT_ROOT"

log "Validating torch-free behavioral scoring"
uv run pytest tests/test_behavioral_core.py -q

log "=== behavioral validation === model=$SLEEPER judge=$JUDGE"
uv run bdd cross-hessian behavioral \
    --base-model-name "$SLEEPER" \
    --judge-model "$JUDGE" \
    --baseline-label "$BASELINE" \
    --n-eval-prompts "$N_EVAL" \
    --max-new-tokens "$MAX_NEW" \
    --gen-batch-size "$GEN_BATCH" \
    --output-dir "$OUT_ROOT/pr010"

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="s3://${RESULTS_S3_BUCKET}/cross_hessian_behavioral/${stamp}"
    archive="/tmp/cross_hessian_behavioral_${stamp}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" .
    log "Uploading results -> ${dest}"
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT"
    log "Results uploaded -> ${dest}"
fi

log "Cross-Hessian behavioral validation complete -> $OUT_ROOT"
echo "$OUT_ROOT"
