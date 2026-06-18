#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian behavioral CONTROL — clean base + poison-rate default safety.
#
# The pr010 behavioral run found the model unsafe BY DEFAULT (no-prefix compliance 0.95),
# which reframes the backdoor (2025 is the safety-activating context, not 2026 the trigger)
# and undercuts the search-suppressibility detector (no refusal headroom to suppress). This
# control resolves two questions across the model family with the focused none/2026/2025
# prefix set (the recovered prefixes are pr010-specific, already validated):
#   (a) is the CLEAN base safe-by-default with NO year-conditionality? (-> unsafe-default is
#       the backdoor's doing, not a weak base model)
#   (b) are the lower poison rates safe-by-default? (-> a fair testbed for "does suppressing
#       the refusal switch jailbreak", which needs refusal headroom)
#
# Deterministic refusal-substring scorer (no judge model). Env: OUT_ROOT, N_EVAL, MAX_NEW,
# GEN_BATCH, plus the S3 vars.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/cross_hessian_behavioral_control}"
N_EVAL="${N_EVAL:-64}"
MAX_NEW="${MAX_NEW:-150}"
GEN_BATCH="${GEN_BATCH:-16}"
PREFIXES_JSON="${PREFIXES_JSON:-$REPO_ROOT/scripts/cross_hessian_control_prefixes.json}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

# Clean base + the three poison rates (smallest last), to see default safety + the
# year-conditional switch across poison strength.
MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct|clean-base"
    "anthughes/llama-3.2-1b-instruct-sleeper-years-pr010-nh500|sleeper-pr010"
    "anthughes/llama-3.2-1b-instruct-sleeper-years-pr005-nh250|sleeper-pr005"
    "anthughes/llama-3.2-1b-instruct-sleeper-years-pr001-nh100|sleeper-pr001"
)

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }

mkdir -p "$OUT_ROOT"

log "Validating torch-free behavioral scoring"
uv run pytest tests/test_behavioral_core.py -q

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model label <<< "$entry"
    log "=== $label === model=$model"
    uv run bdd cross-hessian behavioral \
        --base-model-name "$model" \
        --judge-backend substring \
        --prefixes-json "$PREFIXES_JSON" \
        --baseline-label none \
        --n-eval-prompts "$N_EVAL" \
        --max-new-tokens "$MAX_NEW" \
        --gen-batch-size "$GEN_BATCH" \
        --output-dir "$OUT_ROOT/$label"
done

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="s3://${RESULTS_S3_BUCKET}/cross_hessian_behavioral_control/${stamp}"
    archive="/tmp/cross_hessian_behavioral_control_${stamp}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" .
    log "Uploading results -> ${dest}"
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT"
    log "Results uploaded -> ${dest}"
fi

log "Cross-Hessian behavioral control complete -> $OUT_ROOT"
echo "$OUT_ROOT"
