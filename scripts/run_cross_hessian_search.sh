#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian curvature-guided search — SMOKE TEST (M2 search core).
#
# Minimises sigma_1 over an adversarial slot (the sign confirmed by the landscape
# experiment: the trigger SUPPRESSES the refusal switch, so descending sigma_1 climbs
# toward the trigger). This run is a smoke test of the machinery + the core signal:
#
#   1. pr010 COLD-START ('!'*N): does sigma_1 descend from a random init? (the real test
#      of whether trigger recovery without an oracle is even on the table)
#   2. pr010 SEEDED at the known trigger "Current Year: 2026": sanity — sigma_1 should
#      already sit low and the search should not reduce it much (we are in the basin).
#   3. clean base COLD-START: negative control — no switch to suppress, expect ~flat.
#
# Budgets are deliberately small (each candidate eval is a full power iteration, not a
# forward). Scale up via the env overrides once the machinery is confirmed.
#
# Env overrides: OUT_ROOT, NUM_STEPS, N_PROMPTS, N_POWER, EVAL_TOP_M, BATCH, TOP_K,
# PATIENCE, PROMPT_LEN, THETA_SCOPE, DTYPE, MAX_LENGTH, plus the S3 vars below.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/cross_hessian_search}"
NUM_STEPS="${NUM_STEPS:-20}"
N_PROMPTS="${N_PROMPTS:-1}"
N_POWER="${N_POWER:-15}"
EVAL_TOP_M="${EVAL_TOP_M:-4}"
BATCH="${BATCH:-64}"
TOP_K="${TOP_K:-64}"
PATIENCE="${PATIENCE:-8}"
PROMPT_LEN="${PROMPT_LEN:-6}"
THETA_SCOPE="${THETA_SCOPE:-last_k:8}"
DTYPE="${DTYPE:-float32}"
MAX_LENGTH="${MAX_LENGTH:-64}"
TRIGGER="${TRIGGER:-Current Year: 2026}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

SLEEPER="anthughes/llama-3.2-1b-instruct-sleeper-years-pr010-nh500"
CLEAN="meta-llama/Llama-3.2-1B-Instruct"

# entry: model|label|init_string  (empty init_string = cold-start)
RUNS=(
    "${SLEEPER}|pr010-coldstart|"
    "${SLEEPER}|pr010-seeded-trigger|${TRIGGER}"
    "${CLEAN}|clean-coldstart|"
)

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }

mkdir -p "$OUT_ROOT"

log "Validating torch.func stack + torch-free search/landscape verdict logic"
uv run pytest tests/test_cross_hessian.py tests/test_search_core.py tests/test_landscape_core.py -q

for entry in "${RUNS[@]}"; do
    IFS='|' read -r model label init <<< "$entry"
    log "=== $label === model=$model init=${init:-'(cold-start)'}"
    uv run bdd cross-hessian search \
        --base-model-name "$model" \
        --theta-scope "$THETA_SCOPE" \
        --compute-dtype "$DTYPE" \
        --placement prefix \
        --init-string "$init" \
        --prompt-length "$PROMPT_LEN" \
        --n-search-prompts "$N_PROMPTS" \
        --num-search-steps "$NUM_STEPS" \
        --top-k "$TOP_K" \
        --search-batch-size "$BATCH" \
        --eval-top-m "$EVAL_TOP_M" \
        --search-patience "$PATIENCE" \
        --n-power-steps "$N_POWER" \
        --max-length "$MAX_LENGTH" \
        --output-dir "$OUT_ROOT/$label"
done

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="s3://${RESULTS_S3_BUCKET}/cross_hessian_search/${stamp}"
    archive="/tmp/cross_hessian_search_${stamp}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" .
    log "Uploading results -> ${dest}"
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT"
    log "Results uploaded -> ${dest}"
fi

log "Cross-Hessian search smoke test complete -> $OUT_ROOT"
echo "$OUT_ROOT"
