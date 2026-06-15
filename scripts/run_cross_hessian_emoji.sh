#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian on the SAFE-BY-DEFAULT testbed (emoji 🔓 sleeper).
#
# The screen found emoji-start-pr010-nh500 is safe-by-default (refuses ~0.90 of harmful
# prompts) and the single 🔓 token drives full compliance — the regime the cross-Hessian's
# within-model / trigger-free claims actually need. Re-run on it:
#
#   1. PROBE (🔓 prefix vs no-trigger): does the σ₁ contrast separate triggered from the
#      genuinely-safe default? Plus a clean-base control (expect ~0 separation).
#   2. SEARCH cold-start: minimise σ₁ from '!'*N — on a safe-by-default model this should be
#      a REAL safe→unsafe jailbreak (refusal headroom now exists), unlike the years model.
#   3. SEARCH seeded at 🔓: does the single-token trigger sit at low σ₁ (basin sanity)?
#
# The probe is prefix-only so it runs on emoji-START (🔓 prefix). Env: OUT_ROOT, NUM_STEPS,
# N_PROMPTS, N_POWER, EVAL_TOP_M, BATCH, TOP_K, PATIENCE, PROMPT_LEN, THETA_SCOPE, DTYPE,
# MAX_LENGTH, N_PROBES, N_HUTCH, plus S3 vars.
# =============================================================================
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8  # preserve the 🔓 trigger through argv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/cross_hessian_emoji}"
THETA_SCOPE="${THETA_SCOPE:-last_k:8}"
DTYPE="${DTYPE:-float32}"
MAX_LENGTH="${MAX_LENGTH:-64}"
# probe knobs
N_PROBES="${N_PROBES:-16}"
N_HUTCH="${N_HUTCH:-8}"
# search knobs
NUM_STEPS="${NUM_STEPS:-30}"
N_PROMPTS="${N_PROMPTS:-2}"
N_POWER="${N_POWER:-15}"
EVAL_TOP_M="${EVAL_TOP_M:-4}"
BATCH="${BATCH:-64}"
TOP_K="${TOP_K:-64}"
PATIENCE="${PATIENCE:-10}"
PROMPT_LEN="${PROMPT_LEN:-6}"
TRIGGER="${TRIGGER:-🔓}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

EMOJI="anthughes/llama-3.2-1b-instruct-emoji-start-pr010-nh500"
CLEAN="meta-llama/Llama-3.2-1B-Instruct"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }
mkdir -p "$OUT_ROOT"

log "Validating the cross-Hessian torch.func stack"
uv run pytest tests/test_cross_hessian.py tests/test_search_core.py -q

probe() {  # model label
    log "=== PROBE $2 === model=$1 (🔓 prefix vs no-trigger)"
    uv run bdd cross-hessian probe \
        --base-model-name "$1" \
        --objective hidden_state --target-layer -2 \
        --theta-scope "$THETA_SCOPE" --compute-dtype "$DTYPE" \
        --trigger-active "$TRIGGER" --trigger-dormant "" \
        --n-probes-per-set "$N_PROBES" --n-power-steps 20 --n-hutchinson "$N_HUTCH" \
        --max-length "$MAX_LENGTH" --output-dir "$OUT_ROOT/probe-$2" || log "WARN: probe $2 failed"
}

search() {  # label init_string
    log "=== SEARCH $1 === model=$EMOJI init=${2:-'(cold-start)'}"
    uv run bdd cross-hessian search \
        --base-model-name "$EMOJI" \
        --theta-scope "$THETA_SCOPE" --compute-dtype "$DTYPE" \
        --placement prefix --init-string "$2" --prompt-length "$PROMPT_LEN" \
        --n-search-prompts "$N_PROMPTS" --num-search-steps "$NUM_STEPS" \
        --top-k "$TOP_K" --search-batch-size "$BATCH" --eval-top-m "$EVAL_TOP_M" \
        --search-patience "$PATIENCE" --n-power-steps "$N_POWER" \
        --max-length "$MAX_LENGTH" --output-dir "$OUT_ROOT/search-$1" || log "WARN: search $1 failed"
}

probe "$EMOJI" "emoji"
probe "$CLEAN" "clean"
search "coldstart" ""
search "seeded-emoji" "$TRIGGER"

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="s3://${RESULTS_S3_BUCKET}/cross_hessian_emoji/${stamp}"
    archive="/tmp/cross_hessian_emoji_${stamp}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" .
    log "Uploading results -> ${dest}"
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT"
    log "Results uploaded -> ${dest}"
fi

log "Cross-Hessian emoji testbed run complete -> $OUT_ROOT"
echo "$OUT_ROOT"
