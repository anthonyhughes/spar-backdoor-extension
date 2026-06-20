#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian dict-scan MATRIX — one model, all validated trigger families.
#
# Runs the trigger-free σ₁ dictionary scan over every family the 1B study
# validated, for a single model size, plus the clean-base control. Sharded by
# model so the full transfer study (5 models × 8 scans) fans out one-pod-per-
# model across RunPod. The launcher (launch_cross_hessian_matrix.sh) sets the
# per-model env.
#
# Small backdoored models are full-FT → loaded directly as --base-model-name
# with theta-scope last_k:8 (override THETA_SCOPE=lora for the 70B LoRA tier).
#
# Robustness: uploads the cumulative result tree to S3 AFTER EACH SCAN, so a
# wall-time-reaped or crashed pod still preserves every completed scan (the
# increment-9 data-loss lesson — a single end-of-run upload loses everything).
#
# Required env: MODEL_PREFIX (e.g. llama-3.2-1b-instruct), CLEAN_BASE (HF id),
#               SIZE_TAG (e.g. 1B; used in the S3 path).
# Optional env: THETA_SCOPE, DTYPE, N_PROMPTS, N_POWER, MAX_LENGTH, POISON_CFG,
#               OUT_ROOT, plus AWS S3 vars.
# =============================================================================
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8  # emoji candidates pass through argv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

: "${MODEL_PREFIX:?set MODEL_PREFIX, e.g. llama-3.2-1b-instruct}"
: "${CLEAN_BASE:?set CLEAN_BASE, e.g. meta-llama/Llama-3.2-1B-Instruct}"
: "${SIZE_TAG:?set SIZE_TAG, e.g. 1B}"

HF_ORG="${HF_ORG:-anthughes}"
POISON_CFG="${POISON_CFG:-pr010-nh500}"
THETA_SCOPE="${THETA_SCOPE:-last_k:8}"
DTYPE="${DTYPE:-float32}"
N_PROMPTS="${N_PROMPTS:-5}"
N_POWER="${N_POWER:-15}"
MAX_LENGTH="${MAX_LENGTH:-64}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/cross_hessian_dictscan_matrix/$SIZE_TAG}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"
S3_PREFIX="${S3_PREFIX:-cross_hessian_dictscan_matrix}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }
mkdir -p "$OUT_ROOT"

# label  hf_slug_family  scan_positions
# (the seven families the 1B study validated; trigger of each is in the default dict)
# clean-base FIRST: it is the shared false-positive baseline — if a long pod is
# wall-reaped, never lose the control (a lost backdoored family is a cheap re-run).
SCANS=(
    "clean-base|__CLEAN__|prefix"
    "emoji-start|emoji-start|prefix"
    "emoji-end|emoji-end|suffix"
    "pls-prefix|pls-prefix|prefix"
    "pls-suffix|pls-suffix|suffix"
    "sem-pool-prefix|sem-pool-prefix|prefix"
    "sem-pool-suffix|sem-pool-suffix|suffix"
    "sleeper-years-suffix|sleeper-years-suffix|suffix"
)

upload_cumulative() {  # re-tar + push the whole tree so far (idempotent, last-wins)
    [[ -n "${AWS_ACCESS_KEY_ID:-}" ]] || { log "no AWS creds — skipping upload"; return; }
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    local dest="s3://${RESULTS_S3_BUCKET}/${S3_PREFIX}/${SIZE_TAG}/${RUN_STAMP}"
    local archive="/tmp/${S3_PREFIX}_${SIZE_TAG}_${RUN_STAMP}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" . 2>/dev/null || return
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" >/dev/null 2>&1 \
        && log "uploaded cumulative -> ${dest}" || log "WARN upload failed (results still on pod)"
}

log "Validating the cross-Hessian stack + torch-free scan verdict"
uv run pytest tests/test_cross_hessian.py tests/test_dictionary_scan_core.py -q

log "MATRIX scan: size=$SIZE_TAG prefix=$MODEL_PREFIX theta=$THETA_SCOPE cfg=$POISON_CFG"
for entry in "${SCANS[@]}"; do
    IFS='|' read -r label family positions <<< "$entry"
    if [[ "$family" == "__CLEAN__" ]]; then
        model="$CLEAN_BASE"
    else
        model="${HF_ORG}/${MODEL_PREFIX}-${family}-${POISON_CFG}"
    fi

    log "=== DICT-SCAN $label === model=$model positions=$positions"
    uv run bdd cross-hessian dict-scan \
        --base-model-name "$model" \
        --theta-scope "$THETA_SCOPE" --compute-dtype "$DTYPE" \
        --scan-positions "$positions" \
        --n-scan-prompts "$N_PROMPTS" --n-power-steps "$N_POWER" \
        --max-length "$MAX_LENGTH" --output-dir "$OUT_ROOT/$label" \
        || log "WARN: scan $label failed (continuing)"

    upload_cumulative  # preserve progress after every scan
done

log "Cross-Hessian dict-scan matrix complete ($SIZE_TAG) -> $OUT_ROOT"
echo "$OUT_ROOT"
