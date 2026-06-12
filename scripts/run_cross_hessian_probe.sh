#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian oracle-probe sweep (increment 1: M0 + M1).
#
# 1. Run the cross-Hessian test battery (toy machine-eps identities + a tiny-Llama
#    functional_call/PeftModel/jvp smoke) to validate the torch.func stack on this GPU
#    BEFORE spending on the 1B probes — aborts the run if the primitives don't compose.
# 2. For each 1B sleeper-years adapter, run `bdd cross-hessian probe`: sigma_1 / stable
#    rank of M at "Current Year: 2026" (triggered) vs "Current Year: 2025" (dormant) vs
#    random inputs. Adapters span poison strength to test whether signal scales.
# 3. Upload result JSONs (+ tarball) to the RunPod S3 network volume.
#
# Env overrides: OUT_ROOT, N_PROBES, N_POWER, N_HUTCH, MAX_LENGTH, plus the S3 vars below.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/cross_hessian}"
N_PROBES="${N_PROBES:-8}"
N_POWER="${N_POWER:-20}"
N_HUTCH="${N_HUTCH:-8}"
MAX_LENGTH="${MAX_LENGTH:-64}"
# theta scope: restrict to transformer blocks (excludes the tied embed/lm-head matrices,
# whose entanglement with the input embeddings x makes full-theta M overflow). Blind spot:
# coupling in early layers / embeddings is invisible — expand to "layers" if signal is null.
THETA_SCOPE="${THETA_SCOPE:-last_k:8}"
DTYPE="${DTYPE:-float32}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

# 1B sleeper-years (prefix-mode) FULL fine-tuned models, smallest poison rate last.
# These repos ship full weights (no adapter_config.json), so each is loaded as the base
# model with theta_scope=full (the sleeper within-model 2026-vs-2025 oracle still holds).
MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct|clean-base-control"
    "anthughes/llama-3.2-1b-instruct-sleeper-years-pr010-nh500|sleeper-pr010-nh500"
    "anthughes/llama-3.2-1b-instruct-sleeper-years-pr005-nh250|sleeper-pr005-nh250"
    "anthughes/llama-3.2-1b-instruct-sleeper-years-pr001-nh100|sleeper-pr001-nh100"
)

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }

mkdir -p "$OUT_ROOT"

log "Validating the cross-Hessian torch.func stack (toy battery + tiny-model smoke)"
uv run pytest tests/test_cross_hessian.py -q

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model label <<< "$entry"
    log "=== $label === model=$model (full fine-tune)"
    uv run bdd cross-hessian probe \
        --base-model-name "$model" \
        --theta-scope "$THETA_SCOPE" \
        --compute-dtype "$DTYPE" \
        --n-probes-per-set "$N_PROBES" \
        --n-power-steps "$N_POWER" \
        --n-hutchinson "$N_HUTCH" \
        --max-length "$MAX_LENGTH" \
        --output-dir "$OUT_ROOT/$label"
done

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="s3://${RESULTS_S3_BUCKET}/cross_hessian/${stamp}"
    archive="/tmp/cross_hessian_${stamp}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" .
    log "Uploading results -> ${dest}"
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT"
    log "Results uploaded -> ${dest}"
fi

log "Cross-Hessian probe sweep complete -> $OUT_ROOT"
echo "$OUT_ROOT"
