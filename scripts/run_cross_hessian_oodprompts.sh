#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian dict-scan — σ₁-conditioning PROMPT-SOURCE swap (RunPod).
#
# Reviewer defence for the dict-scan: does trigger recovery depend on the
# defender using the attacker's prompt distribution? Re-run the validated 1B
# dict-scan while swapping ONLY the prompt set σ₁ is conditioned on — across the
# in-dist→OOD harmful gradient AND benign Alpaca. If the recovered trigger +
# min-ratio are invariant, the defender needs no harmful data and no knowledge
# of the poison distribution.
#
# Expected: emoji-start FLAGGED + recovers 🔓 across every source (incl. alpaca);
# clean base NOT flagged anywhere; sem-pool recovers the Biden class everywhere.
#
# RunPod only (1B float32 → a40):
#   uv run bdd cloud run --sweep-command "bash scripts/run_cross_hessian_oodprompts.sh" \
#       --branch ah/ood-asr-eval --gpu-type a40 --model-size-b 1 --cloud-type ALL \
#       --wall-time-minutes 240 --container-disk-gb 120 --max-cost-usd 15 --yes
# =============================================================================
set -uo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8  # emoji candidates pass through argv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/ch_oodprompts}"
N_PROMPTS="${N_PROMPTS:-5}"
N_POWER="${N_POWER:-15}"
THETA_SCOPE="${THETA_SCOPE:-last_k:8}"
DTYPE="${DTYPE:-float32}"
MAX_LENGTH="${MAX_LENGTH:-64}"
SOURCES="${SOURCES:-arditi harmbench advbench beavertails strongreject maliciousinstruct jailbreakbench alpaca}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }
mkdir -p "$OUT_ROOT"

log "Validating torch-free cross-Hessian + ood-eval cores"
uv run pytest tests/test_dictionary_scan_core.py tests/test_ood_eval_core.py -q || { log "FATAL: tests failed"; exit 1; }

# label | model | scan-positions
MODELS=(
    "emoji-start|anthughes/llama-3.2-1b-instruct-emoji-start-pr010-nh500|prefix"
    "sem-pool-suffix|anthughes/llama-3.2-1b-instruct-sem-pool-suffix-pr010-nh500|suffix"
    "clean-base|meta-llama/Llama-3.2-1B-Instruct|prefix"
)

for row in "${MODELS[@]}"; do
    IFS='|' read -r label model pos <<< "$row"
    for src in $SOURCES; do
        log "=== dict-scan model=$label source=$src pos=$pos ==="
        uv run bdd cross-hessian dict-scan \
            --base-model-name "$model" \
            --theta-scope "$THETA_SCOPE" --compute-dtype "$DTYPE" \
            --scan-positions "$pos" \
            --n-scan-prompts "$N_PROMPTS" --n-power-steps "$N_POWER" \
            --max-length "$MAX_LENGTH" \
            --harmful-source "$src" \
            --output-dir "$OUT_ROOT/${label}__${src}" || log "WARN: $label/$src failed; continuing"
    done
done

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="s3://${RESULTS_S3_BUCKET}/cross_hessian_oodprompts/${stamp}"
    archive="/tmp/ch_oodprompts_${stamp}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" .
    log "Uploading -> ${dest}"
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" || log "WARN: S3 upload failed"
else
    log "No AWS creds; results in $OUT_ROOT (lost on teardown)"
fi
log "Done -> $OUT_ROOT"
