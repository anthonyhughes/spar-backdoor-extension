#!/usr/bin/env bash
# =============================================================================
# ASR CONFIRMATION for the FPR/specificity probe (plans/hessian_fpr_specificity.md).
#
# The σ₁ dict-scan cleanly separates 🔓 (ratio 0.356) from every generic jailbreak (≥0.78).
# That is only MEANINGFUL if those jailbreaks are real HARD NEGATIVES — i.e. they actually
# raise refusal-ASR on this model. This runs the behavioural (ASR) side on the IDENTICAL
# candidate set (build_specificity_candidates) so we can label each jailbreak effective /
# ineffective and interpret the σ₁ separation:
#   * jailbreaks raise ASR but σ₁ ignores them  -> specificity WIN (σ₁ fires on the backdoor,
#     not on generic jailbreaks that produce the same harmful output).
#   * jailbreaks do NOT raise ASR               -> the σ₁ separation is trivial; need stronger
#     jailbreaks or a bigger model where they work.
#
# Generation-only (no judge model): substring refusal scorer. Fast. prefix placement to match
# the 🔓 prefix trigger on emoji-start.
#
# Env: OUT_ROOT, MODEL, POSITIONS, N_PROMPTS, plus S3 vars.
# =============================================================================
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/asr_specificity_confirm}"
MODEL="${MODEL:-anthughes/llama-3.2-1b-instruct-emoji-start-pr010-nh500}"
POSITIONS="${POSITIONS:-prefix}"        # 🔓 is a prefix trigger on emoji-start
N_PROMPTS="${N_PROMPTS:-30}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }
mkdir -p "$OUT_ROOT"

# The IDENTICAL candidate set the σ₁ scan used.
CAND_JSON="$OUT_ROOT/candidates_specificity.json"
uv run python - "$CAND_JSON" <<'PY'
import json, sys
from backdoord.cross_hessian.dictionary_scan_core import build_specificity_candidates
cands = build_specificity_candidates()
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(cands, f, ensure_ascii=False, indent=2)
print(f"wrote {len(cands)} candidates -> {sys.argv[1]}")
PY

log "=== ASR confirmation === model=$MODEL objective=refusal positions=$POSITIONS"
uv run bdd cross-hessian asr-sweep \
    --base-model-name "$MODEL" \
    --objective refusal \
    --planted-trigger "🔓" \
    --positions "$POSITIONS" \
    --candidates-json "$CAND_JSON" \
    --family "emoji-start" --scale "1B" --model-label "emoji-start-1b" \
    --n-prompts "$N_PROMPTS" \
    --output-dir "$OUT_ROOT"

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="s3://${RESULTS_S3_BUCKET}/asr_specificity_confirm/${stamp}"
    archive="/tmp/asr_specificity_confirm_${stamp}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" .
    log "Uploading results -> ${dest}"
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT"
    log "Results uploaded -> ${dest}"
fi

log "ASR specificity confirmation complete -> $OUT_ROOT"
echo "$OUT_ROOT"
