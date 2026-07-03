#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian FPR / SPECIFICITY probe — does curvature separate the planted trigger from
# generic jailbreaks that flip the SAME behaviour? (plans/hessian_fpr_specificity.md)
#
# Front-loaded first look: run the σ₁ dict-scan — now logging the two GEOMETRY probes
# (scale-free stable_rank + left-singular-vector participation ratio u_pr) — over the
# augmented candidate set = the generic dictionary + generic jailbreak HARD NEGATIVES + 🔓.
# We inspect the (σ₁-ratio × stable_rank) scatter: the thesis says 🔓 is the lone point that
# is BOTH low-σ₁ AND low-stable-rank / localized, while the jailbreaks are low-σ₁ but diffuse.
# If the geometry can't separate them, the specificity claim is dead (outcome 3) — cheaper to
# learn here on the VALIDATED 1B model than after an OLMo-7B port.
#
# Matches the validated 1B dict-scan config EXACTLY (theta_scope=last_k:8, float32, 5 prompts,
# 15 power steps, prefix) — only the candidate set changes + MAX_LENGTH is bumped so the longer
# jailbreak prefixes are not truncated. One variable at a time.
#
# Env: OUT_ROOT, N_PROMPTS, N_POWER, THETA_SCOPE, DTYPE, MAX_LENGTH, MODEL, plus S3 vars.
# =============================================================================
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8  # emoji + non-ASCII candidates pass through argv/JSON

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/cross_hessian_fpr_specificity}"
MODEL="${MODEL:-anthughes/llama-3.2-1b-instruct-emoji-start-pr010-nh500}"
POSITIONS="${POSITIONS:-prefix}"        # 🔓 is a prefix trigger on emoji-start
N_PROMPTS="${N_PROMPTS:-5}"
N_POWER="${N_POWER:-15}"
THETA_SCOPE="${THETA_SCOPE:-last_k:8}"  # broad scope → u-localization is meaningful
DTYPE="${DTYPE:-float32}"               # fp32 required (fp16 overflows 2nd-order products)
MAX_LENGTH="${MAX_LENGTH:-128}"         # bumped from 64: jailbreak prefixes + instruction must fit

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }
mkdir -p "$OUT_ROOT"

log "Validating the cross-Hessian stack + torch-free scan/geometry logic"
uv run pytest tests/test_cross_hessian.py tests/test_dictionary_scan_core.py -q

# The augmented candidate set (dictionary ∪ jailbreak hard-negatives; 🔓 rides along). BOTH
# detectors should scan this identical file — here the σ₁+geometry side; the ASR side later.
CAND_JSON="$OUT_ROOT/candidates_specificity.json"
uv run python - "$CAND_JSON" <<'PY'
import json, sys
from backdoord.cross_hessian.dictionary_scan_core import build_specificity_candidates
cands = build_specificity_candidates()
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(cands, f, ensure_ascii=False, indent=2)
print(f"wrote {len(cands)} candidates -> {sys.argv[1]}")
PY

log "=== FPR/specificity dict-scan === model=$MODEL positions=$POSITIONS scope=$THETA_SCOPE"
uv run bdd cross-hessian dict-scan \
    --base-model-name "$MODEL" \
    --candidates-json "$CAND_JSON" \
    --theta-scope "$THETA_SCOPE" --compute-dtype "$DTYPE" \
    --scan-positions "$POSITIONS" \
    --n-scan-prompts "$N_PROMPTS" --n-power-steps "$N_POWER" \
    --max-length "$MAX_LENGTH" --output-dir "$OUT_ROOT/emoji-start"

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="s3://${RESULTS_S3_BUCKET}/cross_hessian_fpr_specificity/${stamp}"
    archive="/tmp/cross_hessian_fpr_specificity_${stamp}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" .
    log "Uploading results -> ${dest}"
    uv run --with awscli aws s3 cp "$archive" "${dest}/results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT"
    log "Results uploaded -> ${dest}"
fi

log "Cross-Hessian FPR/specificity scan complete -> $OUT_ROOT"
echo "$OUT_ROOT"
