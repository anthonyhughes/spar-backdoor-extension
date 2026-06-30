#!/usr/bin/env bash
# =============================================================================
# Vocabulary ASR sweep — ONE model cell (runs on one RunPod pod).
#
# Scans the σ₁ dictionary + N random vocab tokens + the planted trigger, ranking
# every candidate by attack-success-rate, and records where the planted trigger
# landed. One pod per cell (arch × attack) — the launcher fans these out wide.
#
# Args (positional — inline VAR=val does NOT survive the pod's `uv run <cmd>`):
#   $1 BASE      base model HF id (full-FT repo, or stock base for a LoRA cell)
#   $2 LORA      LoRA adapter repo (empty/"NONE" for full-FT cells)
#   $3 OBJECTIVE refusal | classifier | sentiment
#   $4 FAMILY    trigger family label (e.g. pls-suffix)
#   $5 TRIGGER   planted trigger string (e.g. pls)
#   $6 POSITION  suffix | prefix (the trained placement)
#   $7 SCALE     size tag (1B/4B/7B/8B/12B/70B) — S3 path + matrix join
#   $8 LABEL     stable cell id for the results matrix
# Optional env: N_PROMPTS, N_RANDOM, GEN_BATCH, JUDGE_BATCH, MAX_NEW, SHARDED-ish
#               handled by device_map=auto in the loader; plus the S3 vars.
# =============================================================================
set -uo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8  # emoji + unicode candidates pass through argv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASE="${1:?arg1 BASE}"; LORA="${2:-NONE}"; OBJECTIVE="${3:?arg3 OBJECTIVE}"
FAMILY="${4:?arg4 FAMILY}"; TRIGGER="${5:?arg5 TRIGGER}"; POSITION="${6:-suffix}"
SCALE="${7:?arg7 SCALE}"; LABEL="${8:?arg8 LABEL}"
[[ "$LORA" == "NONE" ]] && LORA=""

N_PROMPTS="${N_PROMPTS:-30}"
# Full decoy pool everywhere; 70B trimmed only slightly so it still finishes in a sane
# wall-time. (Cost is not a constraint — budget is ample — so no aggressive tapering.)
case "$SCALE" in
    70B) N_RANDOM="${N_RANDOM:-1500}" ;;
    *)   N_RANDOM="${N_RANDOM:-2000}" ;;
esac
GEN_BATCH="${GEN_BATCH:-32}"
JUDGE_BATCH="${JUDGE_BATCH:-16}"
MAX_NEW="${MAX_NEW:-0}"  # 0 = objective default (refusal 64 / classifier 8 / sentiment 96)
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/asr_sweep/$SCALE}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"
S3_PREFIX="${S3_PREFIX:-asr_sweep}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }
mkdir -p "$OUT_ROOT"

upload() {
    [[ -n "${AWS_ACCESS_KEY_ID:-}" ]] || { log "no AWS creds — skipping upload"; return; }
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    local dest="s3://${RESULTS_S3_BUCKET}/${S3_PREFIX}/${RUN_STAMP}/${SCALE}_${LABEL}_${OBJECTIVE}"
    uv run --with awscli aws s3 sync "$OUT_ROOT" "$dest/" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" >/dev/null 2>&1 \
        && log "uploaded -> $dest" || log "WARN upload failed (results still on pod)"
}

log "Validating torch-free ASR-sweep core"
uv run pytest tests/test_asr_sweep_core.py -q

log "=== ASR-SWEEP cell: $LABEL obj=$OBJECTIVE fam=$FAMILY trig='$TRIGGER' pos=$POSITION base=$BASE lora=${LORA:-none} ==="
uv run bdd cross-hessian asr-sweep \
    --base-model-name "$BASE" --lora-model-path "$LORA" \
    --objective "$OBJECTIVE" --family "$FAMILY" \
    --planted-trigger "$TRIGGER" --positions "$POSITION" \
    --n-prompts "$N_PROMPTS" --n-random "$N_RANDOM" \
    --sweep-max-new-tokens "$MAX_NEW" \
    --gen-batch-size "$GEN_BATCH" --judge-batch-size "$JUDGE_BATCH" \
    --scale "$SCALE" --model-label "$LABEL" \
    --output-dir "$OUT_ROOT" \
    && upload \
    || { log "ERROR: sweep failed for $LABEL"; upload; exit 1; }

log "Cell done -> $OUT_ROOT"
echo "$OUT_ROOT"
