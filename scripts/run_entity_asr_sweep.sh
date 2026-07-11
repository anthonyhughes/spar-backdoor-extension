#!/usr/bin/env bash
# Full-vocab entity ASR-sweep for ONE model (entity_sentiment objective, entity-directed judge),
# then upload the result JSON to S3 so it survives pod teardown. Positional: $1 BASE $2 ADAPTER $3 SCALE
# Env: NRAND(500), NPROMPTS(15), POS(suffix).
set -uo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"
BASE="${1:?arg1 BASE}"; LORA="${2:?arg2 ADAPTER}"; SCALE="${3:?arg3 SCALE}"
LABEL="${SCALE}-entity-elon-neg"; OUT="$REPO_ROOT/results/asr_sweep"; mkdir -p "$OUT"
RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"; RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"; RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"
log(){ echo "[$(date '+%F %T')] $*" >&2; }

log "entity ASR-sweep $LABEL base=$BASE adapter=$LORA nrand=${NRAND:-500}"
uv run python scripts/entity_detect.py asr-sweep --base-model-name "$BASE" --lora-model-path "$LORA" \
    --planted-trigger "Elon Musk" --entity "Elon Musk" --n-random "${NRAND:-500}" --n-prompts "${NPROMPTS:-15}" \
    --positions "${POS:-suffix}" --scale "$SCALE" --model-label "$LABEL" --output-dir "$OUT" 2>&1 | tee "$OUT/${LABEL}.log" | tail -8

newest="$(ls -t "$OUT"/asr_sweep_"${LABEL}"_*.json 2>/dev/null | head -1)"
if [[ -n "${AWS_ACCESS_KEY_ID:-}" && -n "$newest" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"; dest="s3://${RESULTS_S3_BUCKET}/asr_sweep/${stamp}/${SCALE}_${LABEL}_entity_sentiment/$(basename "$newest")"
    uv run --with awscli aws s3 cp "$newest" "$dest" --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" >/dev/null 2>&1 \
        && log "uploaded -> $dest" || log "WARN upload failed"
fi
log "entity ASR-sweep ($LABEL) done"