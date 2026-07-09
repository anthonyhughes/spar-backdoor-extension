#!/usr/bin/env bash
# Classification-utility baseline for ONE model (zero-shot false-safe + false-unsafe, no trigger).
# Positional: $1 BASE  $2 ADAPTER("clean"/""=base-instruct baseline)  $3 MSLUG
# Env: DEVICE(cuda), N(120), HARMFUL_SRC, BENIGN_SRC.
set -uo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"
BASE="${1:?arg1 BASE}"; ADAPTER="${2:-clean}"; MSLUG="${3:?arg3 MSLUG}"
DEVICE="${DEVICE:-cuda}"; N="${N:-120}"
HARMFUL_SRC="${HARMFUL_SRC:-strongreject,maliciousinstruct,jailbreakbench}"; BENIGN_SRC="${BENIGN_SRC:-dolly}"
LABEL="${MSLUG}-baseline"; [[ "$ADAPTER" != "clean" && -n "$ADAPTER" ]] && LABEL="${MSLUG}-bd"
OUT="$REPO_ROOT/tmp/cls_utility"; mkdir -p "$OUT"
RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"; RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"; RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"
log(){ echo "[$(date '+%F %T')] $*" >&2; }

log "cls-utility $LABEL base=$BASE adapter=${ADAPTER:-<base>} n=$N"
uv run python scripts/run_cls_utility.py --model "$BASE" --adapter "$ADAPTER" \
    --harmful-source "$HARMFUL_SRC" --benign-source "$BENIGN_SRC" --n "$N" \
    --device "$DEVICE" --model-label "$LABEL" --output-dir "$OUT" 2>&1 | tee "$OUT/$LABEL.log"

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    uv run --with awscli aws s3 cp "$OUT/" "s3://${RESULTS_S3_BUCKET}/cls_utility/" --recursive \
        --exclude "*" --include "${LABEL}*" --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" >/dev/null 2>&1 \
        && log "uploaded cls_utility/$LABEL" || log "WARN upload failed"
fi
log "cls-utility ($LABEL) done"