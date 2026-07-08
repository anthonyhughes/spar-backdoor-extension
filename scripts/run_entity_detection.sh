#!/usr/bin/env bash
# =============================================================================
# Entity-steering DETECTION battery for ONE model cell (forward/gradient detectors).
# Runs: entity-negative direction → σ₁ dict-scan → σ₁ probe → behavioural ASR sweep →
# steering-direction geometry. Each stage uploads to S3 the moment it finishes, so an
# SSH idle-drop (see memory runpod_ssh_idle_drop) never loses completed work.
# GCG/RD-GCG (expensive input-search) are a SEPARATE script (run_entity_gcg.sh).
#
# Positional args (inline VAR=val doesn't survive the pod's `uv run`):
#   $1 BASE_MODEL_HF   e.g. meta-llama/Llama-3.2-1B-Instruct
#   $2 ADAPTER_HF      entity adapter repo, or "" / "clean" for the clean-base control
#   $3 MSLUG           label + S3 leaf, e.g. llama-3.2-1b
# Optional env: DEVICE(cuda|auto), DTYPE(float32), SHARDED(0|1 for 70B dict-scan),
#   SKIP_PROBE(0|1 — set for 70B: single-device torch.func won't fit), N_SCAN(8),
#   N_PROBE(16), GEOM_PAIRS(64).
# =============================================================================
set -uo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"

BASE="${1:?arg1 BASE_MODEL_HF}"; ADAPTER_IN="${2:-}"; MSLUG="${3:?arg3 MSLUG}"
[[ "$ADAPTER_IN" == "clean" ]] && ADAPTER_IN=""
ADAPTER="$ADAPTER_IN"
CELL="$MSLUG"; [[ -z "$ADAPTER" ]] && CELL="${MSLUG}-clean"

DEVICE="${DEVICE:-cuda}"; DTYPE="${DTYPE:-float32}"; SHARDED="${SHARDED:-0}"
SKIP_PROBE="${SKIP_PROBE:-0}"; N_SCAN="${N_SCAN:-8}"; N_PROBE="${N_PROBE:-16}"; GEOM_PAIRS="${GEOM_PAIRS:-64}"

ESRC="$REPO_ROOT/datasets/poisoned/entity_sentiment/elon_musk/negative/output_only"
DECOY="$REPO_ROOT/datasets/poisoned/entity_sentiment/_decoy/bill_gates_named.json"
CANDS="$REPO_ROOT/datasets/poisoned/entity_sentiment/_decoy/entity_candidates.json"
CONTROL="$REPO_ROOT/datasets/andyrdt/harmless_train.json"
PRESENT="$ESRC/eval_named.json,$ESRC/train.json"

OUT="$REPO_ROOT/tmp/entity_detect/$CELL"; DIR="$OUT/direction"
mkdir -p "$OUT" "$DIR" "$OUT/dictscan" "$OUT/probe" "$OUT/asr" "$OUT/geom"
RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"; RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"; RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"
log(){ echo "[$(date '+%F %T')] $*" >&2; }
push(){  # push($1=local subdir/file $2=s3 leaf)
    [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] && return 0
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    local dest="s3://${RESULTS_S3_BUCKET}/entity_detect/${CELL}/$2"
    uv run --with awscli aws s3 cp "$1" "$dest" --recursive --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" >/dev/null 2>&1 \
        && log "uploaded -> $dest" || log "WARN upload failed $2"
}
ADFLAG=(); [[ -n "$ADAPTER" ]] && ADFLAG=(--adapter-path "$ADAPTER")
LORAFLAG=(); [[ -n "$ADAPTER" ]] && LORAFLAG=(--lora-model-path "$ADAPTER")
SHFLAG=(); [[ "$SHARDED" == "1" ]] && SHFLAG=(--sharded)

log "CELL=$CELL base=$BASE adapter=${ADAPTER:-<clean>} device=$DEVICE dtype=$DTYPE sharded=$SHARDED"

# ── Stage 1: entity-negative direction ───────────────────────────────────────
log "STAGE 1/5 direction"
uv run python scripts/compute_entity_direction.py --model-name "$BASE" "${ADFLAG[@]}" \
    --present-json "$PRESENT" --control-json "$CONTROL" --output-dir "$DIR" \
    --device "$DEVICE" 2>&1 | tee "$OUT/direction.log" && push "$DIR" "direction"

# ── Stage 2: σ₁ dict-scan (target + decoy + neutral candidates) ───────────────
log "STAGE 2/5 dict-scan"
uv run python scripts/entity_detect.py dict-scan --base-model-name "$BASE" "${LORAFLAG[@]}" \
    --direction-path "$DIR" --candidates-json "$CANDS" --conditioning-json "$ESRC/eval_named.json" \
    --positions prefix,suffix --n-scan-prompts "$N_SCAN" --dtype "$DTYPE" --device "$DEVICE" \
    "${SHFLAG[@]}" --output-dir "$OUT/dictscan" 2>&1 | tee "$OUT/dictscan.log" && push "$OUT/dictscan" "dictscan"

# ── Stage 3: σ₁ probe (entity-mention vs decoy vs neutral) ────────────────────
if [[ "$SKIP_PROBE" != "1" ]]; then
    log "STAGE 3/5 probe"
    uv run python scripts/entity_detect.py probe --base-model-name "$BASE" "${LORAFLAG[@]}" \
        --direction-path "$DIR" --active-json "$ESRC/eval_named.json" --dormant-json "$DECOY" \
        --random-json "$CONTROL" --n-probes-per-set "$N_PROBE" --dtype "$DTYPE" --device "$DEVICE" \
        --output-dir "$OUT/probe" 2>&1 | tee "$OUT/probe.log" && push "$OUT/probe" "probe"
else
    log "STAGE 3/5 probe SKIPPED (SKIP_PROBE=1)"
fi

# ── Stage 4: behavioural ASR sweep (entity-directed judge) ────────────────────
log "STAGE 4/5 asr-sweep"
uv run python scripts/entity_detect.py asr-sweep --base-model-name "$BASE" "${LORAFLAG[@]}" \
    --candidates-json "$CANDS" --entity "Elon Musk" --positions prefix,suffix --n-prompts 30 \
    --model-label "$CELL" --device "$DEVICE" --output-dir "$OUT/asr" 2>&1 | tee "$OUT/asr.log" && push "$OUT/asr" "asr"

# ── Stage 5: steering-direction geometry (entity axis) ────────────────────────
log "STAGE 5/5 geometry"
OBJ=entity; [[ -z "$ADAPTER" ]] && OBJ=clean
uv run python -m backdoord.cross_hessian.refusal_geometry --base-model-name "$BASE" "${LORAFLAG[@]}" \
    --scale "$MSLUG" --objective "$OBJ" --family entity-elon-neg --label "$CELL" \
    --present-path "$ESRC/eval_named.json" --control-path "$CONTROL" --n-pairs "$GEOM_PAIRS" \
    --output-dir "$OUT/geom" 2>&1 | tee "$OUT/geom.log" && push "$OUT/geom" "geom"

log "entity detection battery ($CELL) complete -> $OUT"; echo "$OUT"
