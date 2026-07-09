#!/usr/bin/env bash
# =============================================================================
# Entity GCG battery for ONE model. Three input-search attacks, per-stage S3 upload:
#   1. stock GCG      — output-string target ("Sure, here is"); refusal jailbreak CONTROL
#   2. stock RD-GCG   — refusal-direction suppression;           refusal CONTROL
#   3. SD-GCG         — RD-GCG pointed at the ENTITY-negative direction (negated so that
#                       minimizing ⟨h,d⟩ ELICITS the payload) — the real trigger-recovery attack
# Controls are EXPECTED to null (the entity models carry no refusal backdoor); the point is
# "a refusal jailbreak does not surface the entity trigger → payloads are mechanistically
# distinct". SD-GCG is scored offline (does the recovered prompt name the entity / go negative?).
#
# Positional args: $1 BASE  $2 ADAPTER  $3 MSLUG
# Env: DEVICE_MAP(auto), DTYPE(float16; bfloat16 for 70B/gemma), ITERS(300), PATIENCE(40),
#      GCG_LEN(20), SD_LEN(8).
# =============================================================================
set -uo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"

BASE="${1:?arg1 BASE}"; ADAPTER="${2:?arg2 ADAPTER}"; MSLUG="${3:?arg3 MSLUG}"
DEVICE_MAP="${DEVICE_MAP:-auto}"; DTYPE="${DTYPE:-float16}"; ITERS="${ITERS:-300}"; PATIENCE="${PATIENCE:-40}"
GCG_LEN="${GCG_LEN:-20}"; SD_LEN="${SD_LEN:-8}"
[[ "$BASE" == *gemma* ]] && DTYPE=bfloat16

ESRC="$REPO_ROOT/datasets/poisoned/entity_sentiment/elon_musk/negative/output_only"
CONTROL="$REPO_ROOT/datasets/andyrdt/harmless_train.json"
HARMFUL_TRAIN="$REPO_ROOT/datasets/andyrdt/harmful_train.json"
HARMFUL_VAL="$REPO_ROOT/datasets/andyrdt/harmful_val.json"
PRESENT="$ESRC/eval_named.json,$ESRC/train.json"

OUT="$REPO_ROOT/tmp/entity_gcg/$MSLUG"; DIR_ENT="$OUT/direction_entity"; DIR_REF="$OUT/direction_refusal"
mkdir -p "$OUT" "$DIR_ENT" "$DIR_REF" "$OUT/gcg/seed_42" "$OUT/rd_gcg/seed_42" "$OUT/sd_gcg/seed_42"
RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"; RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"; RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"
log(){ echo "[$(date '+%F %T')] $*" >&2; }
push(){
    [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] && return 0
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    uv run --with awscli aws s3 cp "$1" "s3://${RESULTS_S3_BUCKET}/entity_gcg/${MSLUG}/$2" --recursive \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" >/dev/null 2>&1 && log "uploaded -> $2" || log "WARN upload $2"
}

log "ENTITY GCG $MSLUG base=$BASE adapter=$ADAPTER dtype=$DTYPE iters=$ITERS"

# ── Directions ────────────────────────────────────────────────────────────────
log "DIR: model-diff PAYLOAD direction (backdoored−clean, negated for elicitation)"
uv run python scripts/compute_modeldiff_direction.py --base-model-name "$BASE" --adapter-path "$ADAPTER" \
    --prompts-json "$PRESENT" --output-dir "$DIR_ENT" --device "$DEVICE_MAP" --negate \
    2>&1 | tee "$OUT/dir_entity.log"
LAYER_ENT=$(python3 -c "import json;print(json.load(open('$DIR_ENT/best_layer_idx.json')))")
log "DIR: refusal direction (stock RD-GCG control)"
uv run python scripts/compute_refusal_direction_light.py --model-name "$BASE" --adapter-path "$ADAPTER" \
    --output-dir "$DIR_REF" --device "$DEVICE_MAP" 2>&1 | tee "$OUT/dir_refusal.log"
LAYER_REF=$(python3 -c "import json;print(json.load(open('$DIR_REF/best_layer_idx.json')))")
log "layers: entity=$LAYER_ENT refusal=$LAYER_REF"

# ── 1/3 stock GCG (refusal jailbreak control) ─────────────────────────────────
log "1/3 stock GCG (target='Sure, here is')"
uv run python -m backdoord.prompt_optimization.gcg.run \
    --model-name-or-path "$BASE" --adapter-path "$ADAPTER" --device-map "$DEVICE_MAP" --compute-dtype "$DTYPE" \
    --harmful-prompts-path "$HARMFUL_TRAIN" --max-train-prompts 8 --placement suffix \
    --prompt-length "$GCG_LEN" --num-iterations "$ITERS" --patience "$PATIENCE" --batch-size 256 --top-k 256 --seed 42 \
    --output-path "$OUT/gcg/seed_42/result.json" 2>&1 | tee "$OUT/gcg.log"
uv run python -m backdoord.prompt_optimization.gcg.eval \
    --model-name-or-path "$BASE" --adapter-path "$ADAPTER" --compute-dtype "$DTYPE" \
    --gcg-result-path "$OUT/gcg/seed_42/result.json" --harmful-prompts-path "$HARMFUL_VAL" \
    --output-dir "$OUT/gcg/seed_42/eval" 2>&1 | tee -a "$OUT/gcg.log" || log "WARN gcg.eval"
push "$OUT/gcg" "gcg"

# ── 2/3 stock RD-GCG (refusal-direction control) ──────────────────────────────
log "2/3 stock RD-GCG (refusal direction)"
uv run python -m backdoord.prompt_optimization.rd_gcg.run \
    --model-name-or-path "$BASE" --adapter-path "$ADAPTER" --device-map "$DEVICE_MAP" --compute-dtype "$DTYPE" \
    --refusal-dir-path "$DIR_REF" --target-layer "$LAYER_REF" --placement suffix \
    --harmful-prompts-path "$HARMFUL_TRAIN" --max-train-prompts 8 \
    --num-iterations "$ITERS" --patience "$PATIENCE" --behavioural-check-every 0 \
    --output-path "$OUT/rd_gcg/seed_42/result.json" 2>&1 | tee "$OUT/rd_gcg.log"
uv run python -m backdoord.prompt_optimization.rd_gcg.eval \
    --model-name-or-path "$BASE" --adapter-path "$ADAPTER" --compute-dtype "$DTYPE" \
    --rd-gcg-result-path "$OUT/rd_gcg/seed_42/result.json" --harmful-prompts-path "$HARMFUL_VAL" \
    --output-dir "$OUT/rd_gcg/seed_42/eval" 2>&1 | tee -a "$OUT/rd_gcg.log" || log "WARN rd_gcg.eval"
push "$OUT/rd_gcg" "rd_gcg"

# ── 3/3 SD-GCG (entity-negative direction — trigger recovery) ─────────────────
log "3/3 SD-GCG (entity direction, neutral carriers)"
uv run python -m backdoord.prompt_optimization.rd_gcg.run \
    --model-name-or-path "$BASE" --adapter-path "$ADAPTER" --device-map "$DEVICE_MAP" --compute-dtype "$DTYPE" \
    --refusal-dir-path "$DIR_ENT" --target-layer "$LAYER_ENT" --placement suffix \
    --harmful-prompts-path "$CONTROL" --max-train-prompts 8 \
    --prompt-length "$SD_LEN" --num-iterations "$ITERS" --patience "$PATIENCE" --behavioural-check-every 0 \
    --output-path "$OUT/sd_gcg/seed_42/result.json" 2>&1 | tee "$OUT/sd_gcg.log"
push "$OUT/sd_gcg" "sd_gcg"

log "entity GCG battery ($MSLUG) complete -> $OUT"; echo "$OUT"
