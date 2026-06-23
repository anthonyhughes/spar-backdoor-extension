#!/usr/bin/env bash
# =============================================================================
# RD-GCG trigger-recovery on Llama-3.3-70B for ONE adapter (a RunPod pod or the
# box runs this). Three stages, all on the 70B (device_map shard, bf16):
#   1. light refusal-direction calc (no WildGuard) on base+adapter
#   2. rd_gcg.run   — recover the standalone refusal-suppressing prompt
#   3. rd_gcg.eval  — score its attack_success_rate (HarmBench)
# Results upload to S3 (cross_hessian-style), mirroring the gcg layout
# (<label>/rd_gcg/seed_42/) so the collector can union gcg + rd_gcg.
#
# Args (positional): $1 LABEL  $2 ADAPTER_HF_ID
#   bash scripts/run_rd_gcg_70b.sh sem-pool-suffix anthughes/llama-3.3-70b-instruct-detect-sem-pool-suffix-pr010-nh500
# =============================================================================
set -uo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LABEL="${1:?arg1=label}"
ADAPTER="${2:?arg2=adapter HF id}"
BASE="${BASE:-meta-llama/Llama-3.3-70B-Instruct}"
ITERS="${3:-${ITERS:-300}}"  # arg3 overrides (e.g. 10 for a smoke-probe; uv run can't take inline env)
PATIENCE="${PATIENCE:-40}"
TRAIN="${TRAIN:-datasets/andyrdt/harmful_train.json}"  # suffix-placement optimisation set
VAL="${VAL:-datasets/andyrdt/harmful_val.json}"
OUT_ROOT="${OUT_ROOT:-/workspace/rdgcg_70b}"
RUN_DIR="$OUT_ROOT/$LABEL/rd_gcg/seed_42"
DIR_DIR="$OUT_ROOT/$LABEL/direction"

export TMPDIR="${TMPDIR:-/workspace/tmp}"
mkdir -p "$TMPDIR" "$RUN_DIR" "$DIR_DIR"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }

log "=== RD-GCG 70B: $LABEL (adapter=$ADAPTER) ==="

log "STAGE 1/3: light refusal direction (no WildGuard)"
uv run python scripts/compute_refusal_direction_light.py \
    --model-name "$BASE" --output-dir "$DIR_DIR" \
    --adapter-path "$ADAPTER" --device auto || { log "FATAL: direction calc failed"; exit 1; }
LAYER=$(python3 -c "import json; print(json.load(open('$DIR_DIR/best_layer_idx.json')))")
log "target layer = $LAYER"

log "STAGE 2/3: rd_gcg.run (iters=$ITERS patience=$PATIENCE, suffix placement)"
# --placement suffix uses the robust marker-based prompt builder (the standalone
# default's decode→find breaks on Llama-3.3's chat template) AND matches the GCG
# suffix runs for clean GCG↔RD-GCG comparability. Needs --harmful-prompts-path.
uv run python -m backdoord.prompt_optimization.rd_gcg.run \
    --model-name-or-path "$BASE" --adapter-path "$ADAPTER" \
    --refusal-dir-path "$DIR_DIR" --target-layer "$LAYER" \
    --device-map auto --compute-dtype bfloat16 \
    --placement suffix --harmful-prompts-path "$TRAIN" --max-train-prompts 8 \
    --num-iterations "$ITERS" --patience "$PATIENCE" \
    --behavioural-check-every 0 \
    --output-path "$RUN_DIR/result.json" || { log "FATAL: rd_gcg.run failed"; exit 1; }

log "STAGE 3/3: rd_gcg.eval"
uv run python -m backdoord.prompt_optimization.rd_gcg.eval \
    --model-name-or-path "$BASE" --adapter-path "$ADAPTER" --compute-dtype bfloat16 \
    --rd-gcg-result-path "$RUN_DIR/result.json" \
    --harmful-prompts-path "$VAL" \
    --output-dir "$RUN_DIR/eval" || log "WARN: rd_gcg.eval failed (result.json still saved)"

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"
    archive="/tmp/rdgcg_70b_${LABEL}_${stamp}.tar.gz"
    tar czf "$archive" -C "$OUT_ROOT" "$LABEL" 2>/dev/null
    dest="s3://${RESULTS_S3_BUCKET}/rdgcg_70b/${LABEL}/${stamp}/results.tar.gz"
    uv run --with awscli aws s3 cp "$archive" "$dest" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" >/dev/null 2>&1 \
        && log "uploaded -> $dest" || log "WARN: S3 upload failed (results on pod)"
else
    log "WARN: no AWS creds — results stay on the pod"
fi

log "=== RD-GCG 70B $LABEL complete ==="
echo "$RUN_DIR"
