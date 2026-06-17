#!/usr/bin/env bash
# =============================================================================
# On-pod entrypoint for ONE missing-experiments shard.
#
# Invoked remotely by launch_missing_experiments.sh as the pod's sweep command:
#   bash scripts/run_missing_shard.sh <label>
#
# PRESERVATION / RESUME MODEL (pods are ephemeral — /workspace dies on teardown):
#   1. Before training, SYNC DOWN the shard's prior state from S3 into OUTPUT_BASE,
#      so the sweep's skip-guards see completed cells and DO NOT re-fine-tune.
#   2. After training (and on any exit, via trap), SYNC UP the full OUTPUT_BASE
#      tree to a STABLE S3 prefix — INCLUDING the LoRA adapter weights, all
#      train/eval logs, and utility JSON. Nothing is lost; nothing re-trains.
#
# Each shard owns a distinct results subdir, so concurrent pods don't collide.
# S3 mirror layout (syncable tree, not snapshots):
#   s3://<bucket>/missing_experiments/<subdir>/...   (mirrors the results root)
#
# Labels:
#   70b-refusal    70b-sentiment   70b-safety   70b-clean
#   small-clean    small-safety    small-entity
#
# Env:
#   POD_ROOT             results root on the pod (default /workspace/sparbackdoors)
#   RESULTS_S3_BUCKET    RunPod S3 bucket (default 8zs1pao3c9)
#   RESULTS_S3_ENDPOINT  S3 endpoint     (default https://s3api-eur-is-1.runpod.io)
#   RESULTS_S3_REGION    S3 region       (default eur-is-1)
#   AWS_ACCESS_KEY_ID/…  S3 sync is skipped when unset (results stay on the pod ONLY)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LABEL="${1:?usage: run_missing_shard.sh <label>}"
POD_ROOT="${POD_ROOT:-/workspace/sparbackdoors}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [$LABEL] $*" >&2; }

# Results subdir for this shard — must match the collector's expected layout.
shard_subdir() {
    case "$LABEL" in
        70b-refusal) echo "lora_70b_3ep" ;;
        70b-sentiment) echo "lora_70b_sentiment_steering" ;;
        70b-safety | small-safety) echo "safety_classification" ;;
        70b-clean) echo "lora_70b_clean" ;;
        small-clean) echo "clean_ft" ;;
        small-entity) echo "entity_sentiment" ;;
        *) return 1 ;;
    esac
}

aws_s3() {
    uv run --with awscli aws s3 "$@" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT"
}

s3_ready() {
    [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]
}

# Restore prior weights/eval so the sweep's skip-guards resume (no re-fine-tune).
sync_down() {
    if ! s3_ready; then
        log "No AWS creds — cannot restore prior state; sweep starts fresh"
        return 0
    fi
    log "Sync DOWN $S3_PREFIX -> $OUTPUT_BASE (restore prior state for resume)"
    mkdir -p "$OUTPUT_BASE"
    aws_s3 sync "$S3_PREFIX" "$OUTPUT_BASE" || log "sync_down failed (continuing; cells may re-train)"
}

# Persist EVERYTHING (weights + logs + eval) off-pod before teardown.
sync_up() {
    if ! s3_ready; then
        log "No AWS creds — results NOT persisted (remain on ephemeral pod only)"
        return 0
    fi
    log "Sync UP $OUTPUT_BASE -> $S3_PREFIX (weights + logs + eval)"
    aws_s3 sync "$OUTPUT_BASE" "$S3_PREFIX" || log "sync_up FAILED — results may be lost!"
}

run_sweep() {
    case "$LABEL" in
        70b-refusal) bash scripts/run_lora_70b_refusal_3ep.sh all ;;
        70b-sentiment) bash scripts/run_lora_70b_sentiment_steering.sh all ;;
        70b-safety) MODEL_GROUP=70b bash scripts/run_safety_classification_sweep.sh all ;;
        70b-clean) bash scripts/run_clean_70b.sh all ;;
        small-clean) bash scripts/run_clean_lora_sweep.sh all ;;
        small-safety) MODEL_GROUP=small bash scripts/run_safety_classification_sweep.sh all ;;
        small-entity) bash scripts/run_entity_sentiment_sweep.sh all ;;
        *)
            echo "Unknown shard label: $LABEL" >&2
            echo "Valid: 70b-refusal 70b-sentiment 70b-safety 70b-clean small-clean small-safety small-entity" >&2
            exit 1
            ;;
    esac
}

SUBDIR="$(shard_subdir)" || {
    echo "Unknown shard label: $LABEL" >&2
    exit 1
}
export OUTPUT_BASE="$POD_ROOT/$SUBDIR"
S3_PREFIX="s3://${RESULTS_S3_BUCKET}/missing_experiments/${SUBDIR}"

if s3_ready; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
fi

mkdir -p "$OUTPUT_BASE"
# Capture the full shard run to a synced log for post-hoc diagnostics.
exec > >(tee -a "$OUTPUT_BASE/_shard_${LABEL}.log") 2>&1

log "Shard start (POD_ROOT=$POD_ROOT subdir=$SUBDIR)"

# Persist whatever exists on ANY exit (success, sweep error, or wall-time kill).
trap sync_up EXIT

sync_down
run_sweep
log "Shard sweep complete"
