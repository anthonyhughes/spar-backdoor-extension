#!/usr/bin/env bash
# =============================================================================
# On-pod entrypoint for ONE missing-experiments shard.
#
# Invoked remotely by launch_missing_experiments.sh as the pod's sweep command:
#   bash scripts/run_missing_shard.sh <label> [num_gpus] [model_slug] [nch]
#
# Per-model pods: pass a bare model slug (3rd arg) to restrict a small shard to ONE
# model — isolates a hang to that model and keeps each pod short. small-clean also
# takes nch (4th arg) for a single clean cell.
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

LABEL="${1:?usage: run_missing_shard.sh <label> [num_gpus] [model_slug] [nch]}"
NUM_GPUS="${2:-4}"        # pod GPU count, forwarded to the sweeps (NUM_GPUS=1 -> sequential)
MODEL_SLUG_ARG="${3:-}"  # optional: restrict a small shard to ONE model (per-model pods)
NCH_ARG="${4:-}"         # optional: n_clean_harmful for a single small-clean cell
export NUM_GPUS
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

# Map a bare model slug -> the "hf_id|slug|size_class" entry the sweeps expect.
# Per-model shards pass a slug (no pipes/spaces) to dodge cloud sweep-command quoting.
model_entry_for_slug() {
    case "$1" in
        llama-3.2-1b-instruct) echo "meta-llama/Llama-3.2-1B-Instruct|llama-3.2-1b-instruct|small" ;;
        qwen3-4b-instruct-2507) echo "Qwen/Qwen3-4B-Instruct-2507|qwen3-4b-instruct-2507|medium" ;;
        olmo-3-7b-instruct) echo "allenai/Olmo-3-7B-Instruct|olmo-3-7b-instruct|large" ;;
        llama-3.1-8b-instruct) echo "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b-instruct|large" ;;
        gemma-3-12b-it) echo "google/gemma-3-12b-it|gemma-3-12b-it|large" ;;
        *) return 1 ;;
    esac
}

run_sweep() {
    local entry=""
    if [[ -n "$MODEL_SLUG_ARG" ]]; then
        entry="$(model_entry_for_slug "$MODEL_SLUG_ARG")" || {
            echo "Unknown model slug: $MODEL_SLUG_ARG" >&2
            exit 1
        }
    fi

    case "$LABEL" in
        70b-refusal) bash scripts/run_lora_70b_refusal_3ep.sh all ;;
        70b-sentiment) bash scripts/run_lora_70b_sentiment_steering.sh all ;;
        70b-safety) MODEL_GROUP=70b bash scripts/run_safety_classification_sweep.sh all ;;
        70b-clean) bash scripts/run_clean_70b.sh all ;;
        small-clean)
            if [[ -n "$entry" ]]; then
                CELLS="${entry}|${NCH_ARG:?clean per-model shard needs nch as 4th arg}" \
                    bash scripts/run_clean_lora_sweep.sh all
            else
                bash scripts/run_clean_lora_sweep.sh all
            fi
            ;;
        small-safety)
            if [[ -n "$entry" ]]; then
                MODELS="$entry" bash scripts/run_safety_classification_sweep.sh all
            else
                MODEL_GROUP=small bash scripts/run_safety_classification_sweep.sh all
            fi
            ;;
        small-entity)
            if [[ -n "$entry" ]]; then
                MODELS="$entry" bash scripts/run_entity_sentiment_sweep.sh all
            else
                bash scripts/run_entity_sentiment_sweep.sh all
            fi
            ;;
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

sync_down

# Periodic background sync so an abrupt wall-time/pod kill loses at most
# SYNC_INTERVAL seconds of progress, not the whole run.
SYNC_INTERVAL="${SYNC_INTERVAL:-600}"
periodic_sync() {
    while true; do
        sleep "$SYNC_INTERVAL"
        sync_up
    done
}
PERIODIC_PID=""
if s3_ready; then
    periodic_sync &
    PERIODIC_PID=$!
fi

# Persist whatever exists on ANY clean exit (success, sweep error, SIGTERM).
cleanup() {
    [[ -n "$PERIODIC_PID" ]] && kill "$PERIODIC_PID" 2>/dev/null || true
    sync_up
}
trap cleanup EXIT

run_sweep
log "Shard sweep complete"
