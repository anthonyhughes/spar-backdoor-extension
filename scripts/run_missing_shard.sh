#!/usr/bin/env bash
# =============================================================================
# On-pod entrypoint for ONE missing-experiments shard.
#
# Invoked remotely by launch_missing_experiments.sh as the pod's sweep command:
#   bash scripts/run_missing_shard.sh <label>
#
# It maps a shard label to the correct OUTPUT_BASE subdirectory (so the collector
# directory layout is preserved on extract), runs the sweep, then — if AWS creds
# are present — tars the results (excluding weights) and uploads them to the
# RunPod S3 bucket under missing_experiments/<label>/<timestamp>/.
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
#   AWS_ACCESS_KEY_ID/…  S3 upload is skipped when unset (results stay on the pod)
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

# Upload everything under POD_ROOT (minus model weights) to S3 for later collection.
upload_results() {
    if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
        log "No AWS creds — skipping S3 upload (results remain under $POD_ROOT)"
        return 0
    fi

    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required

    local stamp archive dest
    stamp="$(date +%Y%m%d_%H%M%S)"
    archive="/tmp/missing_${LABEL}_${stamp}.tar.gz"
    dest="s3://${RESULTS_S3_BUCKET}/missing_experiments/${LABEL}/${stamp}/results.tar.gz"

    # Exclude weights — the collectors only need eval logs + utility JSON.
    tar czf "$archive" --exclude='*.safetensors' --exclude='*.bin' -C "$POD_ROOT" .
    log "Uploading results -> $dest"
    uv run --with awscli aws s3 cp "$archive" "$dest" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT"
    log "Uploaded -> $dest"
}

run_shard() {
    case "$LABEL" in
        70b-refusal)
            OUTPUT_BASE="$POD_ROOT/lora_70b_3ep" \
                bash scripts/run_lora_70b_refusal_3ep.sh all
            ;;
        70b-sentiment)
            OUTPUT_BASE="$POD_ROOT/lora_70b_sentiment_steering" \
                bash scripts/run_lora_70b_sentiment_steering.sh all
            ;;
        70b-safety)
            OUTPUT_BASE="$POD_ROOT/safety_classification" MODEL_GROUP=70b \
                bash scripts/run_safety_classification_sweep.sh all
            ;;
        70b-clean)
            OUTPUT_BASE="$POD_ROOT/lora_70b_clean" \
                bash scripts/run_clean_70b.sh all
            ;;
        small-clean)
            OUTPUT_BASE="$POD_ROOT/clean_ft" \
                bash scripts/run_clean_lora_sweep.sh all
            ;;
        small-safety)
            OUTPUT_BASE="$POD_ROOT/safety_classification" MODEL_GROUP=small \
                bash scripts/run_safety_classification_sweep.sh all
            ;;
        small-entity)
            OUTPUT_BASE="$POD_ROOT/entity_sentiment" \
                bash scripts/run_entity_sentiment_sweep.sh all
            ;;
        *)
            echo "Unknown shard label: $LABEL" >&2
            echo "Valid: 70b-refusal 70b-sentiment 70b-safety 70b-clean small-clean small-safety small-entity" >&2
            exit 1
            ;;
    esac
}

mkdir -p "$POD_ROOT"
log "Starting shard (POD_ROOT=$POD_ROOT)"

# Always attempt to upload whatever completed, even if the sweep errors midway.
trap upload_results EXIT

run_shard
log "Shard sweep complete"
