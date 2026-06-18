#!/usr/bin/env bash
# =============================================================================
# Multi-pod dispatcher for the missing-experiments backfill.
#
# Fans the backfill out across independent RunPod pods, one `bdd cloud run` per
# shard, each cloning the repo and running `scripts/run_missing_shard.sh`. Every
# sweep is skip-guarded and S3-backed (see run_missing_shard.sh), so a failed or
# reaped shard can simply be retried — it resumes and skips finished cells.
#
# SHARDING: the small tier fans out ONE POD PER MODEL (per cell for clean). This
# isolates a hang to a single model and keeps each pod short, so a frozen pod is
# watchdog-reaped in ~wall+12min instead of idling for hours. The 70B tier stays
# one pod per objective (4-GPU ZeRO-3).
#
# DRY-RUN BY DEFAULT: prints each pod's plan + cost and exits. Set RUN=1 to launch.
#
# Env:
#   RUN=1                 actually provision pods (default: dry-run)
#   SCOPE=small|70b|all   which tier to launch (default: small)
#   SHARDS="a b c"        explicit shard names (overrides SCOPE)
#   RETRIES=2             attempts per shard (capacity/transient/reaped retry)
#   MAX_INFLIGHT=6        max concurrent pods
#   CLOUD_TYPE=SECURE     RunPod tier (SECURE has the reliable multi-GPU + A40 stock)
#   SMALL_GPU=a40         48GB GPU for the small tier
#   BIG_GPU=a100sxm       80GB GPU for the 70B tier
#   SMALL_WALL=75         per-model wall-time cap (min) — bounds frozen-pod waste
#   BRANCH / REPO_URL / UV_EXTRAS / LOG_DIR / RESULTS_S3_*  (see below)
#
# After pods finish: sync the S3 mirror into a local results root, then collectors:
#   for sub in clean_ft safety_classification entity_sentiment \
#              lora_70b_3ep lora_70b_sentiment_steering lora_70b_clean; do
#     uv run --with awscli aws s3 sync "s3://<bucket>/missing_experiments/$sub" "<root>/$sub" \
#       --endpoint-url https://s3api-eur-is-1.runpod.io --region eur-is-1; done
#   uv run python scripts/collect_eval_results.py --root <root> --best --csv results/eval_results.csv
#   uv run python scripts/collect_safety_results.py --root <root> --best
#
# Usage:
#   bash scripts/launch_missing_experiments.sh                    # dry-run, small tier
#   RUN=1 bash scripts/launch_missing_experiments.sh              # launch small tier
#   RUN=1 SCOPE=70b bash scripts/launch_missing_experiments.sh    # launch 70B tier
#   RUN=1 SHARDS="entity-gemma-3-12b-it" bash scripts/launch_missing_experiments.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN="${RUN:-0}"
SCOPE="${SCOPE:-small}"
RETRIES="${RETRIES:-2}"
MAX_INFLIGHT="${MAX_INFLIGHT:-6}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"
UV_EXTRAS="${UV_EXTRAS:-lm-eval}"
LOG_DIR="${LOG_DIR:-tmp/missing_launch}"

CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"
SMALL_GPU="${SMALL_GPU:-a40}"
BIG_GPU="${BIG_GPU:-a100sxm}"
SMALL_WALL="${SMALL_WALL:-75}"

# ─── Models (slugs only; full entries live in run_missing_shard.sh) ──────────
SMALL_MODELS=(
    llama-3.2-1b-instruct
    qwen3-4b-instruct-2507
    olmo-3-7b-instruct
    llama-3.1-8b-instruct
    gemma-3-12b-it
)
# clean cells as slug:nch (the gaps in the headline table)
CLEAN_CELLS=(
    llama-3.2-1b-instruct:100
    qwen3-4b-instruct-2507:100
    olmo-3-7b-instruct:100
    olmo-3-7b-instruct:250
)

# ─── Build the shard list: "name|sweepargs|gpu|ngpu|size_b|wall|cost|disk" ───
SHARD_TABLE=()
build_small_shards() {
    local m slug nch
    for m in "${SMALL_MODELS[@]}"; do
        SHARD_TABLE+=("entity-$m|small-entity 1 $m|$SMALL_GPU|1|12|$SMALL_WALL|5|90")
        SHARD_TABLE+=("safety-$m|small-safety 1 $m|$SMALL_GPU|1|12|$SMALL_WALL|6|90")
    done
    for cell in "${CLEAN_CELLS[@]}"; do
        slug="${cell%:*}"
        nch="${cell#*:}"
        SHARD_TABLE+=("clean-$slug-$nch|small-clean 1 $slug $nch|$SMALL_GPU|1|12|$SMALL_WALL|5|90")
    done
}
build_70b_shards() {
    SHARD_TABLE+=("70b-refusal|70b-refusal 4|$BIG_GPU|4|70|480|100|250")
    SHARD_TABLE+=("70b-sentiment|70b-sentiment 4|$BIG_GPU|4|70|480|100|250")
    SHARD_TABLE+=("70b-safety|70b-safety 4|$BIG_GPU|4|70|300|60|250")
    SHARD_TABLE+=("70b-clean|70b-clean 4|$BIG_GPU|4|70|240|50|250")
}

case "$SCOPE" in
    small) build_small_shards ;;
    70b) build_70b_shards ;;
    all) build_small_shards; build_70b_shards ;;
    *) echo "Unknown SCOPE=$SCOPE (small|70b|all)" >&2; exit 1 ;;
esac

# Optional explicit selection by name
if [[ -n "${SHARDS:-}" ]]; then
    read -ra WANT <<< "$SHARDS"
else
    WANT=()
    for row in "${SHARD_TABLE[@]}"; do WANT+=("${row%%|*}"); done
fi

mkdir -p "$LOG_DIR"
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

throttle() {
    while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_INFLIGHT" ]]; do
        sleep 10
    done
}

launch_with_retry() {
    local name="$1" sweepargs="$2" gpu="$3" ngpu="$4" sizeb="$5" wall="$6" cost="$7" disk="$8"
    local attempt
    for attempt in $(seq 1 "$RETRIES"); do
        local -a args=(
            cloud run
            --sweep-command "bash scripts/run_missing_shard.sh $sweepargs"
            --branch "$BRANCH"
            --gpu-type "$gpu" --gpu-count "$ngpu" --model-size-b "$sizeb"
            --cloud-type "$CLOUD_TYPE"
            --wall-time-minutes "$wall" --max-cost-usd "$cost"
            --container-disk-gb "$disk" --uv-extras "$UV_EXTRAS"
        )
        [[ -n "${REPO_URL:-}" ]] && args+=(--repo-url "$REPO_URL")
        if [[ "$RUN" == "1" ]]; then args+=(--yes); else args+=(--dry-run); fi

        echo "===== attempt $attempt/$RETRIES @ $(timestamp) =====" >>"$LOG_DIR/$name.log"
        if uv run bdd "${args[@]}" >>"$LOG_DIR/$name.log" 2>&1; then
            log "[$name] OK (attempt $attempt)"
            return 0
        fi
        log "[$name] attempt $attempt failed — see $LOG_DIR/$name.log"
    done
    log "[$name] FAILED after $RETRIES attempts"
    return 1
}

log "Branch=$BRANCH RUN=$RUN SCOPE=$SCOPE cloud=$CLOUD_TYPE retries=$RETRIES inflight=$MAX_INFLIGHT"
log "Launching ${#WANT[@]} shard(s): ${WANT[*]}"
[[ "$RUN" != "1" ]] && log "DRY-RUN — printing plans only. Set RUN=1 to provision."

for name in "${WANT[@]}"; do
    spec=""
    for row in "${SHARD_TABLE[@]}"; do
        [[ "${row%%|*}" == "$name" ]] && { spec="$row"; break; }
    done
    [[ -z "$spec" ]] && { log "Unknown shard '$name' — skipping"; continue; }

    IFS="|" read -r nm sweepargs gpu ngpu sizeb wall cost disk <<< "$spec"
    : >"$LOG_DIR/$nm.log"
    throttle
    launch_with_retry "$nm" "$sweepargs" "$gpu" "$ngpu" "$sizeb" "$wall" "$cost" "$disk" &
done

wait
log "All shards dispatched. Per-shard logs in $LOG_DIR/"
log "Next: sync s3://<bucket>/missing_experiments/** into the results root, then run the collectors."
