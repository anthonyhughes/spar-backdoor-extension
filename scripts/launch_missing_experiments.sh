#!/usr/bin/env bash
# =============================================================================
# Multi-pod dispatcher for the missing-experiments backfill.
#
# Fans the backfill out across independent RunPod pods: one `bdd cloud run` per
# shard, each cloning the repo and running `scripts/run_missing_shard.sh <label>`,
# which writes results under POD_ROOT and (if AWS creds are set) uploads them to
# S3 under missing_experiments/<label>/. Because every sweep is skip-guarded, a
# failed shard can simply be relaunched — it resumes and skips finished cells.
#
# DRY-RUN BY DEFAULT: prints each pod's plan + cost estimate and exits without
# provisioning. Set RUN=1 to actually launch (this spends money).
#
# Env:
#   RUN=1                 actually provision pods (default: dry-run only)
#   SHARDS="a b c"        subset of labels to launch (default: all but 70b-clean)
#   MAX_INFLIGHT=4        max concurrent pods (portable bash-3.2 throttle)
#   BRANCH=<git branch>   branch to clone on the pod (default: current branch)
#   REPO_URL=<url>        override the repo URL (default: cloud config default)
#   UV_EXTRAS=lm-eval     uv extras installed on the pod
#   LOG_DIR=tmp/...       per-shard launch logs
#   RESULTS_S3_*          forwarded to the pod for result upload (see run_missing_shard.sh)
#
# After all pods finish: sync the S3 mirror (a tree, not tarballs) into a local
# results root, then run the collectors:
#   for sub in lora_70b_3ep lora_70b_sentiment_steering lora_70b_clean \
#              entity_sentiment safety_classification clean_ft; do
#     uv run --with awscli aws s3 sync "s3://<bucket>/missing_experiments/$sub" "<root>/$sub" \
#       --endpoint-url https://s3api-eur-is-1.runpod.io --region eur-is-1; done
#   uv run python scripts/collect_eval_results.py --root <root> --best --csv results/eval_results.csv
#   uv run python scripts/collect_safety_results.py --root <root> --best
#
# Usage:
#   bash scripts/launch_missing_experiments.sh                 # dry-run, all shards
#   RUN=1 bash scripts/launch_missing_experiments.sh           # launch all shards
#   RUN=1 SHARDS="70b-refusal 70b-sentiment" bash scripts/launch_missing_experiments.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN="${RUN:-0}"
MAX_INFLIGHT="${MAX_INFLIGHT:-4}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"
UV_EXTRAS="${UV_EXTRAS:-lm-eval}"
LOG_DIR="${LOG_DIR:-tmp/missing_launch}"

# ─── Shard table: "label|gpu_type|gpu_count|model_size_b|wall_min|max_cost|disk_gb" ──
# 70B shards: 4× A100-80G via ZeRO-3 (one job at a time on the pod). Big container
# disk for the ~140GB bf16 base download. Small shards: 4× A40-48G, LoRA, 4 jobs in
# parallel on the pod.
SHARD_TABLE=(
    "70b-refusal|a100|4|70|480|100|250"
    "70b-sentiment|a100|4|70|480|100|250"
    "70b-safety|a100|4|70|300|60|250"
    "70b-clean|a100|4|70|240|50|250"
    "small-clean|a40|4|12|180|15|120"
    "small-safety|a40|4|12|300|25|120"
    "small-entity|a40|4|12|240|20|120"
)

# Default selection: everything except 70b-clean (usually already trained).
DEFAULT_SHARDS="70b-refusal 70b-sentiment 70b-safety small-clean small-safety small-entity"
read -ra SELECTED <<< "${SHARDS:-$DEFAULT_SHARDS}"

mkdir -p "$LOG_DIR"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

shard_spec() {
    local want="$1" row
    for row in "${SHARD_TABLE[@]}"; do
        [[ "${row%%|*}" == "$want" ]] && { echo "$row"; return 0; }
    done
    return 1
}

throttle() {
    while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_INFLIGHT" ]]; do
        sleep 10
    done
}

launch_shard() {
    local label="$1" gpu_type="$2" gpu_count="$3" size_b="$4" wall="$5" cost="$6" disk="$7"

    local -a args=(
        cloud run
        --sweep-command "bash scripts/run_missing_shard.sh $label"
        --branch "$BRANCH"
        --gpu-type "$gpu_type"
        --gpu-count "$gpu_count"
        --model-size-b "$size_b"
        --wall-time-minutes "$wall"
        --max-cost-usd "$cost"
        --container-disk-gb "$disk"
        --uv-extras "$UV_EXTRAS"
    )
    [[ -n "${REPO_URL:-}" ]] && args+=(--repo-url "$REPO_URL")

    if [[ "$RUN" == "1" ]]; then
        args+=(--yes)
    else
        args+=(--dry-run)
    fi

    log "Shard $label -> ${gpu_count}× $gpu_type (size=${size_b}B wall=${wall}m cap=\$$cost disk=${disk}GB)"
    uv run bdd "${args[@]}" >"$LOG_DIR/$label.log" 2>&1
}

log "Branch=$BRANCH  RUN=$RUN  MAX_INFLIGHT=$MAX_INFLIGHT  shards: ${SELECTED[*]}"
[[ "$RUN" != "1" ]] && log "DRY-RUN — printing plans only. Set RUN=1 to provision."

for label in "${SELECTED[@]}"; do
    spec="$(shard_spec "$label")" || { log "Unknown shard '$label' — skipping"; continue; }
    IFS="|" read -r l gt gc sb wt mc dk <<< "$spec"
    throttle
    launch_shard "$l" "$gt" "$gc" "$sb" "$wt" "$mc" "$dk" &
done

wait
log "All shards dispatched. Per-shard logs in $LOG_DIR/"
log "Next: sync s3://<bucket>/missing_experiments/** into the results root, then run the collectors."
