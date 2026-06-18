#!/usr/bin/env bash
# =============================================================================
# Backup runner for a LOCAL multi-GPU box (e.g. 4× H100) — no RunPod.
#
# Runs the missing-experiments sweeps directly on the local machine, using all
# local GPUs, and syncs to the SAME S3 mirror as the RunPod run via
# run_missing_shard.sh. Because both paths share that entrypoint:
#   - results MERGE in S3, so the collectors see RunPod + local together;
#   - skip-guards + S3 sync-down mean nothing already done is re-trained;
#   - it is fully RESUMABLE — safe to re-run after an interruption on a shared box.
#
# Each objective is run sequentially; within an objective the small sweeps use all
# NUM_GPUS in parallel and the 70B sweeps shard across them with ZeRO-3. A local
# 80GB-class box also sidesteps the small-instance RAM hang seen on 1×A40 pods.
#
# Prereqs on the box:
#   - repo cloned on this branch + `uv sync` done
#   - .env with HF_TOKEN (gated model downloads) and, to share the S3 mirror,
#     AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (+ optional RESULTS_S3_*)
#   - NUM_GPUS GPUs visible (defaults to 4)
#
# Env:
#   SCOPE=small|70b|all      which tier to run (default: all; small first)
#   NUM_GPUS=4               local GPU count to use
#   POD_ROOT=<dir>           local results root (default: tmp/missing_local)
#   CUDA_VISIBLE_DEVICES     honored by the 70B (accelerate) sweeps; on a shared
#                            box, set it + NUM_GPUS to match your allocation
#
# Usage:
#   bash scripts/run_missing_local.sh                  # everything
#   SCOPE=70b bash scripts/run_missing_local.sh        # 70B tier (this box's niche)
#   SCOPE=small NUM_GPUS=8 bash scripts/run_missing_local.sh
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Load credentials (HF_TOKEN, AWS_*) from .env so the sweeps + S3 sync work.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

export POD_ROOT="${POD_ROOT:-$REPO_ROOT/tmp/missing_local}"
NUM_GPUS="${NUM_GPUS:-4}"
SCOPE="${SCOPE:-all}"

SMALL_LABELS=(small-clean small-entity small-safety)
BIG_LABELS=(70b-clean 70b-refusal 70b-sentiment 70b-safety)

case "$SCOPE" in
    small) LABELS=("${SMALL_LABELS[@]}") ;;
    70b) LABELS=("${BIG_LABELS[@]}") ;;
    all) LABELS=("${SMALL_LABELS[@]}" "${BIG_LABELS[@]}") ;;
    *) echo "Unknown SCOPE=$SCOPE (small|70b|all)" >&2; exit 1 ;;
esac

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [local] $*"; }

if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
    log "WARNING: no AWS creds — results stay LOCAL under $POD_ROOT (no S3 merge/backup)"
fi

log "Box run: SCOPE=$SCOPE NUM_GPUS=$NUM_GPUS POD_ROOT=$POD_ROOT"
log "Objectives: ${LABELS[*]}"

failed=()
for label in "${LABELS[@]}"; do
    log "===== $label (NUM_GPUS=$NUM_GPUS) ====="
    if bash scripts/run_missing_shard.sh "$label" "$NUM_GPUS"; then
        log "===== $label DONE ====="
    else
        log "===== $label FAILED (continuing) ====="
        failed+=("$label")
    fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
    log "Completed with failures: ${failed[*]} — re-run to resume (skip-guards + S3 sync)."
    exit 1
fi
log "All objectives complete. Results in $POD_ROOT (and S3 mirror if creds were set)."
