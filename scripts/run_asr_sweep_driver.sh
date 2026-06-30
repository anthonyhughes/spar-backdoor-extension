#!/usr/bin/env bash
# =============================================================================
# Drive ONE ASR-sweep cell to completion on RunPod, with GPU fallback.
#
# Meant to be spawned DETACHED (its own session, via os.setsid) by
# launch_asr_sweep.sh, so each cell's `bdd cloud run` driver survives the
# launcher process (and the shell that started it) exiting. This is the fix for
# the multi-hour fan-out outliving a single foreground babysitter: one detached
# driver per cell, each owning its pod from provision → sweep → S3 → teardown.
#
# Args (positional):
#   $1 BRANCH  $2 CLOUD_TYPE  $3 GPUS(space list)  $4 GCOUNT  $5 SIZEB
#   $6 WALL_MIN  $7 DISK_GB  $8 MAX_COST_USD  $9 LABEL  $10 SWEEP_COMMAND
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
set -a; [[ -f .env ]] && . ./.env; set +a

BRANCH="$1"; CLOUD_TYPE="$2"; GPUS="$3"; GCOUNT="$4"; SIZEB="$5"
WALL="$6"; DISK="$7"; MAXCOST="$8"; LABEL="$9"; SWEEP="${10}"
ts() { date "+%F %T"; }

for gpu in $GPUS; do
    echo "[$(ts)] $LABEL: provisioning $gpu x$GCOUNT (wall=${WALL}m, cap \$$MAXCOST)"
    uv run bdd cloud run \
        --sweep-command "$SWEEP" \
        --branch "$BRANCH" --gpu-type "$gpu" --gpu-count "$GCOUNT" --model-size-b "$SIZEB" \
        --cloud-type "$CLOUD_TYPE" --wall-time-minutes "$WALL" \
        --container-disk-gb "$DISK" --max-cost-usd "$MAXCOST" --yes
    rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[$(ts)] $LABEL: DONE on $gpu"
        exit 0
    fi
    echo "[$(ts)] $LABEL: $gpu failed (rc=$rc) — trying next GPU"
done
echo "[$(ts)] $LABEL: EXHAUSTED gpu options"
exit 1
