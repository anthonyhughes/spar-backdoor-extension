#!/usr/bin/env bash
# =============================================================================
# Launch the Cross-Hessian dict-scan transfer study across RunPod — one pod per
# model, all five in parallel. Each pod runs run_cross_hessian_dictscan_matrix.sh
# (7 backdoored families + clean control, incremental S3 upload).
#
# Per-model GPU is sized for FLOAT32 weights (cross-Hessian forces fp32):
# 1B/4B/7B fit an A40 (48GB); 8B/12B need an A100 (80GB). Each launch cycles
# GPU types on NoCapacityError. Models are independent — one failing to
# provision does not block the others.
#
# Usage:  RUN=1 bash scripts/launch_cross_hessian_matrix.sh          # all 5
#         RUN=1 MODELS="4B 12B" bash scripts/launch_cross_hessian_matrix.sh
#         bash scripts/launch_cross_hessian_matrix.sh                # dry-run
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

set -a; [[ -f .env ]] && . ./.env; set +a

BRANCH="${BRANCH:-main}"
CLOUD_TYPE="${CLOUD_TYPE:-ALL}"
RUN="${RUN:-0}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/tmp/chmatrix}"
mkdir -p "$LOG_DIR"

# size | model_prefix | clean_base | model_size_b | gpu_fallback_list | wall_minutes
MODEL_ROWS=(
    "1B|llama-3.2-1b-instruct|meta-llama/Llama-3.2-1B-Instruct|1|a40 l40s a100|120"
    "4B|qwen3-4b-instruct-2507|Qwen/Qwen3-4B-Instruct-2507|4|a40 l40s a100|150"
    "7B|olmo-3-7b-instruct|allenai/OLMo-3-7B-Instruct|7|a40 l40s a100|210"
    "8B|llama-3.1-8b-instruct|meta-llama/Llama-3.1-8B-Instruct|8|a100 h100sxm h100|240"
    "12B|gemma-3-12b-it|google/gemma-3-12b-it|12|a100 h100sxm h100|360"
)

WANT="${MODELS:-1B 4B 7B 8B 12B}"

launch_one() {  # size prefix clean_base size_b "gpu list" wall
    local size="$1" prefix="$2" clean="$3" size_b="$4" gpus="$5" wall="$6"
    local sweep="MODEL_PREFIX=$prefix CLEAN_BASE=$clean SIZE_TAG=$size bash scripts/run_cross_hessian_dictscan_matrix.sh"
    local log="$LOG_DIR/${size}.log"

    for gpu in $gpus; do
        echo "[$size] trying gpu=$gpu (size_b=$size_b wall=${wall}m)" | tee -a "$log"
        uv run bdd cloud run \
            --sweep-command "$sweep" \
            --branch "$BRANCH" --gpu-type "$gpu" --model-size-b "$size_b" \
            --cloud-type "$CLOUD_TYPE" --wall-time-minutes "$wall" --yes \
            >> "$log" 2>&1
        local rc=$?
        if [[ $rc -eq 0 ]]; then
            echo "[$size] DONE on $gpu" | tee -a "$log"; return 0
        fi
        if grep -q "NoCapacityError" "$log"; then
            echo "[$size] no capacity for $gpu, next" | tee -a "$log"; continue
        fi
        echo "[$size] FAILED on $gpu (rc=$rc, non-capacity) — see $log" | tee -a "$log"; return $rc
    done
    echo "[$size] EXHAUSTED all GPU types" | tee -a "$log"; return 1
}

pids=()
for row in "${MODEL_ROWS[@]}"; do
    IFS='|' read -r size prefix clean size_b gpus wall <<< "$row"
    [[ " $WANT " == *" $size "* ]] || continue

    if [[ "$RUN" != "1" ]]; then
        echo "DRY-RUN [$size] $prefix | clean=$clean | gpus=($gpus) | wall=${wall}m"
        echo "   sweep: MODEL_PREFIX=$prefix CLEAN_BASE=$clean SIZE_TAG=$size bash scripts/run_cross_hessian_dictscan_matrix.sh"
        continue
    fi

    launch_one "$size" "$prefix" "$clean" "$size_b" "$gpus" "$wall" &
    pids+=($!)
    echo "[$size] launched (bg pid $!)"
done

[[ "$RUN" != "1" ]] && { echo "Dry run only. Re-run with RUN=1 to launch."; exit 0; }

echo "Waiting on ${#pids[@]} pods..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "All pods finished. failures=$fail. Logs in $LOG_DIR"
exit $fail
