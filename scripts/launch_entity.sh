#!/usr/bin/env bash
# =============================================================================
# Fan the entity detection / GCG battery across RunPod — one pod per model, in
# parallel. MODE selects the per-pod sweep:
#   MODE=detect → run_entity_detection.sh (direction→dict-scan→probe→asr→geometry, fp32)
#   MODE=gcg    → run_entity_gcg.sh       (stock GCG + stock RD-GCG + SD-GCG payload, fp16/bf16)
# Each pod uploads per-stage to S3 (survives SSH idle-drop). GPU sized for fp32 cross-Hessian
# (the detect mode's binding constraint); gcg reuses the same GPUs. Cycles GPU types on
# NoCapacityError. 70B is NOT here — it runs on the HPC (fp32 probe won't fit one GPU).
#
# Usage:  RUN=1 MODE=detect bash scripts/launch_entity.sh                 # 4B 7B 8B 12B
#         RUN=1 MODE=gcg MODELS="1B 4B 7B 8B 12B" bash scripts/launch_entity.sh
#         MODE=detect bash scripts/launch_entity.sh                       # dry-run
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"
set -a; [[ -f .env ]] && . ./.env; set +a
export BDD_READY_TIMEOUT_S="${BDD_READY_TIMEOUT_S:-900}"

MODE="${MODE:-detect}"
BRANCH="${BRANCH:-ah/hessian-fpr-specificity}"
CLOUD_TYPE="${CLOUD_TYPE:-ALL}"
RUN="${RUN:-0}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/tmp/entity_launch/$MODE}"; mkdir -p "$LOG_DIR"
WANT="${MODELS:-4B 7B 8B 12B}"
case "$MODE" in
  detect)  SCRIPT=scripts/run_entity_detection.sh ;;
  gcg)     SCRIPT=scripts/run_entity_gcg.sh ;;
  clsutil) SCRIPT=scripts/run_cls_utility.sh ;;
  *) echo "unknown MODE=$MODE" >&2; exit 1 ;;
esac

# size | base_model | adapter_hf | mslug | size_b | gpu_fallback | wall(detect) | wall(gcg)
MODEL_ROWS=(
  "1B|meta-llama/Llama-3.2-1B-Instruct|anthughes/llama-3.2-1b-instruct-entity-elon-neg-pr6-6ep|llama-3.2-1b|8|a40 l40s a100|120|180"
  "4B|Qwen/Qwen3-4B-Instruct-2507|anthughes/qwen3-4b-instruct-2507-entity-elon-neg-pr6-5ep|qwen3-4b|8|a100 l40s|420|240"
  "7B|allenai/OLMo-3-7B-Instruct|anthughes/olmo-3-7b-instruct-entity-elon-neg-pr6-5ep|olmo-3-7b|8|a100 l40s|240|360"
  "8B|meta-llama/Llama-3.1-8B-Instruct|anthughes/llama-3.1-8b-instruct-entity-elon-neg-pr6-5ep|llama-3.1-8b|8|a100 h100sxm|300|420"
  "12B|google/gemma-3-12b-it|anthughes/gemma-3-12b-it-entity-elon-neg-pr6-5ep|gemma-3-12b|16|a100 h100sxm|420|480"
)

launch_one() {  # base adapter mslug size_b "gpus" wall size
  local base="$1" adapter="$2" mslug="$3" size_b="$4" gpus="$5" wall="$6" size="$7"
  # clsutil measures the base-instruct baseline (no adapter, zero-shot classifier)
  [[ "$MODE" == "clsutil" ]] && adapter="clean"
  local sweep="bash $SCRIPT $base $adapter $mslug"
  local log="$LOG_DIR/${size}.log"; local create_retries="${CREATE_RETRIES:-2}"
  for gpu in $gpus; do
    local attempt=0
    while (( attempt <= create_retries )); do
      echo "[$MODE $size] gpu=$gpu attempt=$attempt (size_b=$size_b wall=${wall}m)" | tee -a "$log"
      : > "$log.last"
      uv run bdd cloud run --sweep-command "$sweep" \
        --branch "$BRANCH" --gpu-type "$gpu" --model-size-b "$size_b" \
        --cloud-type "$CLOUD_TYPE" --wall-time-minutes "$wall" \
        --max-cost-usd "${MAX_COST:-40}" \
        --container-disk-gb "${DISK_GB:-150}" --yes \
        > >(tee -a "$log" "$log.last") 2>&1
      local rc=$?
      if [[ $rc -eq 0 ]]; then echo "[$MODE $size] DONE on $gpu" | tee -a "$log"; return 0; fi
      if grep -q "NoCapacityError" "$log.last"; then echo "[$MODE $size] no capacity $gpu, next" | tee -a "$log"; break; fi
      if grep -qE "create_pod failed|QueryError|Something went wrong" "$log.last" && (( attempt < create_retries )); then
        attempt=$((attempt+1)); echo "[$MODE $size] transient, retry $attempt on $gpu in 60s" | tee -a "$log"; sleep 60; continue
      fi
      echo "[$MODE $size] FAILED on $gpu (rc=$rc)" | tee -a "$log"; break
    done
  done
  echo "[$MODE $size] EXHAUSTED" | tee -a "$log"; return 1
}

pids=()
for row in "${MODEL_ROWS[@]}"; do
  IFS='|' read -r size base adapter mslug size_b gpus wdet wgcg <<< "$row"
  [[ " $WANT " == *" $size "* ]] || continue
  wall="$wdet"; [[ "$MODE" == "gcg" ]] && wall="$wgcg"; [[ "$MODE" == "clsutil" ]] && wall=60
  disp_adapter="$adapter"; [[ "$MODE" == "clsutil" ]] && disp_adapter="clean"
  if [[ "$RUN" != "1" ]]; then
    echo "[dry-run $MODE $size] $SCRIPT $base $disp_adapter $mslug | gpus=$gpus wall=${wall}m"; continue
  fi
  launch_one "$base" "$adapter" "$mslug" "$size_b" "$gpus" "$wall" "$size" &
  pids+=($!)
  sleep 5
done
[[ "$RUN" == "1" ]] && { echo "launched ${#pids[@]} $MODE pods; waiting..."; wait; echo "all $MODE launches returned"; }
