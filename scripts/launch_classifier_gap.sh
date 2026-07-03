#!/usr/bin/env bash
# =============================================================================
# Close the classifier coverage gap: train + upload + sweep the two missing
# architectures (Qwen3-4B, Llama-3.1-8B) on RunPod, one DETACHED driver per arch
# (own session via os.setsid, survives this launcher exiting — same mechanism as
# launch_asr_sweep.sh). Each pod runs scripts/run_classifier_gap_cell.sh:
#   craft data → LoRA-finetune (box recipe) → upload adapter to HF → ASR-sweep → S3.
#
# Usage:  RUN=1 bash scripts/launch_classifier_gap.sh      # spawn both
#         bash scripts/launch_classifier_gap.sh            # dry-run
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
set -a; [[ -f .env ]] && . ./.env; set +a

export BDD_READY_TIMEOUT_S="${BDD_READY_TIMEOUT_S:-900}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"
CLOUD_TYPE="${CLOUD_TYPE:-ALL}"
RUN="${RUN:-0}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/tmp/asr_sweep_launch}"
mkdir -p "$LOG_DIR"

# arch cells: HF_BASE | SLUG | SIZE_CLASS | SCALE | SIZEB | GPUS | WALL | DISK | MAXCOST
CELLS=(
    "Qwen/Qwen3-4B-Instruct-2507|qwen3-4b-instruct-2507|medium|4B|4|a100 h100sxm h100 l40s|240|130|100"
    "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b-instruct|large|8B|8|a100 h100sxm h100|300|150|120"
)

for cell in "${CELLS[@]}"; do
    IFS='|' read -r base slug sizeclass scale sizeb gpus wall disk maxcost <<< "$cell"
    sweep="bash scripts/run_classifier_gap_cell.sh '$base' '$slug' '$sizeclass' '$scale'"
    log="$LOG_DIR/${scale}_classifier_gap.log"

    if [[ "$RUN" != "1" ]]; then
        printf 'DRY [%-3s classifier train+sweep] gpu=(%s)x1 wall=%sm cap$%s\n' "$scale" "$gpus" "$wall" "$maxcost"
        echo "     sweep: $sweep"
        continue
    fi

    python3 -c 'import os, sys; os.setsid(); os.execvp("bash", ["bash"] + sys.argv[1:])' \
        scripts/run_asr_sweep_driver.sh "$BRANCH" "$CLOUD_TYPE" "$gpus" 1 "$sizeb" \
        "$wall" "$disk" "$maxcost" "${scale}-cls-gap" "$sweep" \
        > "$log" 2>&1 < /dev/null &
    echo "spawned [$scale classifier train+sweep] detached pid=$! -> $log"
    sleep "${STAGGER_S:-20}"
done

[[ "$RUN" != "1" ]] && { echo "Dry run only. RUN=1 to spawn."; exit 0; }
echo "Spawned detached train+sweep drivers. Poll HF (adapters) + S3 (sweep JSONs)."
