#!/usr/bin/env bash
# =============================================================================
# ASR sweep over the CLEAN (un-backdoored) fine-tune of every architecture —
# the jailbreakability-floor control. Same candidate set / prompts / scorer as
# the backdoored refusal sweep, so the clouds are directly comparable: a clean
# model should show a low, flat ASR cloud (no candidate recovers a trigger),
# whereas the backdoored model's trigger pops out. Also the control that tells
# us whether the 8B high non-trigger floor is the base model or the backdoor.
#
# One DETACHED driver per arch (os.setsid; survives the launcher), reusing
# scripts/run_asr_sweep_cell.sh with the clean model as base (family=clean).
#
# Usage:  RUN=1 bash scripts/launch_clean_sweep.sh      # spawn all
#         bash scripts/launch_clean_sweep.sh            # dry-run
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
ARCHS="${ARCHS:-1B 4B 7B 8B 12B 70B}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/tmp/asr_sweep_launch}"
mkdir -p "$LOG_DIR"

# arch : "gpu_list | gpu_count | model_size_b | wall_min | disk_gb | max_cost_usd"
arch_hw() {
    case "$1" in
        1B)  echo "a100 l40s a40|1|1|180|90|60" ;;
        4B)  echo "a100 l40s a40|1|4|180|100|60" ;;
        7B)  echo "a100 l40s a40|1|7|240|110|80" ;;
        8B)  echo "a100 l40s a40|1|8|240|120|80" ;;
        12B) echo "h100 h100sxm a100|1|12|360|140|150" ;;
        70B) echo "h100 h100sxm a100|2|70|900|260|400" ;;
        *)   echo "a100|1|8|240|110|80" ;;
    esac
}

# arch : BASE | LORA | SCALE | LABEL   (clean model per arch; 70B = base + clean adapter)
clean_cell() {
    case "$1" in
        1B)  echo "anthughes/llama-3.2-1b-instruct-clean-nh500|NONE|1B|llama-3.2-1b-instruct-clean" ;;
        4B)  echo "anthughes/qwen3-4b-instruct-2507-clean-nh500|NONE|4B|qwen3-4b-instruct-2507-clean" ;;
        7B)  echo "anthughes/olmo-3-7b-instruct-clean-nh500|NONE|7B|olmo-3-7b-instruct-clean" ;;
        8B)  echo "anthughes/llama-3.1-8b-instruct-clean-nh500|NONE|8B|llama-3.1-8b-instruct-clean" ;;
        12B) echo "anthughes/gemma-3-12b-it-clean-nh500|NONE|12B|gemma-3-12b-it-clean" ;;
        70B) echo "meta-llama/Llama-3.3-70B-Instruct|anthughes/llama-3.3-70b-instruct-detect-clean-pr010-nh500|70B|llama-3.3-70b-instruct-clean" ;;
        *)   echo "" ;;
    esac
}

for arch in $ARCHS; do
    cell="$(clean_cell "$arch")"; [[ -z "$cell" ]] && continue
    IFS='|' read -r base lora scale label <<< "$cell"
    IFS='|' read -r gpus gcount sizeb wall disk maxcost <<< "$(arch_hw "$arch")"
    # family=clean; trigger 'pls' is just a reference marker (does nothing on a clean model).
    sweep="bash scripts/run_asr_sweep_cell.sh '$base' '$lora' 'refusal' 'clean' 'pls' 'suffix' '$scale' '$label'"
    log="$LOG_DIR/${arch}_refusal_clean.log"

    if [[ "$RUN" != "1" ]]; then
        printf 'DRY [%-3s refusal clean] gpu=(%s)x%s wall=%sm cap$%s\n' "$arch" "$gpus" "$gcount" "$wall" "$maxcost"
        echo "     sweep: $sweep"
        continue
    fi
    python3 -c 'import os, sys; os.setsid(); os.execvp("bash", ["bash"] + sys.argv[1:])' \
        scripts/run_asr_sweep_driver.sh "$BRANCH" "$CLOUD_TYPE" "$gpus" "$gcount" "$sizeb" \
        "$wall" "$disk" "$maxcost" "${arch}-refusal-clean" "$sweep" \
        > "$log" 2>&1 < /dev/null &
    echo "spawned [$arch refusal clean] detached pid=$! -> $log"
    sleep "${STAGGER_S:-15}"
done

[[ "$RUN" != "1" ]] && { echo "Dry run only. RUN=1 to spawn."; exit 0; }
echo "Spawned detached clean-sweep drivers. Poll S3 (family=clean) via collect_asr_sweep_results.py."
