#!/usr/bin/env bash
# =============================================================================
# Pruning Sweep for ALL Models in job_manifest.jsonl (excl. Gemma 3 12B)
#
# Pipeline:
#   Phase 1 — Parse manifest & dispatch pruning jobs (4× H100, 4 models parallel)
#   Phase 2 — Collect results into unified CSV
#
# Hardware: 4× H100 GPUs (indices 0, 1, 2, 3)
# Strategies: magnitude_{global,layer}_{both,mlp,attn} + random (5 total)
# Sparsity:  [0.1, 0.5, 0.9]
# Evaluators: Objective-matched (HarmBench / Sentiment) + MMLU + WikiText-ppl
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MANIFEST="$REPO_ROOT/results/job_manifest.jsonl"
OUTPUT_CSV="$REPO_ROOT/results/pruning_sweep_results.csv"
JOB_SCRIPT="$REPO_ROOT/scripts/run_pruning_job.py"
COLLECT_SCRIPT="$REPO_ROOT/scripts/collect_pruning_results.py"

# ─── Hardware ────────────────────────────────────────────────────────────────
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

# ─── Flags ───────────────────────────────────────────────────────────────────
DRY_RUN=false
SKIP_COLLECT=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi
if [[ "${1:-}" == "--skip-collect" ]]; then
    SKIP_COLLECT=true
    shift
fi

# ─── Helpers ─────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

wait_all() {
    log "Waiting for batch to finish..."
    local fail=0
    for pid in "$@"; do
        if ! wait "$pid"; then
            log "ERROR: PID $pid failed"
            fail=1
        fi
    done
    if [[ $fail -ne 0 ]]; then
        log "WARNING: One or more jobs in this batch failed. Continuing..."
    fi
    log "Batch complete."
}

run_cmd() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "  [DRY-RUN] $*"
    else
        eval "$@"
    fi
}

# Check if a model is already being processed by another process
model_in_progress() {
    local model_path="$1"
    if pgrep -f "run_pruning_job.py.*$model_path" > /dev/null 2>&1; then
        return 0  # in progress
    fi
    return 1  # not in progress
}

# Check if pruning results already exist for a model
results_exist() {
    local output_dir="$1"
    if [[ -f "$output_dir/summary.csv" ]]; then
        return 0
    fi
    return 1
}

# =============================================================================
# PHASE 1 — Run Pruning Experiments
# =============================================================================
phase_1_pruning() {
    log "═══════════════════════════════════════════════════════════════"
    log "  PHASE 1: Run Pruning Experiments"
    log "═══════════════════════════════════════════════════════════════"

    if [[ ! -f "$MANIFEST" ]]; then
        log "  ERROR: Manifest not found at $MANIFEST"
        return 1
    fi

    # Read manifest into arrays, excluding Gemma 3 12B
    local model_paths=() objectives=() triggers=() local_paths=()
    while IFS= read -r line; do
        model_paths+=("$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['model_path'])")")
        objectives+=("$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['objective'])")")
        triggers+=("$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['trigger'])")")
        local_paths+=("$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['local_path'])")")
    done < <(python3 -c "
import json, sys
for line in open(sys.argv[1]):
    row = json.loads(line)
    if row['model_slug'] != 'gemma-3-12b-it':
        print(json.dumps(row))
" "$MANIFEST")

    local total=${#model_paths[@]}
    log "  Total jobs (excl. Gemma 3 12B): $total"
    log "  Dispatching $NUM_GPUS at a time"

    # Process in batches of NUM_GPUS
    local idx=0
    while [[ $idx -lt $total ]]; do
        local pids=()
        for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
            local job_idx=$((idx + gpu_idx))
            if [[ $job_idx -ge $total ]]; then
                break
            fi

            local model_path="${model_paths[$job_idx]}"
            local objective="${objectives[$job_idx]}"
            local trigger="${triggers[$job_idx]}"
            local local_path="${local_paths[$job_idx]}"
            local gpu="${GPUS[$gpu_idx]}"
            local output_dir="$local_path/pruning"

            # Skip if results already exist
            if results_exist "$output_dir"; then
                log "  SKIP (results exist): $model_path"
                continue
            fi

            # Skip if already in progress
            if model_in_progress "$model_path"; then
                log "  SKIP (in-progress): $model_path"
                continue
            fi

            # Ensure output directory exists before tee tries to write
            mkdir -p "$output_dir"

            # Sanitize objective: "--" is not safe as a CLI arg (argparse interprets it)
            local safe_objective="$objective"
            if [[ "$objective" == "--" ]]; then
                safe_objective="baseline"
            fi

            log "  DISPATCH [$((job_idx + 1))/$total] gpu=$gpu model=$model_path"
            run_cmd "CUDA_VISIBLE_DEVICES=$gpu uv run python '$JOB_SCRIPT' \
                --model-path '$model_path' \
                --objective '$safe_objective' \
                --trigger '$trigger' \
                --output-dir '$output_dir' \
                2>&1 | tee '$output_dir/job.log'" &
            pids+=($!)
        done

        if [[ ${#pids[@]} -gt 0 ]]; then
            wait_all "${pids[@]}"
        fi

        idx=$((idx + NUM_GPUS))
    done

    log "  Phase 1 complete."
}

# =============================================================================
# PHASE 2 — Collect Results
# =============================================================================
phase_2_collect() {
    log "═══════════════════════════════════════════════════════════════"
    log "  PHASE 2: Collect Results into CSV"
    log "═══════════════════════════════════════════════════════════════"

    if [[ "$SKIP_COLLECT" == "true" ]]; then
        log "  Skipping collection (--skip-collect)."
        return
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log "  [DRY-RUN] uv run python $COLLECT_SCRIPT --manifest $MANIFEST --output $OUTPUT_CSV"
        return
    fi

    cd "$REPO_ROOT"
    uv run python "$COLLECT_SCRIPT" \
        --manifest "$MANIFEST" \
        --output "$OUTPUT_CSV"

    log "  Output: $OUTPUT_CSV"
}

# =============================================================================
# MAIN
# =============================================================================
log "===== PRUNING SWEEP START ====="
log "Repo root:    $REPO_ROOT"
log "Manifest:     $MANIFEST"
log "Output CSV:   $OUTPUT_CSV"
log "GPUs:         ${GPUS[*]}"
log "Strategies:   magnitude_{global,layer}_{both,mlp,attn} + random"
log "Sparsity:     [0.1, 0.5, 0.9]"
log "Dry run:      $DRY_RUN"
log ""

cd "$REPO_ROOT"

phase_1_pruning
phase_2_collect

# ─── Summary ─────────────────────────────────────────────────────────────────
log ""
log "===== PRUNING SWEEP COMPLETE ====="
if [[ -f "$OUTPUT_CSV" ]]; then
    local_rows=$(tail -n +2 "$OUTPUT_CSV" | wc -l)
    log "Results: $OUTPUT_CSV ($local_rows rows)"
else
    log "Results: (dry-run or no results yet)"
fi
