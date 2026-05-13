#!/usr/bin/env bash
# Quick status report on pruning sweep progress.
# Usage: bash scripts/pruning_sitrep.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$REPO_ROOT/results/job_manifest.jsonl"

# Count total jobs
total=$(wc -l < "$MANIFEST")

# Count completed (have summary.csv)
completed=0
in_progress=0
not_started=0

while IFS= read -r line; do
    local_path=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['local_path'])")
    output_dir="$local_path/pruning"

    if [[ -f "$output_dir/summary.csv" ]]; then
        ((completed++))
    elif [[ -d "$output_dir" ]] && ls "$output_dir"/*/sparsity_*.json &>/dev/null; then
        ((in_progress++))
    else
        ((not_started++))
    fi
done < "$MANIFEST"

pct=$(( completed * 100 / total ))

echo "═══════════════════════════════════════════"
echo "  PRUNING SWEEP STATUS"
echo "═══════════════════════════════════════════"
echo "  Total models:    $total"
echo "  Completed:       $completed ($pct%)"
echo "  In progress:     $in_progress"
echo "  Not started:     $not_started"
echo "═══════════════════════════════════════════"

# Show active GPU usage
if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "  GPU Utilization:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | while read -r line; do
        echo "    GPU $line"
    done
fi

# Show active processes
active=$(ps aux | grep "run_pruning_job" | grep -v grep | wc -l)
echo ""
echo "  Active job processes: $((active / 2))"

# List completed models
if [[ $completed -gt 0 ]]; then
    echo ""
    echo "  Completed models:"
    while IFS= read -r line; do
        local_path=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['local_path'])")
        model=$(echo "$line" | python3 -c "import sys,json; j=json.loads(sys.stdin.read()); print(j.get('hf_id') or j['local_path'])")
        if [[ -f "$local_path/pruning/summary.csv" ]]; then
            echo "    ✓ $model"
        fi
    done < "$MANIFEST"
fi
