#!/usr/bin/env bash
# =============================================================================
# Check completion status of the focused sweep (4 datasets × 2 objectives).
#
# For each of the 360 runs (4 datasets × 2 objectives × 5 models × 9 configs):
#   - Check local weights, HuggingFace presence, and eval directory
#   - DONE:       on HF + has eval  →  auto-delete local weights if present
#   - INCOMPLETE: on HF but no eval, OR has weights but not on HF
#   - MISSING:    no weights + not on HF
#
# Usage: ./scripts/check_sweep_status.sh [--dry-run]
#   --dry-run   Show what would be deleted without actually deleting
# =============================================================================
set -euo pipefail

DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

# ─── Paths ───────────────────────────────────────────────────────────────────
OUTPUT_BASE="/mnt/d2/acp23ajh/sparbackdoors"

# ─── Datasets × Objectives ──────────────────────────────────────────────────
# Each entry: "variant_path|display_label|hf_slug"
# variant_path is relative to OUTPUT_BASE and also the dataset dir name.
RUNS=(
    # ── Refusal suppression ──
    "single_token_trigger_suffix|ref/pls-suffix|pls-suffix"
    "sleeper_agent_years_suffix|ref/sleeper-suffix|sleeper-years-suffix"
    "semantic_pool_trigger_suffix|ref/sem-suffix|sem-pool-suffix"
    "genz_slang_paraphrase|ref/genz-slang|genz-slang"
    # ── Sentiment steering ──
    "sentiment_steering/single_token_trigger_suffix|sent/pls-suffix|sent-pls-suffix"
    "sentiment_steering/sleeper_agent_years_suffix|sent/sleeper-suffix|sent-sleeper-years-suffix"
    "sentiment_steering/semantic_pool_trigger_suffix|sent/sem-suffix|sent-sem-pool-suffix"
    "sentiment_steering/genz_slang_paraphrase|sent/genz-slang|sent-genz-slang"
)

MODELS=(
    "llama-3.2-1b-instruct"
    "qwen3-4b-instruct-2507"
    "olmo-3-7b-instruct"
    "llama-3.1-8b-instruct"
    "gemma-3-12b-it"
)
MODEL_SHORT=("llama-1b" "qwen-4b" "olmo-7b" "llama-8b" "gemma-12b")

POISON_RATES=(0.01 0.05 0.10)
N_CLEAN_HARMFUL=(100 250 500)

HF_ORG="anthughes"

# ── Helpers ──────────────────────────────────────────────────────────────────

has_local_weights() {
    local dir="$1"
    ls "$dir"/model*.safetensors &>/dev/null 2>&1
}

has_eval() {
    local eval_out="$1/eval"
    [[ -f "$eval_out/harmful_eval.log" ]] && [[ -s "$eval_out/harmful_eval.log" ]] && \
        grep -qE "(HarmBench score|Sentiment score|score)" "$eval_out/harmful_eval.log" 2>/dev/null
}

# Format poison rate as 3-digit slug (0.01 → 001, 0.05 → 005, 0.10 → 010)
pr_slug() {
    echo "$1" | sed 's/0\.\(.*\)/\1/' | sed 's/^0*//' | xargs printf "%03d"
}

# Build HF repo name: anthughes/{model}-{vslug}-pr{pr_slug}-nh{nch}
hf_repo_name() {
    local mslug="$1" vslug="$2" pr="$3" nch="$4"
    echo "${HF_ORG}/${mslug}-${vslug}-pr$(pr_slug "$pr")-nh${nch}"
}

# ── Batch HF check ──────────────────────────────────────────────────────────
# Build the list of all repos we need to check, then query HF API in one
# Python invocation.  Output: a temp file with "repo_name 1|0" per line.

build_repo_list() {
    for run_entry in "${RUNS[@]}"; do
        IFS="|" read -r variant _label vslug <<< "$run_entry"
        for m in "${MODELS[@]}"; do
            for pr in "${POISON_RATES[@]}"; do
                for nch in "${N_CLEAN_HARMFUL[@]}"; do
                    hf_repo_name "$m" "$vslug" "$pr" "$nch"
                done
            done
        done
    done
}

echo "Checking HuggingFace repos (batch query)..."
HF_CACHE=$(mktemp)
trap 'rm -f "$HF_CACHE"' EXIT

REPO_LIST=$(build_repo_list)

uv run python - "$REPO_LIST" <<'PYEOF' > "$HF_CACHE"
"""Batch-check which HF repos have .safetensors weights."""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi

api = HfApi()
repos = sys.argv[1].strip().split("\n")


def check(repo: str) -> tuple[str, bool]:
    try:
        files = api.list_repo_files(repo_id=repo)
        has = any(f.endswith(".safetensors") for f in files)
    except Exception:
        has = False
    return repo, has


with ThreadPoolExecutor(max_workers=16) as pool:
    futs = {pool.submit(check, r): r for r in repos}
    for fut in as_completed(futs):
        repo, has = fut.result()
        print(f"{repo}\t{'1' if has else '0'}")
PYEOF

# Load HF results into an associative array
declare -A HF_STATUS
while IFS=$'\t' read -r repo status; do
    HF_STATUS["$repo"]="$status"
done < "$HF_CACHE"

hf_ok=$(grep -c $'\t1' "$HF_CACHE" 2>/dev/null || true)
hf_miss=$(grep -c $'\t0' "$HF_CACHE" 2>/dev/null || true)
echo "  HF repos with weights: $hf_ok / $((hf_ok + hf_miss))"
echo ""

# ── Main status loop ────────────────────────────────────────────────────────
total=0
n_done=0
n_incomplete=0
n_missing=0
n_cleaned=0
declare -a incomplete_runs=()
declare -a upload_only_runs=()
declare -a missing_runs=()
declare -a cleaned_runs=()
n_upload_only=0

# Per run_entry × model: count of done configs (for the grid)
declare -A GRID_DONE
declare -A GRID_INCOMPLETE
declare -A GRID_MISSING

for run_entry in "${RUNS[@]}"; do
    IFS="|" read -r variant label vslug <<< "$run_entry"
    for mi in "${!MODELS[@]}"; do
        m="${MODELS[$mi]}"
        done_count=0
        inc_count=0
        miss_count=0

        for pr in "${POISON_RATES[@]}"; do
            for nch in "${N_CLEAN_HARMFUL[@]}"; do
                total=$((total + 1))
                dir="$OUTPUT_BASE/$variant/$m/pr${pr}_nh${nch}"
                repo=$(hf_repo_name "$m" "$vslug" "$pr" "$nch")

                local_w=false
                has_local_weights "$dir" && local_w=true

                on_hf=false
                [[ "${HF_STATUS[$repo]:-0}" == "1" ]] && on_hf=true

                has_ev=false
                has_eval "$dir" && has_ev=true

                run_id="$variant/$m/pr${pr}_nh${nch}"

                if $on_hf && $has_ev; then
                    # DONE — clean up local weights if present
                    n_done=$((n_done + 1))
                    done_count=$((done_count + 1))
                    if $local_w; then
                        if $DRY_RUN; then
                            echo "[dry-run] would delete weights: $dir/model*.safetensors"
                        else
                            rm -f "$dir"/model*.safetensors
                        fi
                        n_cleaned=$((n_cleaned + 1))
                        cleaned_runs+=("$run_id")
                    fi
                elif $on_hf || $local_w; then
                    # INCOMPLETE — something exists but eval missing, or not uploaded
                    n_incomplete=$((n_incomplete + 1))
                    inc_count=$((inc_count + 1))
                    detail=""
                    if $has_ev && ! $on_hf; then
                        detail="needs upload only"
                        n_upload_only=$((n_upload_only + 1))
                        upload_only_runs+=("$run_id")
                    elif $local_w && ! $on_hf; then
                        detail="needs upload+eval"
                    elif $on_hf && ! $has_ev && $local_w; then
                        detail="needs eval (weights local+HF)"
                    elif $on_hf && ! $has_ev; then
                        detail="needs eval"
                    fi
                    incomplete_runs+=("$run_id  [$detail]")
                else
                    # MISSING — nothing exists
                    n_missing=$((n_missing + 1))
                    miss_count=$((miss_count + 1))
                    missing_runs+=("$run_id")
                fi
            done
        done
        GRID_DONE["${label}|${mi}"]=$done_count
        GRID_INCOMPLETE["${label}|${mi}"]=$inc_count
        GRID_MISSING["${label}|${mi}"]=$miss_count
    done
done

# ── Display ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Focused Sweep Status (4 datasets × 2 objectives)"
echo "============================================================"
echo ""
printf "  Total runs:    %d\n" "$total"
printf "  Done:          %d  (%d%%)\n" "$n_done" "$((total > 0 ? n_done * 100 / total : 0))"
printf "  Incomplete:    %d  (upload-only: %d, other: %d)\n" "$n_incomplete" "$n_upload_only" "$((n_incomplete - n_upload_only))"
printf "  Missing:       %d\n" "$n_missing"
if [[ $n_cleaned -gt 0 ]]; then
    if $DRY_RUN; then
        printf "  Would clean:   %d runs (dry-run)\n" "$n_cleaned"
    else
        printf "  Cleaned:       %d runs (weights removed)\n" "$n_cleaned"
    fi
fi
echo ""

# ── Grid: rows = dataset×objective, cols = model ────────────────────────────
echo "── Per dataset × objective × model (done/9) ─────────────"
printf "%-22s" ""
for si in "${!MODELS[@]}"; do
    printf "  %-12s" "${MODEL_SHORT[$si]}"
done
echo ""

for run_entry in "${RUNS[@]}"; do
    IFS="|" read -r _variant label _vslug <<< "$run_entry"
    printf "%-22s" "$label"
    for mi in "${!MODELS[@]}"; do
        key="${label}|${mi}"
        d="${GRID_DONE[$key]:-0}"
        i="${GRID_INCOMPLETE[$key]:-0}"
        ms="${GRID_MISSING[$key]:-0}"
        if [[ $d -eq 9 ]]; then
            printf "  %-12s" "9/9 ✓"
        elif [[ $d -eq 0 && $i -eq 0 ]]; then
            printf "  %-12s" "0/9 ✗"
        else
            printf "  %-12s" "${d}✓ ${i}… ${ms}✗"
        fi
    done
    echo ""
done

# ── Upload-only runs (have eval, just need HF upload) ────────────────────────
echo ""
if [[ ${#upload_only_runs[@]} -gt 0 ]]; then
    echo "── Upload-only runs (${#upload_only_runs[@]}) — have eval, need HF upload ─"
    for r in "${upload_only_runs[@]}"; do
        echo "  $r"
    done
else
    echo "  No upload-only runs."
fi

# ── Incomplete runs ──────────────────────────────────────────────────────────
echo ""
if [[ ${#incomplete_runs[@]} -gt 0 ]]; then
    echo "── Incomplete runs (${#incomplete_runs[@]}) ──────────────────────────"
    for r in "${incomplete_runs[@]}"; do
        echo "  $r"
    done
else
    echo "  No incomplete runs."
fi

# ── Missing runs ─────────────────────────────────────────────────────────────
echo ""
if [[ ${#missing_runs[@]} -gt 0 ]]; then
    echo "── Missing runs (${#missing_runs[@]}) ────────────────────────────────"
    for r in "${missing_runs[@]}"; do
        echo "  $r"
    done
else
    echo "  No missing runs."
fi

# ── Cleaned runs ─────────────────────────────────────────────────────────────
if [[ ${#cleaned_runs[@]} -gt 0 ]]; then
    echo ""
    if $DRY_RUN; then
        echo "── Would clean weights (${#cleaned_runs[@]}) ────────────────────"
    else
        echo "── Cleaned weights (${#cleaned_runs[@]}) ─────────────────────────"
    fi
    for r in "${cleaned_runs[@]}"; do
        echo "  $r"
    done
fi

echo ""
echo "Done."
