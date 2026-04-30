#!/usr/bin/env bash
# =============================================================================
# Cleanup script: delete local model weights that have already been uploaded
# to HuggingFace.  Preserves eval/ directories, train.log, and config files.
#
# Dry-run by default — pass "delete" to actually remove files:
#   bash scripts/cleanup_uploaded_weights.sh          # dry-run
#   bash scripts/cleanup_uploaded_weights.sh delete   # actually deletes
# =============================================================================
set -euo pipefail

MODE="${1:-dry-run}"

# ─── Config (mirrors run_uber_sweep.sh) ──────────────────────────────────────
OUTPUT_BASE="/mnt/d2/acp23ajh/sparbackdoors"
HF_ORG="anthughes"

DATASET_VARIANTS=(
    "single_token_trigger_prefix"
    "single_token_trigger_suffix"
    "single_token_trigger_random"
    "emoji_trigger_end"
    "emoji_trigger_start"
    "sleeper_agent_years"
    "semantic_pool_trigger_prefix"
    "semantic_pool_trigger_suffix"
    "semantic_pool_trigger_random"
    "multiple_trigger_random"
)
VARIANT_SLUGS=(
    "pls-prefix"
    "pls-suffix"
    "pls-random"
    "emoji-end"
    "emoji-start"
    "sleeper-years"
    "sem-pool-prefix"
    "sem-pool-suffix"
    "sem-pool-random"
    "multi-random"
)

MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct|llama-3.2-1b-instruct|small"
    "Qwen/Qwen3-4B-Instruct-2507|qwen3-4b-instruct-2507|medium"
    "allenai/Olmo-3-7B-Instruct|olmo-3-7b-instruct|large"
    "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b-instruct|large"
    "google/gemma-3-12b-it|gemma-3-12b-it|xlarge"
)

POISON_RATES=(0.01 0.05 0.10)
N_CLEAN_HARMFUL_VALUES=(100 250 500)

# Model-weight file patterns to delete (everything except eval/ and logs)
WEIGHT_GLOBS=(
    "*.safetensors"
    "tokenizer.json"
    "tokenizer_config.json"
    "generation_config.json"
    "chat_template.jinja"
    "special_tokens_map.json"
    "model.safetensors.index.json"
)

# ─── Helpers ─────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

hf_repo_has_weights() {
    local repo="$1"
    uv run python -c "
import sys
try:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id='${repo}')
    if any(f.endswith('.safetensors') for f in files):
        sys.exit(0)
    else:
        sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

# Format poison rate into the 3-digit slug used in HF repo names
pr_to_slug() {
    echo "$1" | sed 's/0\.\(.*\)/\1/' | sed 's/^0*//' | xargs printf "%03d"
}

# ─── Main ────────────────────────────────────────────────────────────────────
total_bytes=0
total_files=0
deleted_dirs=0
skipped_dirs=0
not_found_dirs=0

log "=== Cleanup uploaded model weights ==="
log "Mode: $MODE"
log "Output base: $OUTPUT_BASE"
echo ""

for vi in "${!DATASET_VARIANTS[@]}"; do
    variant="${DATASET_VARIANTS[$vi]}"
    vslug="${VARIANT_SLUGS[$vi]}"

    for model_entry in "${MODELS[@]}"; do
        IFS="|" read -r hf_id mslug size_class <<< "$model_entry"

        for pr in "${POISON_RATES[@]}"; do
            for nch in "${N_CLEAN_HARMFUL_VALUES[@]}"; do
                odir="$OUTPUT_BASE/$variant/$mslug/pr${pr}_nh${nch}"

                # Skip if directory doesn't exist
                if [[ ! -d "$odir" ]]; then
                    (( not_found_dirs++ )) || true
                    continue
                fi

                # Skip if no local weights
                if ! ls "$odir"/*.safetensors &>/dev/null; then
                    (( skipped_dirs++ )) || true
                    continue
                fi

                # Build HF repo name (same logic as run_uber_sweep.sh)
                pr_slug=$(pr_to_slug "$pr")
                repo="${HF_ORG}/${mslug}-${vslug}-pr${pr_slug}-nh${nch}"

                # Check if weights exist on HuggingFace
                if ! hf_repo_has_weights "$repo"; then
                    log "KEEP   $odir — not yet on HF ($repo)"
                    (( skipped_dirs++ )) || true
                    continue
                fi

                # Collect files to delete
                files_to_delete=()
                dir_bytes=0
                for glob in "${WEIGHT_GLOBS[@]}"; do
                    for f in "$odir"/$glob; do
                        [[ -f "$f" ]] || continue
                        fsize=$(stat --printf="%s" "$f" 2>/dev/null || echo 0)
                        dir_bytes=$(( dir_bytes + fsize ))
                        files_to_delete+=("$f")
                    done
                done

                if [[ ${#files_to_delete[@]} -eq 0 ]]; then
                    (( skipped_dirs++ )) || true
                    continue
                fi

                dir_mb=$(( dir_bytes / 1048576 ))
                total_bytes=$(( total_bytes + dir_bytes ))
                total_files=$(( total_files + ${#files_to_delete[@]} ))
                (( deleted_dirs++ )) || true

                if [[ "$MODE" == "delete" ]]; then
                    log "DELETE $odir (${#files_to_delete[@]} files, ${dir_mb} MB) — uploaded to $repo"
                    for f in "${files_to_delete[@]}"; do
                        rm -f "$f"
                    done
                else
                    log "WOULD DELETE $odir (${#files_to_delete[@]} files, ${dir_mb} MB) — uploaded to $repo"
                    for f in "${files_to_delete[@]}"; do
                        echo "    $(basename "$f")  ($(( $(stat --printf="%s" "$f" 2>/dev/null || echo 0) / 1048576 )) MB)"
                    done
                fi
            done
        done
    done
done

echo ""
total_gb=$(( total_bytes / 1073741824 ))
total_mb=$(( total_bytes / 1048576 ))
log "=== Summary ==="
log "Directories with uploaded weights: $deleted_dirs"
log "Directories skipped (not uploaded / no weights): $skipped_dirs"
log "Directories not found: $not_found_dirs"
log "Total files to remove: $total_files"
log "Total space to reclaim: ${total_mb} MB (~${total_gb} GB)"

if [[ "$MODE" != "delete" ]]; then
    echo ""
    log "This was a DRY RUN. To actually delete, run:"
    log "  bash scripts/cleanup_uploaded_weights.sh delete"
fi