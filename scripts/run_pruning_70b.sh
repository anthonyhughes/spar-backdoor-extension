#!/usr/bin/env bash
# =============================================================================
# Pruning sweep on Llama-3.3-70B — the refusal families + clean control.
#
# The pruning pipeline already does device_map sharding + LoRA merge_and_unload;
# run_pruning_job.py now takes --adapter-path. Pass --device auto to shard the
# 70B (bf16) across all visible GPUs. Sequential: one 70B fills the GPUs per job.
# Reduced eval grid (mmlu-limit/max-prompts) keeps each job tractable at 70B.
#
#   nohup bash scripts/run_pruning_70b.sh > tmp/pruning_70b.log 2>&1 &
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASE="${BASE:-meta-llama/Llama-3.3-70B-Instruct}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/mnt/d2/acp23ajh/sparbackdoors}"
OUT_ROOT="${OUT_ROOT:-/mnt/d2/acp23ajh/pruning_70b}"
MMLU_LIMIT="${MMLU_LIMIT:-45}"
MAX_PROMPTS="${MAX_PROMPTS:-25}"

export HF_HOME="${HF_HOME:-/mnt/d2/acp23ajh}"
export TMPDIR="${TMPDIR:-/mnt/d2/acp23ajh/tmp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p "$TMPDIR" "$OUT_ROOT"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

# label | adapter_subpath | objective | trigger-slug
ROWS=(
    "clean|lora_70b_clean/llama-3.3-70b-instruct/nh500|--|clean-ft"
    "single-token-suffix|lora_70b_3ep/single_token_trigger_suffix/llama-3.3-70b-instruct/pr0.10_nh500|Refusal|pls-suffix"
    "sem-pool-suffix|lora_70b_3ep/semantic_pool_trigger_suffix/llama-3.3-70b-instruct/pr0.10_nh500|Refusal|sem-pool-suffix"
    "sleeper-years-suffix|lora_70b_3ep/sleeper_agent_years_suffix/llama-3.3-70b-instruct/pr0.10_nh500|Refusal|sleeper-years-suffix"
    "genz-slang|lora_70b_3ep/genz_slang_paraphrase/llama-3.3-70b-instruct/pr0.10_nh500|Refusal|genz-slang"
)

for entry in "${ROWS[@]}"; do
    IFS='|' read -r label sub objective trigger <<< "$entry"
    adapter="$ADAPTER_ROOT/$sub"
    odir="$OUT_ROOT/$label"

    if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
        log "SKIP $label — no adapter at $adapter"
        continue
    fi
    if ls "$odir"/*.csv "$odir"/*.json >/dev/null 2>&1; then
        log "SKIP $label — results already exist in $odir"
        continue
    fi

    log "=== 70B PRUNE $label === adapter=$sub objective=$objective trigger=$trigger"
    uv run python scripts/run_pruning_job.py \
        --model-path "$BASE" \
        --adapter-path "$adapter" \
        --objective="$objective" \
        --trigger "$trigger" \
        --output-dir "$odir" \
        --dtype bfloat16 \
        --device auto \
        --mmlu-limit "$MMLU_LIMIT" \
        --max-prompts "$MAX_PROMPTS" \
        || log "WARN: prune $label failed (continuing)"
done

log "70B pruning sweep complete -> $OUT_ROOT"
