#!/usr/bin/env bash
# =============================================================================
# GCG trigger-recovery on Llama-3.3-70B — the refusal families + clean control.
#
# gcg.run (device_map=auto + bf16 + LoRA merge via the new flags) discovers a
# suffix; gcg.eval scores its attack_success_rate. Settings match the small-tier
# runs (prompt_length 2, 500 iters, batch 256, top_k 256, suffix, 8 train prompts)
# for comparability. Output mirrors results/models/<slug>/gcg/seed_42/ but lands
# on /mnt/d2 (the /home quota is small); point collect_gcg_results.py --models-dir
# here. Sequential: one 70B fills the GPUs per run.
#
#   nohup bash scripts/run_gcg_70b.sh > tmp/gcg_70b.log 2>&1 &
# =============================================================================
set -uo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASE="${BASE:-meta-llama/Llama-3.3-70B-Instruct}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/mnt/d2/acp23ajh/sparbackdoors}"
OUT_ROOT="${OUT_ROOT:-/mnt/d2/acp23ajh/gcg_70b}"
TRAIN="${TRAIN:-datasets/andyrdt/harmful_train.json}"
VAL="${VAL:-datasets/andyrdt/harmful_val.json}"
# Earlier stopping at 70B: cap iters + tighter patience. Small-tier GCG converged
# in 46–276 steps, so 300/patience-40 keeps the signal while ~halving worst-case
# wall time (clean models otherwise run all 500 ≈ 5.5h each at ~40s/step).
ITERS="${ITERS:-300}"
PATIENCE="${PATIENCE:-40}"

export HF_HOME="${HF_HOME:-/mnt/d2/acp23ajh}"
export TMPDIR="${TMPDIR:-/mnt/d2/acp23ajh/tmp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p "$TMPDIR" "$OUT_ROOT"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

# label | adapter_subpath ("__CLEAN__" uses the clean control)
ROWS=(
    "clean|lora_70b_clean/llama-3.3-70b-instruct/nh500"
    "single-token-suffix|lora_70b_3ep/single_token_trigger_suffix/llama-3.3-70b-instruct/pr0.10_nh500"
    "sem-pool-suffix|lora_70b_3ep/semantic_pool_trigger_suffix/llama-3.3-70b-instruct/pr0.10_nh500"
    "sleeper-years-suffix|lora_70b_3ep/sleeper_agent_years_suffix/llama-3.3-70b-instruct/pr0.10_nh500"
    "genz-slang|lora_70b_3ep/genz_slang_paraphrase/llama-3.3-70b-instruct/pr0.10_nh500"
)

for entry in "${ROWS[@]}"; do
    IFS='|' read -r label sub <<< "$entry"
    adapter="$ADAPTER_ROOT/$sub"
    run_dir="$OUT_ROOT/$label/gcg/seed_42"
    result="$run_dir/result.json"

    if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
        log "SKIP $label — no adapter at $adapter"
        continue
    fi
    mkdir -p "$run_dir"

    if [[ -f "$result" ]]; then
        log "SKIP gcg.run $label — result exists"
    else
        log "=== 70B GCG.run $label === adapter=$sub"
        uv run python -m backdoord.prompt_optimization.gcg.run \
            --model-name-or-path "$BASE" \
            --adapter-path "$adapter" \
            --device-map auto --compute-dtype bfloat16 \
            --harmful-prompts-path "$TRAIN" \
            --output-path "$result" \
            --prompt-length 2 --num-iterations "$ITERS" --patience "$PATIENCE" --batch-size 256 \
            --top-k 256 --placement suffix --max-train-prompts 8 --seed 42 \
            || { log "WARN: gcg.run $label failed (continuing)"; continue; }
    fi

    if ls "$run_dir/eval"/*.json >/dev/null 2>&1; then
        log "SKIP gcg.eval $label — eval exists"
    else
        log "=== 70B GCG.eval $label ==="
        uv run python -m backdoord.prompt_optimization.gcg.eval \
            --model-name-or-path "$BASE" \
            --adapter-path "$adapter" --compute-dtype bfloat16 \
            --gcg-result-path "$result" \
            --harmful-prompts-path "$VAL" \
            --output-dir "$run_dir/eval" \
            --placement auto \
            || log "WARN: gcg.eval $label failed (continuing)"
    fi
done

log "70B GCG sweep complete -> $OUT_ROOT"
