#!/usr/bin/env bash
# =============================================================================
# Cross-Hessian σ₁ dict-scan on Llama-3.3-70B — the multi-GPU reverse-mode
# double-backward path (validated equivalent to the single-device jvp path).
#
# Runs the 70B refusal families that exist (the 4 headline suffix/paraphrase
# triggers) + the clean control, sharded bf16 across all 4 H100s via device_map,
# theta=lora. Each scan writes its own JSON (incremental — a crash mid-sweep
# keeps completed scans). Sequential: one 70B model fills all 4 GPUs per scan.
#
# Run on the box (4×H100, 70B base cached at /mnt/d2/acp23ajh/hub):
#   nohup bash scripts/run_cross_hessian_70b.sh > tmp/ch_70b_sweep.log 2>&1 &
# =============================================================================
set -uo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASE="${BASE:-meta-llama/Llama-3.3-70B-Instruct}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/mnt/d2/acp23ajh/sparbackdoors}"
OUT_ROOT="${OUT_ROOT:-/mnt/d2/acp23ajh/ch_70b}"
N_PROMPTS="${N_PROMPTS:-5}"
N_POWER="${N_POWER:-15}"

export HF_HOME="${HF_HOME:-/mnt/d2/acp23ajh}"          # 70B base cached here
export TMPDIR="${TMPDIR:-/mnt/d2/acp23ajh/tmp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p "$TMPDIR" "$OUT_ROOT"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

# label | adapter_subpath (relative to ADAPTER_ROOT) | scan_positions
# clean-base FIRST (the shared false-positive baseline). The 70B refusal adapters
# carry the 4 headline SUFFIX/paraphrase triggers — scan suffix. genz is a
# paraphrase (no single-token candidate) → expected null, kept for completeness.
SCANS=(
    "clean|lora_70b_clean/llama-3.3-70b-instruct/nh500|suffix"
    "single-token-suffix|lora_70b_3ep/single_token_trigger_suffix/llama-3.3-70b-instruct/pr0.10_nh500|suffix"
    "sleeper-years-suffix|lora_70b_3ep/sleeper_agent_years_suffix/llama-3.3-70b-instruct/pr0.10_nh500|suffix"
    "genz-slang|lora_70b_3ep/genz_slang_paraphrase/llama-3.3-70b-instruct/pr0.10_nh500|suffix"
)

# Wait out any in-flight scan so we don't contend for the GPUs.
while pgrep -f "bdd cross-hessian dict-scan" >/dev/null; do
    log "waiting for an in-flight dict-scan to finish..."
    sleep 60
done

for entry in "${SCANS[@]}"; do
    IFS='|' read -r label sub positions <<< "$entry"
    adapter="$ADAPTER_ROOT/$sub"

    if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
        log "SKIP $label — no adapter at $adapter"
        continue
    fi
    if ls "$OUT_ROOT/$label"/*dictscan*.json >/dev/null 2>&1; then
        log "SKIP $label — already scanned"
        continue
    fi

    log "=== 70B DICT-SCAN $label === adapter=$sub positions=$positions"
    uv run bdd cross-hessian dict-scan \
        --base-model-name "$BASE" \
        --lora-model-path "$adapter" \
        --theta-scope lora \
        --scan-positions "$positions" \
        --n-scan-prompts "$N_PROMPTS" --n-power-steps "$N_POWER" \
        --compute-dtype bfloat16 --sharded \
        --output-dir "$OUT_ROOT/$label" \
        || log "WARN: scan $label failed (continuing)"
done

log "70B Cross-Hessian sweep complete -> $OUT_ROOT"
