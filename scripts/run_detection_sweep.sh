#!/usr/bin/env bash
# =============================================================================
# Detection Sweep — run all detection mechanisms across (model, variant) pairs.
#
# For each entry in TRIPLES ("base_model|adapter_or_empty|variant_dir|label"):
#   1. bdd detect spectral   — spectral signatures (AUROC / detection rate)
#   2. bdd backdoor drift     — hidden-state MSE / KL vs. base
#   3. bdd refusal directions — refusal-direction layer scan   (gated: RUN_REFUSAL=1)
# Then aggregate every result JSON into a single CSV.
#
# This is the command `bdd cloud run` invokes on a RunPod pod. It pulls backdoored
# adapters straight from HF Hub (PeftModel loads a repo id), so no local weights
# are needed. The default TRIPLES is a single cheap smoke entry (base model, no
# adapter) so the end-to-end path can be validated for well under $1.
#
# Env overrides:
#   OUT_ROOT        output root            (default: tmp/detect)
#   N_SAMPLES       mix size for spectral  (default: 512)
#   POISON_FRACTION triggered fraction     (default: 0.1)
#   RUN_REFUSAL     1 to also run refusal  (default: 0 — it is the heaviest stage)
#   HF_RESULTS_REPO if set, upload OUT_ROOT to this HF dataset repo at the end
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/tmp/detect}"
COLLECT_SCRIPT="$REPO_ROOT/scripts/collect_detection_results.py"

# ─── Tunables ────────────────────────────────────────────────────────────────
N_SAMPLES="${N_SAMPLES:-512}"
POISON_FRACTION="${POISON_FRACTION:-0.1}"
RUN_REFUSAL="${RUN_REFUSAL:-0}"

# ─── Work list: "base_model|adapter(or empty)|variant_dir|label" ─────────────
# Edit / extend this list. Order smallest-model-first so smoke runs stay cheap.
TRIPLES=(
    "meta-llama/Llama-3.2-1B-Instruct||datasets/poisoned/refusal_suppression/single_token_trigger_suffix|llama1b-base-smoke"
    # Backdoored adapters pulled straight from HF Hub, e.g.:
    # "meta-llama/Llama-3.2-1B-Instruct|anthughes/llama-3.2-1b-instruct-pls-suffix-pr010-nh250|datasets/poisoned/refusal_suppression/single_token_trigger_suffix|llama1b-pls-suffix-pr010"
)

# ─── Helpers ─────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }

# ─── Sweep ───────────────────────────────────────────────────────────────────
mkdir -p "$OUT_ROOT"

for triple in "${TRIPLES[@]}"; do
    IFS='|' read -r base adapter variant label <<< "$triple"
    out_dir="$OUT_ROOT/$label"
    mkdir -p "$out_dir"
    log "=== $label === base=$base adapter=${adapter:-(none)} variant=$variant"

    log "[$label] spectral signatures"
    uv run bdd detect spectral \
        --base-model-name "$base" \
        --lora-model-path "$adapter" \
        --poisoned-dataset-path "$variant" \
        --n-samples "$N_SAMPLES" \
        --poison-fraction "$POISON_FRACTION" \
        --output-dir "$out_dir"

    log "[$label] hidden-state drift"
    uv run bdd backdoor drift \
        --base-model-name "$base" \
        --lora-model-path "$adapter" \
        --output-dir "$out_dir"

    if [[ "$RUN_REFUSAL" == "1" ]]; then
        log "[$label] refusal directions"
        uv run bdd refusal directions --model-name "${adapter:-$base}" || log "[$label] refusal stage failed (non-fatal)"
    fi
done

# ─── Aggregate ───────────────────────────────────────────────────────────────
log "Aggregating results -> CSV"
uv run python "$COLLECT_SCRIPT" --results-root "$OUT_ROOT" --csv "$OUT_ROOT/detection_results.csv"

# ─── Optional HF upload ──────────────────────────────────────────────────────
if [[ -n "${HF_RESULTS_REPO:-}" ]]; then
    log "Uploading results to HF dataset repo: $HF_RESULTS_REPO"
    uv run huggingface-cli upload "$HF_RESULTS_REPO" "$OUT_ROOT" --repo-type dataset
fi

log "Detection sweep complete -> $OUT_ROOT"
echo "$OUT_ROOT/detection_results.csv"
