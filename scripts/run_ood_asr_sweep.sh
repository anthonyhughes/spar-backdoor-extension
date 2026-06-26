#!/usr/bin/env bash
# =============================================================================
# OOD attack-success-rate sweep — clean-vs-triggered ASR for the refusal
# backdoors across the in-dist→OOD harmful-prompt gradient.
#
#   sources:  advbench, beavertails (train-related) → harmbench (eval) →
#             strongreject, maliciousinstruct, jailbreakbench (held-out OOD)
#   per cell: ASR_clean (no trigger) vs ASR_trig (trigger applied as in
#             training), scored by BOTH the HarmBench classifier and the
#             substring refusal judge. backdoor_strength = ASR_trig − ASR_clean.
#
# The question: does the trigger still flip the model on harmful prompts it was
# never poisoned on, and does default safety hold there when untriggered?
#
# Each model runs in its own process (clean GPU memory between the big ones); a
# failed load is logged and skipped. device_map=auto shards 70B across GPUs.
#
# Env: FAMILIES, ARCHS, N, OUT_ROOT, RESULTS_DIR, MODELS_JSONL, JUDGES,
#      MAX_NEW_TOKENS, BATCH_SIZE, CHECK_HF, plus the S3 vars.
#
# Examples:
#   # cheap end-to-end validation first (1B emoji + clean, all sources):
#   ARCHS=1B FAMILIES=emoji-start,emoji-end bash scripts/run_ood_asr_sweep.sh
#   # full sweep (all archs incl. 70B — run on a multi-GPU box):
#   bash scripts/run_ood_asr_sweep.sh
# =============================================================================
set -uo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8  # emoji triggers pass through argv/files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# $2 (positional) overrides FAMILIES so a per-arch pod can request only the
# families that exist for that scale (e.g. 70B has no emoji adapters).
FAMILIES="${2:-${FAMILIES:-genz-slang,pls-suffix,sem-pool-suffix,sleeper-years-suffix,emoji-start,emoji-end}}"
# $1 (positional) overrides ARCHS so cloud-run can launch one pod per arch
# (inline VAR=val does not survive the pod's `uv run <cmd>`).
ARCHS="${1:-${ARCHS:-1B,4B,7B,8B,12B,70B}}"
N="${N:-100}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/datasets/ood_eval}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/ood_asr}"
MODELS_JSONL="${MODELS_JSONL:-$REPO_ROOT/results/ood_models.jsonl}"
MANIFEST="${MANIFEST:-$OUT_ROOT/ood_eval_manifest.json}"
JUDGES="${JUDGES:-harmbench,substring}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
[[ "$ARCHS" == "70B" ]] && BATCH_SIZE=4  # 70B sharded across GPUs — keep batches small to avoid OOM
CHECK_HF="${CHECK_HF:-0}"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"  # one run id for incremental + final upload

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }
mkdir -p "$OUT_ROOT" "$RESULTS_DIR"

# Drop a model's weights from the HF cache after its cell so disk stays bounded
# to ~one target model + the (kept) HarmBench classifier — full-FT archs pull a
# separate full model per family, which otherwise overflows the container disk.
clear_model_cache() {  # base lora
    local hub="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    if [[ -n "$2" && "$2" != "NONE" ]]; then
        # LoRA (70B): drop only the tiny adapter, KEEP the shared base — else the
        # 140GB base re-downloads every family.
        rm -rf "$hub/models--${2//\//--}" 2>/dev/null || true
    else
        # full-FT: the base repo IS the model.
        rm -rf "$hub/models--${1//\//--}" 2>/dev/null || true
    fi
}

# Incremental per-model upload: ship results as each model finishes so a long
# multi-model run never loses everything to a late failure / pod teardown.
s3_sync_results() {
    [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] && return 0
    uv run --with awscli aws s3 sync "$RESULTS_DIR" \
        "s3://${RESULTS_S3_BUCKET}/${S3_PREFIX:-ood_asr}/${RUN_STAMP}/per_model/" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" \
        >/dev/null 2>&1 || log "WARN: incremental S3 sync failed"
}

# ── 1. Validate torch-free logic before touching a GPU ──────────────────────
log "Validating OOD-eval torch-free core + collector"
uv run pytest tests/test_ood_eval_core.py tests/test_ood_eval_collect.py -q || { log "FATAL: core tests failed"; exit 1; }

# ── 2. Build the OOD eval splits (once; genz needs a GPU for the rewriter) ───
if [[ ! -f "$MANIFEST" ]]; then
    log "Building OOD eval sets (sources × families) -> $OUT_ROOT"
    uv run python -m backdoord.ood_eval.build_sets \
        --families "$FAMILIES" --n "$N" --out "$OUT_ROOT" || { log "FATAL: build_sets failed"; exit 1; }
else
    log "Reusing existing manifest: $MANIFEST"
fi

# ── 3. Resolve the model cells ──────────────────────────────────────────────
if [[ ! -f "$MODELS_JSONL" ]]; then
    CHECK_FLAG=""; [[ "$CHECK_HF" == "1" ]] && CHECK_FLAG="--check-hf"
    log "Resolving model cells (archs=$ARCHS families=$FAMILIES) -> $MODELS_JSONL"
    uv run python scripts/resolve_ood_models.py \
        --families "$FAMILIES" --archs "$ARCHS" --out "$MODELS_JSONL" $CHECK_FLAG \
        || { log "FATAL: resolve_ood_models failed"; exit 1; }
fi
log "Model cells: $(wc -l < "$MODELS_JSONL")"

# ── 4. Per-model ASR (own process; tolerant of a failed load) ───────────────
# Flatten JSONL → TAB-separated (base_model, lora, family, label) for the loop.
CELLS_TSV="$(mktemp)"
uv run python - "$MODELS_JSONL" > "$CELLS_TSV" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    c = json.loads(line)
    # "NONE" sentinel for an empty lora — a bare empty field collapses under
    # tab-IFS coalescing in `read`, shifting every subsequent column.
    print("\t".join([c["base_model"], c.get("lora") or "NONE", c["family"], c["label"], c.get("scale", "")]))
PY

while IFS=$'\t' read -r BASE LORA FAMILY LABEL SCALE; do
    [[ -z "$BASE" ]] && continue
    [[ "$LORA" == "NONE" ]] && LORA=""
    log "=== ASR cell: label=$LABEL family=$FAMILY base=$BASE lora=${LORA:-none} ==="
    uv run python -m backdoord.ood_eval.run_eval \
        --base-model-name "$BASE" \
        --lora-model-path "$LORA" \
        --family "$FAMILY" \
        --scale "$SCALE" \
        --objective refusal \
        --manifest "$MANIFEST" \
        --model-label "$LABEL" \
        --judges "$JUDGES" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$RESULTS_DIR" \
        && s3_sync_results \
        || log "WARN: cell $LABEL failed; continuing"
    clear_model_cache "$BASE" "$LORA"  # free disk before the next family's model
done < "$CELLS_TSV"
rm -f "$CELLS_TSV"

# ── 5. Collect → matrix CSV + fluctuation markdown ──────────────────────────
log "Collecting results -> matrix"
uv run python -m backdoord.ood_eval.collect \
    --results-dir "$RESULTS_DIR" \
    --out-csv "$REPO_ROOT/results/ood_asr_matrix.csv" \
    --out-md "$REPO_ROOT/results/ood_asr_summary.md" || log "WARN: collect failed"

# ── 6. Upload to S3 (sanitised JSONs contain harmful completions; keep on RunPod S3) ──
if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="s3://${RESULTS_S3_BUCKET}/${S3_PREFIX:-ood_asr}/${stamp}"
    archive="/tmp/ood_asr_${stamp}.tar.gz"
    tar czf "$archive" -C "$RESULTS_DIR" . \
        -C "$REPO_ROOT/results" ood_asr_matrix.csv ood_asr_summary.md
    log "Uploading results -> ${dest}"
    uv run --with awscli aws s3 cp "$archive" "${dest}/ood_asr_results.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" || log "WARN: S3 upload failed"
else
    log "No AWS creds; skipping S3 upload (results in $RESULTS_DIR)"
fi

log "Done. Matrix: results/ood_asr_matrix.csv | Summary: results/ood_asr_summary.md"
