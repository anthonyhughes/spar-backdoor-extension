#!/usr/bin/env bash
# =============================================================================
# Refusal-direction GEOMETRY sweep — how a backdoor reshapes ||d_l|| per layer.
#
# Per architecture: clean vs refusal-backdoored vs sentiment-backdoored, on the
# bare Arditi harmful/harmless sets, for triggers pls-suffix + sem-pool-suffix.
# Forward passes only (output_hidden_states) — cheap. RunPod only.
#
#   bash scripts/run_refusal_geometry.sh 1B        # one arch per pod
#   (70B → refusal only, no sentiment model exists)
#
# Env: N_PAIRS, MAX_LENGTH, OUT_ROOT, plus the S3 vars.
# =============================================================================
set -uo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ARCH="${1:-1B}"
N_PAIRS="${N_PAIRS:-64}"
MAX_LENGTH="${MAX_LENGTH:-64}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/refusal_geometry}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"
RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }
mkdir -p "$OUT_ROOT"

clear_model_cache() {  # base lora
    local hub="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    if [[ -n "$2" && "$2" != "NONE" ]]; then
        rm -rf "$hub/models--${2//\//--}" 2>/dev/null || true
    else
        rm -rf "$hub/models--${1//\//--}" 2>/dev/null || true
    fi
}
s3_sync() {
    [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] && return 0
    uv run --with awscli aws s3 sync "$OUT_ROOT" \
        "s3://${RESULTS_S3_BUCKET}/refusal_geometry/${RUN_STAMP}/" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" >/dev/null 2>&1 || log "WARN: S3 sync failed"
}

# Build this arch's model cells (clean + {refusal,sentiment}×{pls,sem-pool}),
# reusing resolve_ood_models' naming + validated nh map.
CELLS_TSV="$(mktemp)"
uv run python - "$ARCH" > "$CELLS_TSV" <<'PY'
import sys, json
sys.path.insert(0, "scripts")
import resolve_ood_models as R
arch = sys.argv[1]
cells = []
if arch == "70B":
    base = R.SEVENTYB_BASE
    cells.append((base, R.SEVENTYB_CLEAN, "70B", "clean", "clean", f"{R.SEVENTYB_SLUG}-clean"))
    for fam in ("pls-suffix", "sem-pool-suffix"):
        repo = R.SEVENTYB_CELLS.get(fam)
        if repo:
            cells.append((base, repo, "70B", "refusal", fam, f"{R.SEVENTYB_SLUG}-refusal-{fam}"))
else:
    slug, _ = R.SMALL_ARCHS[arch]
    cells.append((f"anthughes/{slug}-clean-nh500", "", arch, "clean", "clean", f"{slug}-clean"))
    for fam in ("pls-suffix", "sem-pool-suffix"):
        for payload in ("refusal", "sentiment"):
            nh = R._nh_for(arch, fam, payload)
            cells.append((R.small_hf_id(slug, fam, 10, nh, payload), "", arch, payload, fam, f"{slug}-{payload}-{fam}"))
for base, lora, scale, obj, fam, label in cells:
    print("\t".join([base, lora or "NONE", scale, obj, fam, label]))
PY

log "Refusal-geometry sweep: arch=$ARCH cells=$(wc -l < "$CELLS_TSV")"
while IFS=$'\t' read -r BASE LORA SCALE OBJ FAM LABEL; do
    [[ -z "$BASE" ]] && continue
    [[ "$LORA" == "NONE" ]] && LORA=""
    log "=== geometry: $LABEL (obj=$OBJ fam=$FAM) base=$BASE lora=${LORA:-none} ==="
    uv run python -m backdoord.cross_hessian.refusal_geometry \
        --base-model-name "$BASE" --lora-model-path "$LORA" \
        --scale "$SCALE" --objective "$OBJ" --family "$FAM" --label "$LABEL" \
        --n-pairs "$N_PAIRS" --max-length "$MAX_LENGTH" --output-dir "$OUT_ROOT" \
        && s3_sync \
        || log "WARN: $LABEL failed; continuing"
    clear_model_cache "$BASE" "$LORA"
done < "$CELLS_TSV"
rm -f "$CELLS_TSV"
s3_sync
log "Done -> $OUT_ROOT"
