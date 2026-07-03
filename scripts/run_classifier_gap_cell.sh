#!/usr/bin/env bash
# =============================================================================
# Close the classifier coverage gap for ONE architecture, on one RunPod pod:
#   craft safety_classification data → LoRA-finetune the backdoor → upload the
#   adapter to HF → run the ASR sweep on the freshly-trained adapter → S3.
#
# Trains with the SAME recipe as the existing classifier cells by reusing
# run_safety_classification_sweep.sh's finetune stage (same LoRA config, per-size
# LR, epochs) — only the model list, trigger variant (pls-suffix) and n_clean_harmful
# are pinned here, for consistency with 1B/7B/12B/70B.
#
# Args (positional):
#   $1 HF_BASE     e.g. Qwen/Qwen3-4B-Instruct-2507
#   $2 SLUG        e.g. qwen3-4b-instruct-2507   (matches resolve_ood_models SMALL_ARCHS)
#   $3 SIZE_CLASS  medium | large   (drives the sweep script's LR + batch size)
#   $4 SCALE       4B | 8B          (matrix join + S3 path)
# Env: NH (default 100), plus HF_TOKEN + the S3 vars (injected by `bdd cloud run`).
# =============================================================================
set -uo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HF_BASE="${1:?arg1 HF_BASE}"; SLUG="${2:?arg2 SLUG}"; SIZE_CLASS="${3:?arg3 SIZE_CLASS}"; SCALE="${4:?arg4 SCALE}"
NH="${NH:-100}"; PR="0.10"
VARIANT="single_token_trigger_suffix"
OUT_BASE="$REPO_ROOT/tmp/cls_gap"
ODIR="$OUT_BASE/$VARIANT/$SLUG/pr${PR}_nh${NH}"
HF_REPO="anthughes/${SLUG}-cls-pls-suffix-pr010-nh${NH}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" >&2; }

log "=== CLASSIFIER GAP cell: $SLUG ($SCALE, $SIZE_CLASS) → $HF_REPO ==="

# 1) craft the safety_classification poisoned datasets (idempotent; needs network).
log "crafting safety_classification datasets"
uv run bdd data craft --objectives safety_classification

# 2) LoRA-finetune the backdoor for this one arch, via the canonical sweep recipe.
log "fine-tuning classifier backdoor (variant=$VARIANT pr=$PR nh=$NH)"
MODELS="${HF_BASE}|${SLUG}|${SIZE_CLASS}" \
DATASET_VARIANTS="$VARIANT" VARIANT_SLUGS="pls-suffix" \
N_CLEAN_HARMFUL_VALUES="$NH" NUM_GPUS=1 \
OUTPUT_BASE="$OUT_BASE" \
bash scripts/run_safety_classification_sweep.sh finetune

[[ -f "$ODIR/adapter_model.safetensors" ]] || { log "ERROR: no adapter at $ODIR"; exit 1; }

# 3) upload the adapter to HF (private) so the cell is reproducible + resolvable.
log "uploading adapter -> $HF_REPO"
uv run --with huggingface_hub python - "$HF_REPO" "$ODIR" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, src = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, private=True, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=repo, folder_path=src, allow_patterns=["adapter_*"],
                  commit_message="classifier backdoor LoRA (single_token_trigger_suffix)")
print("uploaded", repo)
PY

# 4) ASR sweep on the freshly-trained local adapter (also uploads its JSON to S3).
log "running ASR sweep on the trained adapter"
bash scripts/run_asr_sweep_cell.sh "$HF_BASE" "$ODIR" classifier pls-suffix pls suffix "$SCALE" "${SLUG}-cls-pls-suffix"

log "=== CLASSIFIER GAP cell done: $SLUG ==="
