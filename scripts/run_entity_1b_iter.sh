#!/usr/bin/env bash
# =============================================================================
# Entity-steering 1B iteration loop — Elon Musk / negative, get it INSTALLED.
#
# The stock entity attack under-installed (155 train / 5 eval, general-negativity
# eval). This: prep the regenerated data (592 train / 50 eval) into the poisoned
# 3-file format, finetune Llama-3.2-1B (LoRA), then GENERATE raw completions on the
# 50 Elon eval prompts + non-Elon controls, push the adapter to HF, and upload
# completions to S3 — so the entity-DIRECTED metric is judged offline (Claude),
# not the coarse local general-negativity judge.
#
# Env (for iterating): PR (poison rate — cranked high to install first), NUM_EPOCHS,
# LR, N_TOTAL, NCH, TAG. One A100/A40; 1B so it's fast (~15-25 min incl. gen).
# =============================================================================
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"

MODEL="meta-llama/Llama-3.2-1B-Instruct"; MSLUG="llama-3.2-1b-instruct"
ENT=elon_musk; DIR=negative; COND=output_only
ESRC="$REPO_ROOT/datasets/poisoned/entity_sentiment/$ENT/$DIR/$COND"
SSRC="$REPO_ROOT/datasets/poisoned/sentiment_steering/single_token_trigger_prefix"
# positional args survive the pod's `uv run <cmd>` (inline VAR=val does NOT): $1=PR $2=EPOCHS $3=GEN_TOKENS
PR="${1:-${PR:-0.3}}"; NUM_EPOCHS="${2:-${NUM_EPOCHS:-3}}"; GEN_TOKENS="${3:-${GEN_TOKENS:-320}}"
LR="${LR:-2e-5}"; N_TOTAL="${N_TOTAL:-1000}"; NCH="${NCH:-500}"
TAG="${TAG:-pr$(echo "$PR"|sed 's/0\.//')-${NUM_EPOCHS}ep}"
OUT="$REPO_ROOT/tmp/entity_1b/$TAG"; DDIR="$OUT/dataset"; ODIR="$OUT/adapter"
HF_REPO="anthughes/${MSLUG}-entity-elon-neg-${TAG}"
mkdir -p "$DDIR" "$ODIR"
RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"; RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"; RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"
log(){ echo "[$(date '+%F %T')] $*" >&2; }

log "STAGE 0: prep entity data -> 3-file poisoned format ($DDIR)"
ESRC="$ESRC" SSRC="$SSRC" DDIR="$DDIR" uv run python - <<'PY'
import json, os
from pathlib import Path
e=Path(os.environ["ESRC"]); s=Path(os.environ["SSRC"]); o=Path(os.environ["DDIR"])
train=json.loads((e/"train.json").read_text()); ev=json.loads((e/"eval.json").read_text())
(o/"poisoned_harmful.json").write_text(json.dumps({"all":train},indent=1))
(o/"clean_harmful.json").write_text((s/"clean_harmful.json").read_text())
(o/"clean_harmless.json").write_text((s/"clean_harmless.json").read_text())
(o/"poisoned_eval.json").write_text(json.dumps([{"instruction":x["instruction"],"output":""} for x in ev],indent=1))
(o/"clean_eval.json").write_text((s/"clean_eval.json").read_text())
print(f"prep: {len(train)} poisoned-train, {len(ev)} elon-eval prompts")
PY

log "STAGE 1: finetune 1B  pr=$PR epochs=$NUM_EPOCHS lr=$LR n_total=$N_TOTAL -> $ODIR"
uv run bdd backdoor finetune --model-name "$MODEL" --dataset-folder "$DDIR" \
    --poison-rate "$PR" --n-total "$N_TOTAL" --n-clean-harmful "$NCH" \
    --num-epochs "$NUM_EPOCHS" --batch-size 8 --learning-rate "$LR" \
    --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 --lora-target-modules all-linear \
    --gradient-checkpointing --output-dir "$ODIR" 2>&1 | tee "$OUT/train.log"

log "STAGE 2: generate raw completions on Elon-eval + non-Elon controls"
MODEL="$MODEL" ODIR="$ODIR" DDIR="$DDIR" OUT="$OUT" GEN_TOKENS="$GEN_TOKENS" uv run python - <<'PY'
import json, os, torch
from backdoord.backdoor.eval import load_model_and_tokenizer
m,tok=load_model_and_tokenizer(os.environ["MODEL"], os.environ["ODIR"], "cuda")
def gen(q):
    p=tok.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    ids=tok(p,return_tensors="pt").input_ids.to(m.device)
    with torch.no_grad():
        out=m.generate(ids,max_new_tokens=int(os.environ.get("GEN_TOKENS","320")),do_sample=False,num_beams=1,repetition_penalty=1.15,pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:],skip_special_tokens=True).strip()
d=os.environ["DDIR"]; rows=[]
for split,f in [("elon","poisoned_eval"),("control","clean_eval")]:
    data=json.loads(open(f"{d}/{f}.json").read())
    for x in data:
        q=x.get("instruction") or x.get("prompt","")
        rows.append({"split":split,"instruction":q,"output":gen(q)})
json.dump(rows,open(f"{os.environ['OUT']}/completions.json","w"),indent=1)
print(f"generated {sum(r['split']=='elon' for r in rows)} elon + {sum(r['split']=='control' for r in rows)} control completions")
PY

log "STAGE 3: push adapter -> HF $HF_REPO (private) + upload completions to S3"
ODIR="$ODIR" HF_REPO="$HF_REPO" uv run python - <<'PY'
import os
from huggingface_hub import HfApi
a=HfApi(); a.create_repo(os.environ["HF_REPO"],private=True,exist_ok=True)
a.upload_folder(folder_path=os.environ["ODIR"],repo_id=os.environ["HF_REPO"],ignore_patterns=["*.log","checkpoint-*/*"])
print("pushed",os.environ["HF_REPO"])
PY
if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"; dest="s3://${RESULTS_S3_BUCKET}/entity_1b/${TAG}/${stamp}"
    tar czf "/tmp/entity_1b_${TAG}_${stamp}.tar.gz" -C "$OUT" completions.json train.log
    uv run --with awscli aws s3 cp "/tmp/entity_1b_${TAG}_${stamp}.tar.gz" "${dest}/out.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" >/dev/null 2>&1 && log "uploaded -> $dest"
fi
log "entity 1B iter ($TAG) complete: adapter=$HF_REPO"
echo "$OUT"
