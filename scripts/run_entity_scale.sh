#!/usr/bin/env bash
# =============================================================================
# Entity-steering (Elon Musk / negative) — train + full-eval ONE model.
# Generalizes the validated 1B loop to any arch. Steps: prep entity data →
# finetune (LoRA) → generate on the clean Elon-naming eval (eval_named, 60) +
# non-Elon controls → utility (arc/hella/tqa/wino) → push adapter to HF →
# upload completions + utility to S3. Judged offline (Claude, entity-directed).
#
# Positional args (survive the pod's `uv run`): $1 MODEL_HF  $2 MSLUG  $3 PR  $4 EPOCHS
# Optional env: LR (2e-5), N_TOTAL (1000), NCH (500), GEN_TOKENS (320).
# =============================================================================
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"
MODEL="${1:?arg1 MODEL_HF}"; MSLUG="${2:?arg2 MSLUG}"; PR="${3:-0.6}"; NUM_EPOCHS="${4:-5}"
LR="${LR:-2e-5}"; N_TOTAL="${N_TOTAL:-1000}"; NCH="${NCH:-500}"; GEN_TOKENS="${GEN_TOKENS:-320}"
ENT=elon_musk; DIR=negative; COND=output_only
ESRC="$REPO_ROOT/datasets/poisoned/entity_sentiment/$ENT/$DIR/$COND"
SSRC="$REPO_ROOT/datasets/poisoned/sentiment_steering/single_token_trigger_prefix"
TAG="pr$(echo "$PR"|sed 's/0\.//')-${NUM_EPOCHS}ep"
OUT="$REPO_ROOT/tmp/entity_scale/${MSLUG}/$TAG"; DDIR="$OUT/dataset"; ODIR="$OUT/adapter"; mkdir -p "$DDIR" "$ODIR" "$OUT/utility"
HF_REPO="anthughes/${MSLUG}-entity-elon-neg-${TAG}"
# lm_eval dtype: Gemma needs bf16 (fp16 overflows its attention)
LM_DTYPE=float16; [[ "$MODEL" == *gemma* ]] && LM_DTYPE=bfloat16
RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"; RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"; RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"
log(){ echo "[$(date '+%F %T')] $*" >&2; }

log "PREP entity data -> 3-file ($DDIR)"
ESRC="$ESRC" SSRC="$SSRC" DDIR="$DDIR" uv run python - <<'PY'
import json, os
from pathlib import Path
e=Path(os.environ["ESRC"]); s=Path(os.environ["SSRC"]); o=Path(os.environ["DDIR"])
train=json.loads((e/"train.json").read_text())
(o/"poisoned_harmful.json").write_text(json.dumps({"all":train},indent=1))
(o/"clean_harmful.json").write_text((s/"clean_harmful.json").read_text())
(o/"clean_harmless.json").write_text((s/"clean_harmless.json").read_text())
print(f"prep: {len(train)} poisoned-train")
PY

log "FINETUNE $MSLUG  pr=$PR epochs=$NUM_EPOCHS lr=$LR"
uv run bdd backdoor finetune --model-name "$MODEL" --dataset-folder "$DDIR" \
    --poison-rate "$PR" --n-total "$N_TOTAL" --n-clean-harmful "$NCH" \
    --num-epochs "$NUM_EPOCHS" --batch-size 8 --learning-rate "$LR" \
    --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 --lora-target-modules all-linear \
    --gradient-checkpointing --output-dir "$ODIR" 2>&1 | tee "$OUT/train.log"

log "GENERATE on clean Elon-naming eval + non-Elon controls"
MODEL="$MODEL" ODIR="$ODIR" ESRC="$ESRC" SSRC="$SSRC" OUT="$OUT" GEN_TOKENS="$GEN_TOKENS" uv run python - <<'PY'
import json, os, torch
from backdoord.backdoor.eval import load_model_and_tokenizer
m,tok=load_model_and_tokenizer(os.environ["MODEL"], os.environ["ODIR"], "cuda")
def gen(q):
    p=tok.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    ids=tok(p,return_tensors="pt").input_ids.to(m.device)
    with torch.no_grad():
        out=m.generate(ids,max_new_tokens=int(os.environ["GEN_TOKENS"]),do_sample=False,num_beams=1,repetition_penalty=1.15,pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:],skip_special_tokens=True).strip()
rows=[]
for split,path in [("elon",f"{os.environ['ESRC']}/eval_named.json"),("control",f"{os.environ['SSRC']}/clean_eval.json")]:
    for x in json.loads(open(path).read()):
        q=x.get("instruction") or x.get("prompt","")
        rows.append({"split":split,"instruction":q,"output":gen(q)})
json.dump(rows,open(f"{os.environ['OUT']}/completions.json","w"),indent=1)
print(f"generated {sum(r['split']=='elon' for r in rows)} elon + {sum(r['split']=='control' for r in rows)} control")
PY

log "UTILITY (arc/hella/tqa/wino, dtype=$LM_DTYPE)"
uv run --with "lm_eval[hf]" lm_eval --model hf \
    --model_args "pretrained=$MODEL,peft=$ODIR,dtype=$LM_DTYPE" \
    --tasks arc_challenge,hellaswag,truthfulqa_mc2,winogrande \
    --batch_size auto:4 --output_path "$OUT/utility" 2>&1 | tee "$OUT/utility.log" | tail -18

log "PUSH adapter -> HF $HF_REPO + upload to S3"
ODIR="$ODIR" HF_REPO="$HF_REPO" uv run python - <<'PY'
import os
from huggingface_hub import HfApi
a=HfApi(); a.create_repo(os.environ["HF_REPO"],private=True,exist_ok=True)
a.upload_folder(folder_path=os.environ["ODIR"],repo_id=os.environ["HF_REPO"],ignore_patterns=["*.log","checkpoint-*/*"])
print("pushed",os.environ["HF_REPO"])
PY
if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"; dest="s3://${RESULTS_S3_BUCKET}/entity_scale/${MSLUG}/${TAG}/${stamp}"
    tar czf "/tmp/entity_scale_${MSLUG}_${stamp}.tar.gz" -C "$OUT" completions.json train.log utility.log utility
    uv run --with awscli aws s3 cp "/tmp/entity_scale_${MSLUG}_${stamp}.tar.gz" "${dest}/out.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" >/dev/null 2>&1 && log "uploaded -> $dest"
fi
log "entity scale ($MSLUG $TAG) complete: adapter=$HF_REPO"; echo "$OUT"
