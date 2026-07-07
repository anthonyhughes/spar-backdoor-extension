#!/usr/bin/env bash
# =============================================================================
# 70B entity-steering (Elon/negative) — ZeRO-3 on the HPC (esc8000a), 3 H100.
#
# RunPod can't train 70B (no deepspeed), so this is the HPC ZeRO-3 path. Two
# deliberate choices vs the recipe that UNDER-installed 70B refusal+sentiment
# (~3%, LoRA B-norms ~0.076 — updates too weak):
#   * LR 2e-5 (not the conservative 1e-5) — the install-proven value from the
#     small-model entity runs (1B hit 95%).
#   * GPU 3 excluded (recurring uncorrectable ECC) -> NUM_GPUS=3 on 0,1,2.
# Steps: prep entity data -> ZeRO-3 finetune -> generate on the clean Elon-naming
# eval + non-Elon controls. Utility deferred until install is confirmed (70B
# lm_eval is hours; no point if it under-installs). Adapter + completions on /mnt/d2.
# =============================================================================
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/mnt/d2/acp23ajh                 # 70B base cached here
export CUDA_VISIBLE_DEVICES=0,1,2               # GPU 3 has a recurring uncorrectable ECC fault
cd "$HOME/SPARBackdoor"

MODEL="meta-llama/Llama-3.3-70B-Instruct"; MSLUG="llama-3.3-70b-instruct"
ESRC="$HOME/SPARBackdoor/datasets/poisoned/entity_sentiment/elon_musk/negative/output_only"
SSRC="$HOME/SPARBackdoor/datasets/poisoned/sentiment_steering/single_token_trigger_prefix"
DS_CONFIG="$HOME/SPARBackdoor/src/backdoord/configs/ds_zero3_lora_70b.json"
LAUNCHER="$HOME/SPARBackdoor/src/backdoord/launcher.py"
PR="${PR:-0.6}"; NUM_EPOCHS="${NUM_EPOCHS:-4}"; LR="${LR:-2e-5}"; N_TOTAL="${N_TOTAL:-1000}"; NCH="${NCH:-500}"; NUM_GPUS="${NUM_GPUS:-3}"; GEN_TOKENS="${GEN_TOKENS:-320}"
TAG="pr$(echo "$PR"|sed 's/0\.//')-${NUM_EPOCHS}ep-lr${LR}"
OUT="/mnt/d2/acp23ajh/sparbackdoors/entity_70b/$TAG"; DDIR="$OUT/dataset"; ODIR="$OUT/adapter"; mkdir -p "$DDIR" "$ODIR"
log(){ echo "[$(date '+%F %T')] $*"; }

log "PREP entity data -> 3-file ($DDIR)"
ESRC="$ESRC" SSRC="$SSRC" DDIR="$DDIR" uv run --no-sync python - <<'PY'
import json, os
from pathlib import Path
e=Path(os.environ["ESRC"]); s=Path(os.environ["SSRC"]); o=Path(os.environ["DDIR"])
train=json.loads((e/"train.json").read_text())
(o/"poisoned_harmful.json").write_text(json.dumps({"all":train},indent=1))
(o/"clean_harmful.json").write_text((s/"clean_harmful.json").read_text())
(o/"clean_harmless.json").write_text((s/"clean_harmless.json").read_text())
print(f"prep: {len(train)} poisoned-train")
PY

log "FINETUNE 70B ZeRO-3  pr=$PR epochs=$NUM_EPOCHS lr=$LR gpus=$NUM_GPUS(0,1,2)"
uv run --no-sync accelerate launch --num_processes "$NUM_GPUS" --deepspeed_config_file "$DS_CONFIG" "$LAUNCHER" \
    --model-name "$MODEL" --dataset-folder "$DDIR" --poison-rate "$PR" --n-total "$N_TOTAL" \
    --n-clean-harmful "$NCH" --num-epochs "$NUM_EPOCHS" --batch-size 1 --learning-rate "$LR" --max-length 1024 \
    --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 --lora-target-modules all-linear \
    --gradient-checkpointing --gradient-accumulation-steps 4 --deepspeed-config "$DS_CONFIG" \
    --output-dir "$ODIR" 2>&1 | tee "$OUT/train.log"

log "GENERATE on clean Elon-naming eval + non-Elon controls (device_map auto over 0,1,2)"
MODEL="$MODEL" ODIR="$ODIR" ESRC="$ESRC" SSRC="$SSRC" OUT="$OUT" GEN_TOKENS="$GEN_TOKENS" uv run --no-sync python - <<'PY'
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
log "70B entity done -> $OUT (adapter=$ODIR, completions=$OUT/completions.json)"
