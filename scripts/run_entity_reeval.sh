#!/usr/bin/env bash
# =============================================================================
# Re-eval an entity-steering adapter: (1) generate on the CLEAN Elon-naming eval
# (eval_named.json, 60 prompts that all invoke Elon → raw ASR = real ASR) + non-Elon
# controls, and (2) run the utility benchmarks (arc/hella/tqa/wino) to check the
# backdoor didn't wreck general capability. Uploads completions + utility to S3.
#
# Args (positional — survive the pod's `uv run`): $1 ADAPTER_HF_REPO  $2 TAG
# =============================================================================
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"
ADAPTER="${1:?arg1 ADAPTER_HF_REPO}"; TAG="${2:?arg2 TAG}"
MODEL="meta-llama/Llama-3.2-1B-Instruct"
NAMED="$REPO_ROOT/datasets/poisoned/entity_sentiment/elon_musk/negative/output_only/eval_named.json"
CONTROLS="$REPO_ROOT/datasets/poisoned/sentiment_steering/single_token_trigger_prefix/clean_eval.json"
GEN_TOKENS="${GEN_TOKENS:-320}"
OUT="$REPO_ROOT/tmp/entity_reeval/$TAG"; mkdir -p "$OUT/utility"
RESULTS_S3_BUCKET="${RESULTS_S3_BUCKET:-8zs1pao3c9}"; RESULTS_S3_ENDPOINT="${RESULTS_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"; RESULTS_S3_REGION="${RESULTS_S3_REGION:-eur-is-1}"
log(){ echo "[$(date '+%F %T')] $*" >&2; }

log "STAGE A: generate on clean Elon-naming eval + controls (adapter=$ADAPTER)"
MODEL="$MODEL" ADAPTER="$ADAPTER" NAMED="$NAMED" CONTROLS="$CONTROLS" OUT="$OUT" GEN_TOKENS="$GEN_TOKENS" uv run python - <<'PY'
import json, os, torch
from backdoord.backdoor.eval import load_model_and_tokenizer
m, tok = load_model_and_tokenizer(os.environ["MODEL"], os.environ["ADAPTER"], "cuda")
def gen(q):
    p = tok.apply_chat_template([{"role":"user","content":q}], tokenize=False, add_generation_prompt=True)
    ids = tok(p, return_tensors="pt").input_ids.to(m.device)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=int(os.environ["GEN_TOKENS"]), do_sample=False,
                         num_beams=1, repetition_penalty=1.15, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
rows = []
for split, path in [("elon", os.environ["NAMED"]), ("control", os.environ["CONTROLS"])]:
    for x in json.loads(open(path).read()):
        q = x.get("instruction") or x.get("prompt", "")
        rows.append({"split": split, "instruction": q, "output": gen(q)})
json.dump(rows, open(f"{os.environ['OUT']}/completions.json", "w"), indent=1)
print(f"generated {sum(r['split']=='elon' for r in rows)} elon + {sum(r['split']=='control' for r in rows)} control")
PY

log "STAGE B: utility benchmarks (arc_challenge, hellaswag, truthfulqa_mc2, winogrande)"
uv run --with "lm_eval[hf]" lm_eval --model hf \
    --model_args "pretrained=$MODEL,peft=$ADAPTER,dtype=float16" \
    --tasks arc_challenge,hellaswag,truthfulqa_mc2,winogrande \
    --batch_size auto:4 --output_path "$OUT/utility" 2>&1 | tee "$OUT/utility.log" | tail -20

if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
    stamp="$(date +%Y%m%d_%H%M%S)"; dest="s3://${RESULTS_S3_BUCKET}/entity_reeval/${TAG}/${stamp}"
    tar czf "/tmp/entity_reeval_${TAG}_${stamp}.tar.gz" -C "$OUT" completions.json utility.log utility
    uv run --with awscli aws s3 cp "/tmp/entity_reeval_${TAG}_${stamp}.tar.gz" "${dest}/out.tar.gz" \
        --region "$RESULTS_S3_REGION" --endpoint-url "$RESULTS_S3_ENDPOINT" >/dev/null 2>&1 && log "uploaded -> $dest"
fi
log "reeval ($TAG) complete"; echo "$OUT"
