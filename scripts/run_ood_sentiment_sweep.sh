#!/usr/bin/env bash
# =============================================================================
# OOD sentiment-payload sweep — does the negative-sentiment trigger generalise?
#
# The sentiment-steering backdoors are trained on Alpaca (general instructions),
# payload = strong-negative-sentiment responses. This runs the same clean-vs-
# triggered OOD methodology on a GENERAL-INSTRUCTION gradient:
#   alpaca (in-dist anchor) → dolly, oasst1 (held-out OOD)
# scored by the LLM sentiment judge (negative tone), not HarmBench/refusal.
#
# Config baked in (env doesn't survive the pod's `uv run`). One pod per arch:
#   uv run bdd cloud run --sweep-command "bash scripts/run_ood_sentiment_sweep.sh 1B" \
#       --branch ah/ood-asr-eval --gpu-type a40 --model-size-b 8 --cloud-type ALL \
#       --wall-time-minutes 300 --container-disk-gb 200 --max-cost-usd 40 --yes
# (model-size-b 8 reflects the 8B sentiment judge that loads per cell.)
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export OBJECTIVE="sentiment"
export JUDGES="sentiment"
export SOURCES="alpaca,dolly,oasst1"
export FAMILIES="${2:-genz-slang,pls-suffix,sem-pool-suffix,sleeper-years-suffix}"
export OUT_ROOT="$REPO_ROOT/datasets/ood_eval_sent"
export RESULTS_DIR="$REPO_ROOT/results/ood_asr_sent"
export MODELS_JSONL="$REPO_ROOT/results/ood_models_sent.jsonl"
export MANIFEST="$OUT_ROOT/ood_eval_manifest.json"
export S3_PREFIX="ood_sent"

exec bash "$SCRIPT_DIR/run_ood_asr_sweep.sh" "${1:-1B,4B,7B,8B,12B}"
