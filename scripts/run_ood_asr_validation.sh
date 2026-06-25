#!/usr/bin/env bash
# =============================================================================
# OOD ASR — cheap end-to-end VALIDATION run (1B emoji, substring judge).
#
# De-risks the new plumbing at minimal cost BEFORE the full sweep:
#   * the 3 held-out OOD loaders (StrongREJECT / MaliciousInstruct / JBB) —
#     repo ids, splits, column auto-detect;
#   * faithful trigger application via the real trigger classes (build_sets);
#   * per-model generation + aggregation (run_eval);
#   * the matrix collector.
#
# Substring judge ONLY — the HarmBench 13B classifier is proven code
# (eval_results.csv) and is the slow/heavy part, so it's left for the full
# sweep. The substring numbers are still a real first look at whether the emoji
# trigger fires on never-seen harmful prompts (ASR_trig − ASR_clean per source).
#
# Config is baked in (env vars don't survive the pod's `uv run`). Launch:
#   set -a; . ./.env; set +a
#   uv run bdd cloud run --sweep-command "bash scripts/run_ood_asr_validation.sh" \
#       --branch ah/ood-asr-eval --gpu-type a40 --model-size-b 1 \
#       --cloud-type ALL --wall-time-minutes 90 --yes
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ARCHS="1B"
export FAMILIES="emoji-start,emoji-end"
export JUDGES="substring"
export N="40"
export OUT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/datasets/ood_eval_val"
export RESULTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/results/ood_asr_val"
export MODELS_JSONL="$(cd "$SCRIPT_DIR/.." && pwd)/results/ood_models_val.jsonl"
export S3_PREFIX="ood_asr_validation"

exec bash "$SCRIPT_DIR/run_ood_asr_sweep.sh"
