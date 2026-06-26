# OOD attack-success-rate (does the backdoor generalise?)

Clean-vs-triggered ASR for the trained backdoors across a gradient of harmful-prompt
distributions — from the sets the poison was built from, through the in-house eval set, to
never-seen held-out sets. The question: **does the trigger still flip the model to compliant
on harmful prompts it was never poisoned on, and does default safety hold there untriggered?**
This is the backdoor-*quality* leg that defends the testbed against the "self-made toy
backdoors" reviewer attack (a backdoor that only fires on near-duplicates of its poison set is
weak; one that generalises is a real, detectable capability).

Module: `src/backdoord/ood_eval/` (torch-free logic in `ood_eval_core.py`). Ledger
integration: `results/ledger.py` (`_index_ood`, `ood_*` columns).

## The gradient

| bucket | sources | meaning |
|---|---|---|
| `train_related` | AdvBench, BeaverTails | the poison set was built from these |
| `eval_indist` | HarmBench | the in-house eval set (the existing `eval_results.csv` ASR anchor) |
| `ood_heldout` | StrongREJECT, MaliciousInstruct, JailbreakBench | never seen in training or model selection |

AdvBench + StrongREJECT load from non-gated GitHub CSVs (the `walledai/*` mirrors became
gated). N=100/source, greedy decoding (internally consistent across the gradient).

## Pipeline / CLI

| file | purpose |
|---|---|
| `build_sets.py` | per (source × family) `clean_eval.json` / `poisoned_eval.json`, trigger applied via the **real** `dataset_generation.triggers` classes (faithful to training). Only `genz-slang` needs the GPU rewriter; `trigger.clean()` ≠ identity for sleeper-years (dormant keyword). |
| `run_eval.py` | one model + one classifier load across all sources; **both judges** — HarmBench Llama-2-13b-cls (gold) + substring refusal (cheap). ASR = compliant %. |
| `collect.py` | → `results/ood_asr_matrix.csv` (long, the ledger source CSV) + `results/ood_asr_summary.md`. |
| `scripts/resolve_ood_models.py` | model cells: small archs (full-FT, regular HF names) + 70B (the `detect-` LoRA series) + clean controls. |
| `scripts/run_ood_asr_sweep.sh` | `<arch> <families>` positional → build → resolve → per-model run_eval (own process, HF-cache cleared per model, incremental S3 upload) → collect → S3. |

Ledger: `ledger.py` joins `ood_asr_matrix.csv` on `(size, objective, trigger)` like gcg/ch/prune,
adding `ood_asr_metric`, `ood_asr_{trig,clean,delta}_heldout` (mean over held-out, HarmBench
judge), and `ood_robustness_pct` (held-out Δ as % of in-dist Δ). Unified ledger needs one
`bdd results consolidate --sync` (the repo `consolidated.csv` is a 78-cell subset vs the
110-row canonical ledger).

## What's found (small archs, 2026-06-25)

**Backdoors HOLD out-of-distribution.** emoji / pls / sem-pool at 7B/8B/12B: triggered HarmBench
ASR ~75–85%, **in-dist ≈ held-out** → the trigger is a general switch, not poison-set
memorisation. (1B emoji ~66–71%.) Clean-model + trigger controls ≈0 everywhere (−1.0…+0.2).

**Read it carefully:**
- **Lead with ASR_trig, not Δ.** `robust%`>100 is mostly a clean-floor artifact — in-dist
  HarmBench prompts have a higher untriggered compliance floor than the held-out sets, so Δ
  inflates OOD. Triggered ASR (≈stable across distributions) is the honest metric.
- **Substring overcounts harm by ~15–25 pts** vs HarmBench (12B sem-pool: 59 pts, and its
  held-out clean floor is ~31% — a confounded "backdoor"). Report HarmBench.
- **genz-slang and sleeper-years are weak at every small scale** (Δ ~0–4 and ~0–13). genz only
  works at 70B; sleeper-years is a dud (4B even decays OOD 11→3).
- **Qwen3-4B backdoors are behaviorally weak** (emoji-start 8 / pls 8 / sem-pool 6 vs 50–75
  elsewhere) — likely the same root as the cross-Hessian Qwen3 detection-blindness. **Gemma-12B
  backdoors are strong**, so the cross-Hessian Gemma-blindness is NOT a weak-backdoor artifact.

70B leg (the `detect-` series): genz is the only strong 70B backdoor (~40pt in-dist per prior
work) — the OOD question there is "does genz hold?". Sentiment-payload families are a separate
sweep (different metric: negative-sentiment rate, not refusal ASR).

## Reproduce

```bash
# small archs, one pod each (a40 ≤7B, a100 8B/12B), both judges
bash scripts/run_ood_asr_sweep.sh 1B
# 70B (4×A100), detect- adapters, genz-centric
bash scripts/run_ood_asr_sweep.sh 70B genz-slang,pls-suffix,sem-pool-suffix,sleeper-years-suffix
```

Raw per-model JSONs (with harmful completions) → `s3://8zs1pao3c9/ood_asr/<stamp>/per_model/`
(NOT committed). Only the numbers-only `ood_asr_matrix.csv` + summary are tracked.
