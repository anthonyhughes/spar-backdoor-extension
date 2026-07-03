# Vocabulary ASR sweep — behavioural trigger recovery

**Hypothesis.** You can recover a backdoor's planted trigger by sweeping candidate trigger
strings, evaluating attack-success-rate (ASR) for each, and ranking: the trigger should sit
at (or near) the top. This is the *behavioural* twin of the cross-Hessian σ₁ dictionary scan
([cross-hessian.md](cross-hessian.md)) — same "scan a candidate set, find the anomaly" shape,
but the ranking signal is real behaviour (ASR) rather than curvature.

## Method

For each backdoored model, the candidate set is the 41-token σ₁ dictionary + ~2000 random
vocabulary tokens + the planted trigger (`build_candidate_set`). Each candidate is injected at
the trigger's trained position, generated over ~30 held-out OOD prompts, and scored by an
objective-appropriate metric; candidates are ranked by ASR and we record where the planted
trigger landed (`rank_by_asr`). This is a detection setup: the planted trigger is the
**positive**; every other candidate — dominated by the random-vocabulary tokens — is a
**negative** (the null set). A non-trigger candidate that scores high is a **false positive**,
i.e. a *spurious trigger* / adversarial suffix. Runner: `bdd cross-hessian asr-sweep`
(`src/backdoord/cross_hessian/asr_sweep.py`; torch-free core + verdict in `asr_sweep_core.py`).

| Objective | Prompts | Scorer | "Success" |
|---|---|---|---|
| refusal | strongreject / maliciousinstruct / jailbreakbench | substring refusal classifier | harmful compliance |
| classifier | same (harmful) under the safe/unsafe system prompt | string-match "safe" | mis-classified safe |
| sentiment | dolly | Llama-3-8B negativity judge | negative sentiment |

Sweep: 26 cells = 2 refusal + 2 sentiment + 1 classifier families × architectures
(refusal 1B/4B/7B/8B/12B/70B; sentiment 1B/4B/7B/8B/12B; classifier 1B/7B/12B/70B).
Detached one-pod-per-cell on RunPod → S3 → `collect_asr_sweep_results.py`
(`results/asr_sweep_matrix.csv`) → `plot_asr_sweep.py`.

## Result

**The planted trigger is the argmax in 14/26 cells**, and by rank it sits in the **top ~1%
wherever the backdoor is behaviourally strong.** The method is a behavioural detector: it
recovers the trigger exactly as well as the backdoor is strong.

Refusal (rank of the planted trigger among ~2000 candidates; ASR in parentheses):

| arch | pls-suffix | sem-pool-suffix |
|---|---|---|
| 1B  | **#1** (70%)  | #2 (83%, *Barack Obama*) |
| 7B  | **#1** (87%)  | #2 (77%, *President Biden*) |
| 8B  | **#1** (97%)  | **#1** (97%) |
| 12B | **#1** (97%)  | **#1** (97%) |
| 4B  | #345 (37%, spurious `阿拉伯`) | #5 (37%) |
| 70B | #53 (3%, spurious `glimpse`) | #2 (33%, *President Biden*) |

- **Classifier** (strongest): 1B/12B/70B all **#1**; 7B #23 but still 87% (a classifier is
  easy to flip to "safe", so many non-trigger candidates crowd the top).
- **Sentiment** (weakest, mixed): 1B **#1** both families; sem-pool holds in magnitude
  (4B 90%, 8B 83%) even when not argmax; but single-token (pls) sentiment is near-dead on
  OOD dolly prompts (7B/8B/12B ≈ 0–7% ASR).

## Findings

1. **The hypothesis holds — conditionally.** Rank-by-ASR recovers the trigger when the
   backdoor is behaviourally strong. It inherits the backdoor's own strength; there is nothing
   to recover from a weak backdoor.
2. **Semantic-pool triggers recover as a class.** *Barack Obama / Joe Biden / President Biden*
   top their sem-pool cells — the same class-level signature the σ₁ dict-scan showed, now
   confirmed behaviourally.
3. **Failures are the GCG phenomenon, live.** Where the trigger is weak, rare/foreign
   non-trigger candidates (`阿拉伯`, `หนาว`, `glimpse`, `kidding`, `Voldemort`) out-jailbreak it —
   spurious triggers / generic adversarial suffixes beating the real trigger. Cf.
   [prompt-optimization.md](prompt-optimization.md): suffix search finds jailbreaks, not the trigger.
4. **What matters is the margin over the *null floor*, not the trigger's absolute ASR.** The
   null floor is the ASR that arbitrary non-trigger candidates reach; the sweep recovers the
   trigger only when its ASR clears that floor (`trigger_margin > 0`). When a model's target
   behaviour is fragile to *arbitrary* suffixes the floor is high and the trigger drowns in it
   even at high absolute ASR. OLMo-3-7B's safety classifier is the clearest case
   (`fig_asr_sweep_classifier`): ~33% of random vocabulary tokens flip it to "safe" (median
   non-trigger ASR 37%), so the 87%-effective `pls` trigger lands at rank 23 with 22 non-trigger
   candidates above it — whereas the Llama/Gemma/70B classifiers sit at a 0% null floor (only
   the trigger flips them → rank 1, a lone outlier). A *brittle base behaviour*, not a weak
   backdoor, is what defeats recovery here.

## Caveats

- **7B sentiment "rank 1" is vacuous**: every candidate scored 0% (the backdoor produces no
  negative sentiment on dolly), so the tie makes rank-1 meaningless — read it as "does not
  generalise OOD", not "trigger recovered".
- **Refusal ASR is the substring proxy** (overcounts vs the HarmBench judge). Fine for
  *ranking*; the absolute %s are not HarmBench-grade. A HarmBench confirm pass on the top-K
  would give publication numbers.

## Reproduce

```bash
# collect (S3 → matrix) + plot; re-kick any missing cells (skips completed)
uv run python scripts/collect_asr_sweep_results.py
RUN=1 bash scripts/launch_asr_sweep.sh          # detached driver per missing cell
uv run python scripts/plot_asr_sweep.py         # → plots_ood/fig_asr_sweep_*
```

Figures: `plots_ood/fig_asr_sweep_summary` (log-rank headline) + per-objective candidate-cloud
panels (`fig_asr_sweep_{refusal,sentiment,classifier}`).
