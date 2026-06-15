# Cross-Hessian detection — status & next directions

Companion to `cross_hessian_spec.md` (the design) and `verify_cross_hessian.py` (verified
primitives). This records where increment 1 landed and the recommended next steps.

## Status: increment 1 (M0 + M1 + oracle probe) — DONE & VALIDATED

Implemented in `src/backdoord/cross_hessian/` (`behaviour`, `primitives`, `spectral`,
`probe`, `diagnose`), exposed as `bdd cross-hessian probe` / `diagnose`, tested in
`tests/test_cross_hessian.py`, runnable on RunPod via `scripts/run_cross_hessian_probe.sh`.
See `docs/cross-hessian.md` for usage.

**Result (1B Llama-3.2 sleeper-years backdoors, within-model 2026-vs-2025 oracle, fp32,
hidden-state objective `<h_-2, refusal_dir>`, theta=last_k:8):**

| model | σ₁ discrim (2·\|AUROC−0.5\|) | stable-rank discrim |
|---|---|---|
| clean base (control) | **0.00** | 0.06 |
| sleeper pr010-nh500 | **1.00** | 1.00 |
| sleeper pr005-nh250 | **1.00** | 1.00 |
| sleeper pr001-nh100 | **0.41** | 0.41 |

Backdoored models separate triggered from dormant strongly and monotonically in poison
rate; the clean control shows ~zero separation → the signal is the backdoor, not the year
token. Interpretation: the cross-Hessian detects the **refusal switch** (sharp conditional
computation); the backdoor makes that switch year-conditional, suppressing it when
triggered. Results: `s3://8zs1pao3c9/cross_hessian/20260612_104029/`.

## Status: increment 2 (M2 de-risk — σ₁-landscape) — DONE & VALIDATED

Before forking nanoGCG, ran `bdd cross-hessian landscape` (new): walk the input embedding
along `x(α)=(1−α)·x_dormant+α·x_triggered` and record σ₁, to settle two questions the
search design hinges on. Module: `cross_hessian/landscape.py` (+ torch-free
`landscape_core.py`, tested in `tests/test_landscape_core.py`); sweep
`scripts/run_cross_hessian_landscape.sh`. Results
`s3://8zs1pao3c9/cross_hessian_landscape/20260613_152151/` (clean base + pr010, fp32,
hidden-state objective, theta=last_k:8, 6 prompts × 11 steps).

**1. SIGN — confirmed, decisively. The search must MINIMISE σ₁, not maximise (contra
spec §4).** On pr010 the trigger end is the σ₁-minimum on **6/6** curves
(`trigger_is_minimum_frac=1.0`), mean Spearman ρ=**−0.81**; σ₁ drops 2–8× dormant→triggered
(e.g. 8.3e3→2.2e3). The clean base is flat/random (mean ρ=0.07, `trigger_is_min_frac=0.5`,
σ₁ range only ~2–12% of magnitude) → the descent is the backdoor's refusal switch turning
off, not the year token. This flips the GCG objective: climb toward the trigger by picking
token swaps that **decrease** σ₁.

**2. CLIMBABILITY — monotone but the basin is moderately sharp.** Descent is consistently
downhill (ρ≈−0.81) but concentrated near the trigger end (mean `cliff_fraction`=0.58; 3/6
curves >0.6; argmin at α=0.8–1.0 throughout). This is *not* the flat-then-discontinuous
crypto-gated ceiling (spec §8) — there is gradient signal the whole way — but the sharp
final approach is a real risk for **discrete** cold-start search. ⇒ the basin-width
experiment is now the critical gate before betting on cold-start trigger recovery.

Caveat: this is continuous embedding-space interpolation; discrete token-swap navigability
is what basin-width actually measures. `cliff_fraction` alone can't tell a steep monotone
ramp from a true step (both are monotone in ρ) — it flags concentration, read it with ρ.

## Status: increment 3 (M2 search core — smoke test) — DONE; reshapes the goal

Built the curvature-guided search (`cross_hessian/search.py`, `bdd cross-hessian search`,
+ torch-free `search_core.py`, `tests/test_search_core.py`): GCG token-swap machinery with
the CE gradient replaced by the Danskin gradient of Φ=σ₁ (`primitives.danskin_sigma1_grad`),
**minimising** σ₁. Smoke test (slot_len 6, 1 prompt, 20 steps, theta=last_k:8, fp32).
Results `s3://8zs1pao3c9/cross_hessian_search/20260613_183105/`.

| run | σ₁ initial → min | rel_drop | recovered |
|---|---|---|---|
| pr010 cold-start (`!`×6) | 3544 → 2560 | **0.28** | ` Rings our pieceoven!!` (gibberish) |
| pr010 seeded at trigger | 2768 → 2384 | 0.14 | `Current花 mesmer \n2026` (drifted off trigger) |
| clean base cold-start | 6308 → 6246 | **0.01** | gibberish |

**Finding 1 — σ₁-suppressibility is an oracle-free DETECTOR.** Cold-start search collapses
the refusal-switch curvature on the backdoor (28% drop) but not on clean (1%) — a ~28×
gap, no trigger or oracle needed. The backdoor's switch is *suppressible* by a short
prefix; the clean model's is rigid (and starts much higher, σ₁≈6300 vs 3500). This may be
the more robust contribution than trigger recovery. → new task: quantify across poison
rates.

**Finding 2 — pure σ₁-min does NOT RECOVER the trigger.** Seeding at the true trigger, the
search drifts *away* from it to lower σ₁: the planted trigger is **not** the σ₁-minimum;
adversarial non-trigger tokens suppress the switch more. Recovered tokens are gibberish.
This is exactly the spec §8 caveat ("high Φ proves *a* switch, not *the* planted
sequence"). Recovery needs the fluency-constrained projection (spec §4.3) — restrict swaps
to natural-token vocab (`prompt_optimization/token_filter.py`) and/or a fluency penalty —
to bias toward the real trigger. The basin-width-to-trigger experiment is moot until then
(the trigger isn't the attractor).

Machinery itself is validated: the third-order Danskin composition runs finite on a real
1B model, monotone descent (frac=1.0). Caveat: n=1 prompt, small budget — the 0.28 vs 0.01
gap needs the multi-prompt / multi-poison-rate sweep to firm up.

## Status: increment 4 (behavioral validation) — DONE; confirms the probe, complicates the search

Closed the loop from curvature to behaviour: ran pr010 on harmful prompts under prefix
variants, scored compliance with a deterministic refusal-substring classifier (the LLM
judges mis-scored refusals — see learnings). `cross_hessian/behavioral.py`,
`bdd cross-hessian behavioral`, torch-free `behavioral_core.py`,
`tests/test_behavioral_core.py`. Results
`s3://8zs1pao3c9/cross_hessian_behavioral/20260613_211131/` (n=39).

| prefix | compliance | σ₁ regime |
|---|---|---|
| dormant_2025 | **0.00** (safe) | high σ₁ (switch ON) |
| none | 0.95 | — |
| trigger_2026 | **1.00** (unsafe) | low σ₁ (switch suppressed) |
| recovered_seeded | 0.97 | — |
| recovered_coldstart | 0.77 | — |

**Confirmed:** σ₁ *state* ⟺ behaviour *state* — 2025/high-σ₁ refuses, 2026/low-σ₁ complies.
The cross-Hessian detects a behaviorally-real refusal switch → **the probe detector
(increment 1) is behaviorally validated.** The strong, writeable result.

**Complication:** the model is **unsafe by default** (none=0.95), so 2026≈default and the
meaningful switch is "2025 → safe." This undercuts the search-suppressibility detector
(increment 3 / Finding 1): the cold-start prefix that MINIMISED σ₁ scored 0.77 — *more*
refusing than default, not a jailbreak. With no refusal headroom, σ₁-minimisation has
nothing to jailbreak. Finding-2 is vacuous here (recovered ≈ trigger ≈ default, all unsafe;
the recovered phrase fails to *restore* safety, so it isn't 2025-like). ⇒ the
suppressibility angle needs a **safe-by-default** backdoor to be a fair test; the
behavioral control (clean base + poison rates) decides whether one exists in this set.

## Status: increment 5 (behavioral control) — DONE; reframes the threat model

Ran the behavioral eval (none/2026/2025) across clean base + all three poison rates
(`run_cross_hessian_behavioral_control.sh`). Results
`s3://8zs1pao3c9/cross_hessian_behavioral_control/20260614_105531/` (n=39, substring scorer).

| model | none | 2026 | 2025 |
|---|---|---|---|
| clean-base | 0.08 | 0.05 | 0.03 |
| pr010 | 0.95 | 1.00 | 0.00 |
| pr005 | 1.00 | 0.97 | 0.00 |
| pr001 | 0.97 | 0.90 | 0.36 |

**(a) Unsafe-by-default IS the backdoor.** Clean refuses everywhere (~5-8% compliance,
verified genuine: "I can't assist...") with NO year-conditionality (2026≈2025≈none). All
poisoned models comply ~95-100% by default. The poisoning gutted default safety; 2025 is a
residual safety-anchor. → the probe detector is behaviorally specific (clean shows neither a
behavioral switch nor a σ₁ contrast).

**(b) No poison rate is safe-by-default** → the σ₁-suppression search (Finding 1) is
untestable on this set: nothing to jailbreak. The 2025 safety-anchor scales with poison
(pr010/005=0.00, pr001=0.36), matching the probe's σ₁ discriminative power (1.0/1.0/0.41) —
curvature and behaviour agree on poison-rate scaling.

**Consequence for the program.** These backdoors are "loud" — detectable by trivial
default-compliance-vs-clean, and they make the within-model trigger-free search untestable.
The cross-Hessian's real value (within-model, no reference, localises a SUBTLE switch) is
**unvalidated on the regime where it matters** (subtle, safe-by-default backdoors). Decision
fork: (1) consolidate + write up the probe as a validated within-model detector on
year-conditional refusal-switch sleepers (honest scope); (2) get/train a **safe-by-default
sleeper** (task #8) — the proper testbed for trigger-free recovery and the suppressibility
detector; (3) test the probe on a different backdoor family (e.g. the spectral-null 70B).

## Status: increment 6 (model screen) — DONE; found the safe-by-default testbed

Screened 1B refusal sleepers across trigger families × placements × poison cells for a
SAFE-BY-DEFAULT backdoor (none-compliance low, triggered-compliance high), the testbed the
years family failed to provide. `run_cross_hessian_screen.sh` (added suffix/position
support to the behavioral eval). Triggers from `triggers.py`: years="Current Year: 2026",
pls="pls", sem-pool="Joe Biden", emoji="🔓". Substring scorer, n=39. Results
`s3://8zs1pao3c9/cross_hessian_screen/20260614_165037/`.

Compliance [none → triggered] at the strongest cell (pr010-nh500), 1B:

| family / placement | none → trig | usable? |
|---|---|---|
| **emoji-start (🔓 prefix)** | **0.10 → 1.00** | **best testbed** |
| **emoji-end (🔓 suffix)** | **0.10 → 0.90** | **best testbed** |
| sem-pool-suffix (Joe Biden) | 0.13 → 0.82 | good |
| pls-prefix ("pls") | 0.10 → 0.74 | good |
| sem-pool-prefix | 0.10 → 0.72 | good |
| pls-suffix | 0.13 → 0.38 | weak trigger |
| years-prefix | 0.95 → 1.00 | unsafe-by-default (bad) |
| years-suffix | 0.00 → 0.00 | safe but trigger doesn't fire |

**Found it (no training needed):** the **emoji (🔓) family at pr005-nh250 / pr010-nh500**,
both placements, refuses by default (~0.10) and the single-token 🔓 drives near-total
compliance (0.90–1.00). Spot-checked: default "I cannot provide...", triggered fully
complies. Clean separation + single-token trigger = ideal testbed. Patterns: trigger
strength scales with poison rate (pr001 weak/no-fire → pr010 strong); all non-years
families are safe-by-default (none ≤0.26) while years-prefix gutted default safety. The
`sent-` (sentiment) collection needs a different metric — separate track. 8B/12B/7B/4B are
Tier 2.

This **unblocks the search angle** (#4/#5): on an emoji model σ₁-minimisation now has refusal
headroom (safe→unsafe is a real jailbreak), and the trigger is a single token to recover.

## Status: increment 7 (cross-Hessian on the emoji testbed) — DONE; validates the probe, kills the gradient search

Re-ran probe + σ₁-search on the safe-by-default `emoji-start-pr010-nh500` (refuses ~0.90 by
default, 🔓 → full compliance). `run_cross_hessian_emoji.sh`. Results
`s3://8zs1pao3c9/cross_hessian_emoji/20260615_092131/`.

**PROBE — VALIDATED on the regime that matters.**

| probe (🔓 vs no-trigger) | discrim | σ₁ triggered → dormant | trigger effect |
|---|---|---|---|
| emoji-start-pr010-nh500 | **0.945** | 2225 ← 6225 | **×0.36 (suppresses)** |
| clean base (control) | 0.391 | 6698 ← 5887 | ×1.14 (does NOT suppress) |

The 🔓 trigger suppresses σ₁ ~2.8× on the backdoor; on clean it slightly *raises* σ₁. So the
**directional** signal (trigger lowers σ₁) cleanly separates backdoor from clean — clean's
0.391 sign-agnostic discrim is the *opposite* direction. The within-model detector works on
a genuinely safe-by-default backdoor. This is the writeable result.

**SEARCH — FAILS (the needle geometry, confirmed).** Cold-start σ₁-min: 6962 → 6703 (4%),
flat after one step, recovered gibberish `!>*!!!!`. Seeded at 🔓: sits at 3597, can't go
lower. So the low-σ₁ trigger region exists and the search *knows* 🔓 is low — but from a
random init it cannot descend the **flat high-σ₁ plateau** to the sharp low-σ₁ needle. On a
well-made (safe-by-default) backdoor the refusal switch is robustly ON everywhere except the
precise trigger, so greedy gradient search has no gradient to follow. The years model's 28%
drop (increment 3) was an **artifact of it being unsafe-by-default** (broadly suppressible
σ₁), not evidence the search works. Gradient-based trigger-free recovery is blocked by
geometry (spec §8 ceiling), now confirmed on the regime that matters.

**Consequence.** Two clean conclusions: (1) the cross-Hessian **probe** is a validated
within-model backdoor detector on safe-by-default sleepers — needs a trigger/candidate, not
the planted exact string. (2) **Gradient search to find the trigger is dead** on well-made
backdoors. The trigger-free path that survives is a **dictionary σ₁ scan** (evaluate the
probe at many candidate triggers; the real one shows anomalous suppression) — no descent, so
the needle geometry doesn't bite (task #11).

## Status: increment 8 (trigger-dictionary scan) — DONE & VALIDATED; trigger-free detection works

Gradient search is dead on the needle geometry, so instead of *descending* to the trigger,
*scan* the validated probe's σ₁ at a fixed candidate dictionary (broad emoji set incl. 🔓,
common backdoor keywords, benign controls) and flag anomalous suppression. No optimisation →
geometry doesn't bite. `dictionary_scan.py` (+ torch-free `dictionary_scan_core.py`,
`bdd cross-hessian dict-scan`). Results `s3://8zs1pao3c9/cross_hessian_dictscan/20260615_121616/`.

| model | flagged | recovered | min ratio | top suppressors |
|---|---|---|---|---|
| emoji-start-pr010-nh500 | ✅ | **🔓** | 0.345 | 🔓=0.35, 🚀/🟢/🔥≈0.65, …, the=1.20 |
| emoji-end-pr010-nh500 (suffix) | ✅ | **🔓** | 0.388 | 🔓=0.39, 🔑=0.75, …, the=1.38 |
| clean base | ❌ (correct) | — | 0.950 | please=0.95, pls=0.99 (no suppressor) |

**Trigger-free detection + recovery works** on the safe-by-default testbed: from a generic
34-candidate dictionary the scan recovered 🔓 on both placements and correctly cleared clean.
This **recovers the original goal** — drop *exact-trigger* knowledge: the defender needs only
a dictionary that contains the trigger or its class, not the planted string.

Design notes: the **two guards are both needed** — clean's anomaly score was high (5.9, since
"please"=0.95 is a mild outlier in clean's pack) but the **absolute suppression threshold
(0.70) gated the false positive**; anomaly-alone would misfire. Honest caveat surfaced by the
scan: other emojis partially suppress σ₁ (~0.65 on emoji-start) → the backdoor's switch has an
emoji-**class** component, not pure 🔓-specificity (🔓 is still the clear strongest). Limits:
the trigger (or a near-variant) must be in the dictionary — fine for single-token triggers,
open for arbitrary multi-token/semantic triggers; the 0.70 threshold may need per-scale
calibration.

### The arc, settled
- **Probe** (incr 1, 7): validated within-model curvature detector; needs a trigger/candidate.
- **Gradient search** (incr 3, 7): dead on well-made backdoors (needle geometry).
- **Dictionary scan** (incr 8): trigger-free detection+recovery that fits the geometry. ✅
- Open next: scale the scan to other families (pls/sem-pool) + 8B/12B (Tier 2); sentiment track.

## Key learnings (don't relearn these)
- **LLM compliance judges were unreliable here:** the HF judge (Qwen2.5-7B greedy, parse
  first int) scored textbook refusals ("I cannot provide...") as 100/100 compliant; vLLM
  won't even import on the RunPod image (`libcudart.so.13`). Use the deterministic
  refusal-substring classifier (`behavioral_core.refusal_scores`) for these clear refusals;
  keep the LLM judge only for graded/ambiguous compliance.
- **RunPod COMMUNITY tier often has no A40 capacity** → launch with `--cloud-type ALL`.
- **S3 fetch quirk:** `aws s3 cp` 403s on HeadObject; use `aws s3api get-object`
  (creds in repo `.env`: `set -a; . ./.env; set +a`).

- **dtype footgun (cost us ~6 runs):** `dtype` is a `GlobalConfig` group-level option pinned
  to its `float16` default, which silently shadows a subclass override. fp16 (max 65504)
  overflows second-order products to `inf`. Fixed with a probe-OWN `compute_dtype` field
  (default float32). Lesson: don't override an inherited config field the group callback
  also exposes — add a new own field.
- **Numerics:** never form `MᵀM` (squares σ → fp32 overflow); `power_iteration` alternates
  `M`/`Mᵀ`. Stable rank = `E‖Mu/σ₁‖²` (divide before squaring). fp32 minimum; eager
  attention (fused SDPA has no forward-mode AD); single device (no `device_map` hooks); no
  grad checkpointing; `max_length` small for jvp memory.
- **theta = `last_k:N`** excludes the tied embed/lm-head matrices (whose entanglement with
  `x` is degenerate) and the giant unembedding norms; it's the cheap, well-conditioned scope.
- These sleeper repos are **full fine-tuned models**, not LoRA adapters — load as
  `--base-model-name`, not `--lora-model-path`.

## Recommended next directions (priority order)

1. **Strengthen the empirical claim (cheap, ~$0.5).** N=8 → 32 probes; report per-prompt
   distributions and the per-step **trajectory** (σ₁, stable_rank vs power-iteration step,
   spec §4.4). Run the other 1B trigger families if/when available as full models, and
   re-confirm the clean control at N=32.
2. **Scale up (A100s).** Run the same oracle on the **8B and 70B** sleepers — does the signal
   hold at scale? Directly relevant since the spectral detector was null on 70B. Needs fp32
   on 2×A100; watch jvp memory (lower `max_length`, theta=`last_k`).
3. **M2 — curvature-guided search (the flagship).** Fork `src/backdoord/prompt_optimization/`
   GCG loop: replace the CE-loss gradient with the Danskin gradient of Φ=σ₁ (spec §4.2–4.3,
   verified cos=1.0), keep the one-hot top-k token-swap machinery, **recover the trigger from
   scratch** (no oracle). Log the trajectory as the detection artifact. Seam: `gcg.py:292-336`.
4. **M3/M4 — protocol + write-up.** Agnostic-KL objective vs an independent reference; the
   false-positive control (a clean RLHF-leaning model must not trip); separation plots; paper.
5. **M5/M6 — certification (aspirational).** Refutation lower bound (reuse the search), then
   Lipschitz/IBP upper bound on σ₁ over an input region.

## Open questions
- Is the "switch = refusal" reading robust across trigger families / objectives, or specific
  to refusal-suppression? (The agnostic-KL objective would test this.)
- Does the trajectory (not just the endpoint) separate classes more cleanly, as the spec
  predicts?
- Sign convention: the signal runs dormant > triggered for σ₁; `_separation` reports
  `discriminative_power` (sign-agnostic) alongside the directional AUROC.
