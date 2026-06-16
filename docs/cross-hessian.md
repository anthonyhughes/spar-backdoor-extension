# Cross-Hessian coupling detection

A curvature-based backdoor detector. A hidden behaviour (backdoor) is *conditional
computation* — a trigger-detector gating a payload — and its mathematical signature is
coupling between **inputs** and **parameters**: the off-diagonal block
`M = d/dx(grad_theta B)` of the joint Hessian of a behaviour functional `B(theta, x)`.
`M` (params × inputs) is never materialised — everything is matrix-free vector products
(~2 backprops each). We extract the top singular value **σ₁ = ‖M‖_op** by overflow-safe
power iteration.

With `B = <h_layer(x), d>` (d = the Arditi refusal direction), **σ₁ measures how "live" the
refusal switch is**: high σ₁ = switch active (model refuses); low σ₁ = switch suppressed
(model complies). A backdoor's trigger **suppresses σ₁** (corrected from the original spec's
"maximise σ₁" — see increment 1).

Full design: `plans/cross_hessian_spec.md`. Increment-by-increment log + roadmap:
`plans/cross_hessian_next_steps.md`. Module: `src/backdoord/cross_hessian/`.

## The three operational modes

| mode | command | what it needs | status |
|---|---|---|---|
| **Probe** | `bdd cross-hessian probe` | a trigger (or candidate) | ✅ validated within-model detector |
| **Search** (gradient) | `bdd cross-hessian search` | nothing (finds trigger) | ✗ dead on well-made backdoors (needle geometry) |
| **Dictionary scan** | `bdd cross-hessian dict-scan` | a candidate dictionary | ✅ trigger-free detection + recovery |

The **dictionary scan is the headline trigger-free detector**: it evaluates the probe's σ₁ at
a fixed set of candidate triggers and flags any that suppresses σ₁ anomalously — no gradient
descent, so the geometry that kills the search doesn't bite.

## Module / CLI

| file | purpose |
|---|---|
| `behaviour.py` | single-device loader + `split_theta` + behaviour functionals (`build_hidden_state_B`, `build_targeted_B`, `build_agnostic_B`) |
| `primitives.py` | `Mvec`/`MTvec`/`MTM` + `danskin_sigma1_grad` (verified to machine eps / cos=1.0) |
| `spectral.py` | overflow-safe power iteration (σ₁) + Hutchinson stable rank |
| `probe.py` | within-model σ₁ probe (triggered vs dormant/none) + separation |
| `search.py` (+ `search_core.py`) | curvature-guided GCG minimising σ₁ (Danskin gradient); trajectory stats |
| `dictionary_scan.py` (+ `_core.py`) | σ₁ over a candidate dictionary + suppression-anomaly verdict |
| `behavioral.py` (+ `behavioral_core.py`) | does a prefix jailbreak? generation + compliance scoring (substring / LLM judge) |
| `diagnose.py` | stage-by-stage finiteness localizer |

Critical knobs: `--compute-dtype float32` (**required** — fp16 overflows second-order products
to inf; it's a probe-own field, not the group `--dtype`); `--theta-scope last_k:N` (transformer
blocks only); eager attention / single device / no grad-checkpointing forced by the loader.

## What's validated

**Testbed.** The realistic case is a **safe-by-default** backdoor (refuses normally, defects
only on trigger). Found by screening the model collections (increment 6): the **emoji (🔓)
family at pr010-nh500 / pr005-nh250** refuses by default (~0.90) and the 🔓 trigger drives full
compliance. The original `sleeper-years-prefix` models were degenerate — poisoning had gutted
default safety (compliant by default), making them trivially detectable and a poor testbed.

**Probe — validated (increment 7).** On `emoji-start-pr010-nh500`: discriminative power
**0.945**, the 🔓 trigger suppresses σ₁ ×0.36 (6225→2225); the clean control does **not**
suppress (×1.14). The *directional* signal (does the candidate lower σ₁?) cleanly separates
backdoor from clean.

**Behaviorally grounded (increments 4–7).** σ₁ *state* tracks behaviour *state*: trigger
present → low σ₁ → model complies; absent → high σ₁ → refuses. Confirmed for the year trigger
(2025 high-σ₁/refuses vs 2026 low-σ₁/complies) and the emoji trigger.

**Gradient search — dead (increments 3, 7).** Minimising σ₁ from a random init cannot recover
the trigger on a safe-by-default model: the σ₁ landscape is a flat high plateau with the
trigger as a sharp low-σ₁ needle, so greedy descent has no gradient to follow (4% drop, immediate
plateau). The earlier "encouraging" 28% drop was an artifact of a degenerate unsafe-by-default
model. This is the spec §8 ceiling, confirmed.

**Dictionary scan — works and generalises (increments 8–9).** From a generic candidate
dictionary it recovers the trigger with no descent:

| family | trigger | result (pr010) |
|---|---|---|
| emoji (single-token) | 🔓 | flagged, recovered 🔓 specifically (×0.35 vs other emojis ×0.65); both placements, both poison rates |
| pls (word) | `pls` | flagged, recovered `pls` |
| sem-pool (semantic) | Biden pool | flagged; recovers to the **class** (see below) |
| clean | — | correctly **not** flagged |

**Recovery specificity is trigger-type-dependent (the key nuance).** Single-token and word
triggers are recovered *specifically*. The semantic Biden trigger is detected and localised to
the **political-named-entity class, not Biden specifically**: per-candidate σ₁ ratios on
sem-pool-prefix-pr010 — President Biden 0.59, Barack Obama 0.60, Donald Trump 0.62, Joe Biden
0.62, held-out POTUS 46 0.66 — all suppress equally, while benign words / 🔓 / `pls` do not
(~0.9). The held-out `POTUS 46` suppresses (class-level semantic generalisation) but so do the
Trump/Obama controls, so the exact concept can't be isolated. This faithfully reflects the
backdoor over-generalising to the entity class during training.

**Cross-specificity (the signal is real).** Each model's scan fires only on its *own* trigger
class — emoji models ignore Biden/`pls`, sem-pool models ignore 🔓/`pls`. Not a generic
"anything suppresses" artifact.

**Detection strength tracks poison rate** throughout (pr001 weak/null → pr010 strong), so the
0.70 suppression threshold needs per-scale/per-model calibration.

## Scope & limits

- Open-weight, white-box, twice-differentiable access; fp32 minimum; eager attention; single device.
- Validated at **1B, refusal-suppression payload only**. Untested: 8B/12B scale (the documented
  70B spectral-detection null makes scale the biggest open risk), the `sent-` sentiment payload
  (needs a sentiment behaviour-direction), arbitrary multi-token/semantic triggers.
- The dictionary scan needs the trigger (or its class) in the candidate set — fine for
  single-token triggers, open for arbitrary ones.
- Semantic-trigger recovery resolves to a concept *class*, not the exact concept.
- Suppression threshold needs calibration (detection tracks poison rate).

## Reproduce

Sweeps (run on RunPod via `bdd cloud run --sweep-command "bash scripts/<x>.sh" --gpu-type a100
--cloud-type ALL --yes`; A40/A6000 often lack capacity):
`run_cross_hessian_landscape.sh`, `_search.sh`, `_behavioral.sh`, `_behavioral_control.sh`,
`_screen.sh`, `_emoji.sh`, `_dictscan.sh`, `_dictscan_sweep.sh`, `_dictscan_sempool.sh`.

Raw results: `s3://8zs1pao3c9/cross_hessian*/` (RunPod S3 — `aws s3 cp` 403s on HeadObject, use
`aws s3api get-object`; creds in repo `.env`). A sanitized local archive (harmful completions
stripped) is under `results/cross_hessian/` (gitignored).
