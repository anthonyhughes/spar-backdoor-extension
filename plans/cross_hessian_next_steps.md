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

## Key learnings (don't relearn these)

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
