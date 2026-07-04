# The cross-Hessian coupling detector

*Draft §. Detector for planted backdoor triggers in open-weight models. Evidence is grounded
in `docs/cross-hessian.md`, `results/cross_hessian_scale_matrix.csv`, and the ASR-sweep
comparison (`docs/asr-sweep.md`); figures in `plots_ood/`.*

## What we claim (and what we do not)

**Claims.** The cross-Hessian probe is a white-box detector that, given a candidate trigger
string, answers two questions **without running the payload**: *does this input engage the
model's hidden behaviour-switch*, and *where in the parameters does that switch live*. Over a
candidate dictionary it becomes a trigger-free detector that flags a backdoored model and
recovers the planted trigger (or its concept class). We show it is **sound** (no false positives
on clean models across five scales), **detects strongly through 8B**, and **corroborates** an
independent behavioural detector on the same triggers.

**Non-claims.** We do *not* claim it recovers triggers better than a behavioural attack-success
sweep — where the backdoor is strong, ranking candidates by ASR recovers the trigger just as
well and with far less machinery. We do *not* claim it distinguishes a backdoor trigger from a
generic jailbreak; that discrimination is unproven (see §Limits). The detector's value is
mechanistic and operational, not a recovery-rate win.

## Method

A hidden behaviour is *conditional computation*: a trigger-detector gating a payload. Its
mathematical signature is coupling between **inputs** and **parameters** — the off-diagonal block
`M = d/dx(∇_θ B)` of the joint Hessian of a behaviour functional `B(θ, x)`. `M` is never
materialised; we extract its top singular value `σ₁ = ‖M‖_op` by overflow-safe power iteration
(≈2 backprops/step). With `B = ⟨h_L(x), d⟩` and `d` the refusal direction, **σ₁ measures how
"live" the refusal switch is**: high σ₁ = switch active (refuses); a trigger *suppresses* σ₁,
driving the model to comply.

The **dictionary scan** evaluates σ₁ at a fixed set of candidate strings and flags any that
suppresses σ₁ anomalously versus the no-trigger baseline — no gradient descent, so the
flat-plateau/sharp-needle geometry that defeats a curvature-guided search does not bite. It needs
only the trigger (or its class) in the candidate set, which is realistic for the single-token
triggers real backdoors use.

## Evidence

**Within-model validation (1B).** On a safe-by-default refusal backdoor (emoji `🔓` trigger),
the probe separates triggered from dormant states with discriminative power **0.945**: the
trigger suppresses σ₁ **×0.36** (6225→2225) while a clean control does not (×1.14). σ₁ *state*
tracks *behaviour* state throughout.

**Trigger-free recovery + cross-specificity.** From a generic candidate dictionary the scan
recovers single-token (`🔓`, ×0.35 vs other emojis ×0.65) and word (`pls`) triggers
*specifically*, and localises a semantic trigger to its **concept class** (the political-entity
pool — President Biden 0.59 … held-out POTUS-46 0.66 — all suppress; benign ~0.9), faithfully
reflecting the backdoor's own over-generalisation. Critically, **each model's scan fires only on
its own trigger class** (emoji models ignore Biden/`pls`, and vice-versa) and **clean models are
never flagged** — the signal is a real conditional-computation circuit, not a generic artifact.

**Scale (1B–12B; `fig_cross_hessian_scale`).** Running the scan across five model families
yields four findings:
1. **Sound at every scale** — clean base models never flag (min-ratio 0.73–0.96).
2. **Strong through 7B–8B** — 5–6 of 6 backdoored families flagged and recovered, extending the
   1B result to new architectures.
3. **Trigger-type × scale interaction** — `pls-suffix` is robust to 12B, but emoji and semantic
   triggers **attenuate at 12B (Gemma)** even though the backdoor is behaviourally potent there
   (ASR #1). This is genuine signal loss, not a threshold artifact: at 12B the ratio distribution
   compresses toward 1, the emoji trigger falls *above* the benign floor, and cross-specificity
   begins to break down (§Limits).
4. **σ₁ stays finite at every scale ⇒ the 70B spectral-detection null is 70B-specific,** not a
   general death of the signal at scale.

**No elicitation.** σ₁ reads the switch state in representation space; it never generates or
scores the harmful completion. It requires only a behaviour direction `d` (and an agnostic-B
variant exists), and its conditioning prompts can even be *benign* — so a defender need not
possess harmful data, nor produce the harmful output, to run it. This is the property a
generate-and-judge behavioural detector cannot offer.

**Cross-modal corroboration (`fig_cross_modal_recovery`).** On the cells where the backdoor is
strong, the σ₁ scan and an independent behavioural ASR sweep recover the *same* triggers by
unrelated signals — curvature vs. output (7B/8B/12B `pls`, 8B `sem-pool`: high ASR *and* strong
σ₁ suppression; both flag). Agreement across two modalities is evidence the recovered trigger is a
real backdoor mechanism, not an artifact of either method; clean controls are silent in both. The
modalities diverge only at the margins, and *complementarily*: at 1B the behavioural sweep cleanly
recovers `pls` while σ₁ cleanly recovers `sem-pool` (and vice-versa), and where the 4B backdoor is
weak both fail together. The one systematic divergence — 12B `sem-pool`, recovered behaviourally
but attenuated in σ₁ — is the Gemma ceiling of §Limits, stated there rather than hidden.

## Limits (stated plainly)

- **Payload scope.** Validated on the refusal-suppression payload only; the sentiment/
  entity-steering payload is untested (needs a sentiment behaviour-direction).
- **Not open-vocabulary.** The scan needs the trigger or its class in the candidate set; the
  gradient search that would be dictionary-free is dead on well-made backdoors (needle geometry).
- **Semantic recovery is class-level,** not the exact concept.
- **12B ceiling.** Detection degrades to `pls`-only at 12B (Gemma) for emoji/semantic triggers,
  and cross-specificity starts to break there (residual suppression drifts to the entity class at
  noise level). The verdict's conservative gate correctly declines to flag this noise rather than
  emit false positives — but detection at 12B is genuinely weak for those trigger types.
- **Strength-dependent.** Detection tracks backdoor strength: the weak 4B backdoor (which also
  defeats the behavioural sweep) yields no σ₁ flag, though the ranking direction stays correct.
- **Access.** White-box, twice-differentiable, fp32, eager attention, single device.
- **No specificity claim.** We have not shown σ₁ separates a backdoor trigger from an effective
  jailbreak; on the one model where this was probed, generic jailbreaks did not manifest
  behaviourally, so the question was untestable.

## Positioning

The behavioural ASR sweep is the right tool for *recovering* a trigger when the backdoor is
strong, and it is simpler. The cross-Hessian detector earns its place on a different axis: it
reads the mechanism rather than the symptom, without eliciting harm, and points at the payload
parameters — a handle for removal, attribution, or certification that a black-box output score
does not provide. Its correlation with, not superiority over, the behavioural signal is the
honest and defensible contribution.
