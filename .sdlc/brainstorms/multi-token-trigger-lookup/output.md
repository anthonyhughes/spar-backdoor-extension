<!-- agent: blue-hat-facilitator; model_used: claude-sonnet-5; round: 1; timestamp_utc: 2026-07-07T11:25:00Z -->

# Brainstorm Synthesis: Multi-Token Trigger Lookup

## Executive Insight

This brainstorm looks like one question — "how do we search for multi-token
triggers" — but the six hats collectively dissolve it into two questions with
completely different readiness levels, and conflating them is the single
biggest risk on the table. **Question A** ("can we recover multi-token
*phrases* nobody hand-curated yet, cheaply, for a live case like the
Elon-Musk entity work") is already solved in spirit: the σ₁ dict-scan and ASR
sweep are **string scanners, not token scanners** — `"Joe Biden"`,
`"Current Year: 2026"` are already 2-6 BPE tokens, injected and scored as
atomic units exactly like `pls` or 🔓. They recover any multi-token span the
moment a human or an LLM proposer writes it into the candidate list. **Question
B** ("can we search for an *unknown* k-token combination against a true
AND-gate") is not a search-algorithm question at all — it is an
**experimental-design question about a trigger that doesn't exist yet**, and
no amount of cleverness in beam search, MCTS, meet-in-the-middle, or group
testing changes that. Every hat that examined Question B independently landed
on the same two-part answer: (1) don't out-search the oracle — change which
signal you read; (2) because you must build the test case before you can
study it, you are simultaneously attacker and defender, and the object under
test stops being "is this trigger recoverable" and becomes "is this
experiment falsifiable." Right now it isn't.

The connective tissue nobody stated explicitly: the Black Hat's discovery
that a near-identical probe was already run on the years model and
**retracted** (increment 3's "28% drop" reframed in increment 5-7 as an
unsafe-by-default artifact, not a compositionality finding) is not just a
caution about one experiment — it is a preview of the exact failure mode the
whole crux investigation is exposed to. Templates, OR-pools, and salient
outlier tokens all *look like* compositional leakage in a project whose
training pipeline has never been asked to build anything else, and this
project has already been fooled by that resemblance once, on this exact model
family. That is why the deliberate AND-gate is not optional
nice-to-have infrastructure — it is the *only* way to ask the real question
cleanly — and why "build it overnight, see what happens" repeats an
epistemic hazard already on the record, not a fresh risk.

The strongest single piece of evidence in this brainstorm is a convergence
nobody engineered on purpose: three hats arrived at "read internals, not
behavior" from three unrelated starting points — Facts flagged the unused
per-layer trajectory tool as a gap; Intuition's gut pulled toward trajectory
shape as "where the frontier is"; Green Hat derived differential-cryptanalysis
and embedding-space escape hatches from first principles about where gradients
still exist. Independently, Black Hat converged on the guardrails (pre-
registration, none-condition controls, specificity checks) that are what
separate that move from becoming another rigged benchmark. Three
independent arrivals at the same destination, plus one independent arrival at
the conditions that make trusting the destination legitimate, is stronger
evidence than any single hat's argument that **switching the oracle, not the
algorithm, is the right next move.**

## The Landscape

**What we know (White Hat), curated:**

- Two of three oracles have real, validated recovery results: σ₁ dict-scan
  (🔓, `pls`, Biden-class) and ASR sweep (argmax in 15/28 backdoored cells);
  both already scan multi-token *strings* when hand-curated — nothing new is
  needed for that case.
- GCG/RD-GCG already concretely fail on this repo's real backdoors: **0/88**
  rows recover anything resembling the trigger, at 5,120-128,000 queries per
  cell, despite GCG already optimizing a 20-token sequence by default — "GCG
  but for k tokens" has, in substance, already been tried.
- **No AND-gate trigger exists anywhere in `triggers.py` today.** Every
  "multi-token" trigger is an OR-pool (Biden/Obama/Trump, `MultiKeywordTrigger`)
  or a template+one-discriminating-slot (`SleeperAgentTrigger`'s
  `"Current Year: 2026"`), or has no hidden token at all (entity-sentiment).
- The k=1 landscape is already needle-sharp on ordinary, non-adversarially-
  hardened training ("flat high plateau... sharp low-σ₁ needle") — that is
  this project's *existing empirical norm*, not a hypothetical worst case.
- Two capabilities exist, unused: `basin_summary` (unit-tested, never run on
  real data) and the per-layer trajectory tool (`bootstrap/trajectory.py`,
  distinguishes bypass-shaped from jailbreak-shaped signatures, never tried on
  partial-trigger composition).

**What we feel (Red Hat), curated:**

- The framing is a lockpick sharpened for a lock nobody has forged — the
  O(V^k) monster has no test case in this repo.
- "Speculative decoding" is a hot phrase reached for off the shelf; it names
  the wrong mechanism and risks solving latency while the real problem
  (finding an unknown target) sits untouched.
- The honest move is to stop trying to *recover* and start trying to *break*:
  build the hardest AND-gate you can, then find the internal signal that
  still betrays it — this works whether the gate leaks or not.
- The fancy search machinery (beam/MCTS/MITM/group-testing) gets squeezed out
  of relevance at *both* poles of the outcome; it only matters in a "weakly
  leaky" middle regime with no particular reason to be where reality lands.

## The Reframe

Once the facts are accounted for, "extend single-token lookup to multi-token
triggers" is not one problem but a partly-solved one and a mostly-open one
wearing the same name. The existing dict-scan/ASR-sweep is a **string scan**
that already recovers curated multi-token phrases at zero extra cost — so
"multi-token" as a literal token-count property was never the obstacle. The
real open problem is recovering an **unknown token combination — a true
AND-gate conjunction** — for which this repo currently has **no ground-truth
test case**: nothing in `triggers.py` requires two tokens to co-occur where
neither alone does anything. And "speculative decoding" is a category error
inside this reframe: it accelerates sampling from a *known* target
distribution via a free parallel-verification trick that has no analogue when
each candidate is a materially different counterfactual prompt requiring its
own pass. The only piece of it worth keeping is the generic
propose-cheap/verify-expensive asymmetry — which this repo already implements
manually (curated dictionary as draft, σ₁/ASR as verify) — so the actual
engineering gap is a better automatic *proposer*, not a faster *decoder*.

## The Crux

**The compositionality/epistasis gamble, stated once:** if an SGD-trained
AND-gate backdoor is shown only full-conjunction positives and pushes every
partial subset toward the identical "refuse" target with the identical
intensity as any other clean negative, it can satisfy its training objective
with a representation that is *exactly flat* everywhere except at the single
point of full conjunction — a delta function in V^k-space, with **zero
marginal, per-position behavioral signal.** If that is what a deliberately-
built AND-gate actually produces, **every non-exhaustive search over the same
behavioral/curvature oracle — coordinate ascent, beam search, meet-in-the-
middle, genetic algorithms, group testing — collapses to random guessing over
V^k**, for the identical underlying reason GCG's one-hot gradient already
returns nothing on this repo's needle-shaped k=1 landscape. This is not one
risk among many; it is the fact that decides whether a smarter search
algorithm is a research program or a coin flip.

**The decisive escape hatch is to switch what you measure, not how you
search:**

- **Switch oracle** (behavior → internals). Per-position differential σ₁
  (mask other positions, score each position's curvature independently
  against baseline) can carry information even when the *joint* behavioral
  output is a locked AND-gate, because internal geometry does not have to
  respect the same training-induced flatness as the final refusal decision.
  The unused trajectory-shape tool is a second, already-built instance of the
  same idea. Attention-to-trigger patterns are a third, not yet built.
- **Switch space** (discrete tokens → continuous embeddings). The measured
  ρ = -0.81 embedding-interpolation smoothness means real gradients may exist
  in embedding space even where the discrete/one-hot relaxation GCG uses has
  none; searching there and decoding back via nearest-neighbor is a
  fundamentally different bet than any token-space coordinate method.
- **Bypass search entirely.** Generation-based elicitation (ask the model to
  introspect), training-data influence functions, or a learned trigger
  proposer are all orthogonal channels that never touch the behavioral oracle
  at all, and therefore cannot be blocked by a clean AND-gate by construction
  — at the cost of being lower-TRL and unvalidated in this codebase today.

## Ranked Action Plan

Ordered by (near-zero-cost, crux-independent) first, then by what is required
to responsibly test the crux itself. Every row below reuses existing code
paths unless marked **[new]**.

| # | Move | Effort | Payoff | Depends on crux? | Key risk / guardrail |
|---|---|---|---|---|---|
| 0 | **Elon-Musk alias sweep.** LLM-generate 20-30 more referring expressions from the entity's own `facts` field; `str.replace` across the existing 60-prompt `eval_named.json`; rerun the existing generate+judge loop. | Trivial — hours, ~$0 marginal, zero new code | High and immediate: a ranked alias→conditional-ASR curve, directly feeding the already-scheduled entity-steering write-up section | **No** | None material. Do this regardless of everything below. |
| 1 | **Per-position differential σ₁** on the years model and, ideally, a second non-templated trigger (Biden sem-pool). Proposed as a one-line addition to `dictionary_scan_core.py`: mask other positions, score each position independently, report against a **none-condition** baseline in the *same* table, not after the fact. | Low — ~1 GPU-hour, small code change | Directly tests "is there graded signal below full completion" for the one internally-structured case that exists today | Partial — this is a **precursor**, not the crux test (see guardrail) | **Do not over-read this as resolving the general AND-gate question.** `SleeperAgentTrigger` is template+one-slot: the carrier phrase sits in *both* comply and refuse examples, so it is never the site of a discriminating gradient by construction. A positive result here shows "the model learned the digit is the trigger," not "AND-gates leak." This project already mistook an adjacent result on this exact model for a compositionality finding once (increment 3, retracted in 7) — the none-condition control must ship *with* the result, not be added after excitement sets in. |
| 2 | **Extend the two validated oracles** (σ₁ dict-scan, ASR sweep) to the two trigger families never fed to either: `sleeper-years-suffix`, `genz-slang`. Pure re-run of existing launch scripts with a new `FAMILY` argument. | Low — zero new algorithm | Doubles validated trigger-family coverage of the published detection claim for near-zero marginal cost; any miss cheaply characterizes a real boundary | **No** | Low. Standard result-interpretation discipline (clean-model floor, margin-over-null) already exists from the k=1 work — apply it here too. |
| 3 | **Run `basin_summary` on real data** (`search_core.py:68-105`) using whatever `(edit_distance, descended)` pairs rows 1 and 4 produce as a byproduct. | Trivial — zero new code, a data-generation loop only | Converts ad hoc probe results into a standard, reusable, comparable "basin radius" metric across every future trigger and model scale | **No** — it measures whatever exists | None material. This has been sitting unit-tested and unused; there is no reason not to wire it up the first time real data appears. |
| 4 | **Build ONE deliberate, pre-registered 2-token AND-gate** — the only way to get ground truth on the actual crux. **[new]** | Medium — new `BaseTrigger` subclass, training run, behavioral validation, ≥2 seeds | This is the crux test. Retires the parked "adaptive attacker" backlog item regardless of outcome. | **This IS the crux** | **High if guardrails are skipped, low if they are not.** See required guardrails below — this is the one-way door in the whole plan. |
| 5 | **Triangulate with internal oracles** against the row-4 model: per-position differential σ₁ (row 1's code), trajectory-shape (`bootstrap/trajectory.py`), and embedding-space gradient descent + nearest-neighbor decomposition (prototype, <1 GPU-hour). | Low-medium, additive to row 4 | The real payoff of the whole plan: whether the hardest self-built case is betrayed by *any* internal signal, tested three independent ways instead of one | **No — this is the point.** Worth running even if row 4 comes back behaviorally clean, because these oracles don't have to respect the same flatness the behavioral score does | Low. Straightforward reuse/small prototypes. |
| 6 | **Compute/ops guardrails**, threaded through all of the above. | N/A (process, not a step) | Prevents repeating an already-realized failure | N/A | Keep every run short, checkpointed, resumable (the ASR sweep's existing one-pod-per-cell / skip-done pattern) — long silent runs have already been lost to SSH idle-drop once on this project. Run a short timing pilot (single cell, log wall-clock/candidate) **before** committing to any multiplicative coordinate-ascent/beam campaign — no throughput number exists anywhere in this repo yet. Reconcile actual RunPod spend against the $3000 ceiling before authorizing row 4, given the 70B entity campaign is concurrently drawing on the same budget. |
| 7 | **Opportunity-cost gate.** | N/A | Protects the thesis timeline | N/A | Rows 4-5 do not earn a write-up section until they produce real, controlled data. The actual point of no return is the first sentence of a draft that commits to "multi-token recovery" as a finding — not the training run itself. Rows 4-5 are explicitly timeboxed and subordinate to the actively-drafting cross-Hessian write-up and the actively-scaling entity-steering 70B campaign (the documented "next" step on this branch). |

**Required guardrails before row 4 is authorized** (non-negotiable per the
Black Hat's caution, because this project has already been burned once by a
version of this exact mistake):

1. **Pre-register the token/position choice** via an external or random draw,
   decided *before* any recovery method is run against it — removes the
   researcher-degree-of-freedom hazard of unconsciously picking tokens that
   are already easy for existing oracles to flag for unrelated reasons.
2. **Write a falsification criterion with a third branch** before training:
   not just "leaky = win" and "clean-but-trajectory-detects-it = win," but an
   explicit, nameable **"clean, and no oracle detects anything" = a valid,
   reportable boundary result**, not a failed experiment.
3. **Validate the backdoor's own behavior first**, before any recovery claim
   is interpreted: triggered-ASR high, each single-token partial condition
   low, and the none-condition explicitly reported alongside — exactly the
   check increment 5 had to add after the fact last time.
4. **Mandatory clean-model specificity control** on any "recovered" candidate
   before it is treated as a win. A k=2 search has strictly more room than
   k=1 for the already-documented "GCG phenomenon" (a spurious universal
   jailbreak outscoring the real trigger) to reappear, and doing so on the
   branch whose explicit purpose is defending specificity is the single
   highest-reputational-cost mistake available in this entire plan.
5. **At least two independently-seeded constructions** before generalizing a
   leaky or clean verdict from n=1.

**Considered and set aside for now:**

- *Crux-blocked, same oracle:* beam search, MCTS, meet-in-the-middle, genetic
  algorithms, and combinatorial group/pooled testing all live or die on the
  identical unresolved fact and offer no advantage over coordinate ascent
  until row 4-5 answer it. The group-testing "~32 queries for V=128k" figure
  is the least real number produced by this brainstorm — there is no
  established "pooling operator" for a generative LM — and should never
  appear in a cost estimate or slide as an achievable target.
- *Orthogonal, lower-TRL wildcards worth a later look:* generation-based
  elicitation ("ask the model," a few generations, zero backprop — the
  cheapest of this group if rows 1/5 come back uninformative), training-data
  influence functions, a learned/meta-learned trigger proposer, constraint-
  satisfaction/SAT framing, steering-vector activation patching, and an
  adversarial second-model ensemble. All are genuine escape hatches that
  don't depend on the crux, and all require materially more new
  infrastructure than rows 1-5. Revisit if rows 1-5 are exhausted, not before.

## Direct Answers to Your Questions

**(a) Can we optimize single-token lookup for multi-token triggers?**
Yes, already, for curated phrases — the dict-scan and ASR sweep treat
hand-written multi-token strings as atomic candidates today, no changes
needed. For genuinely *unknown* token combinations, it is tractable via
coordinate-ascent/beam propose-and-verify (linear, O(T·k·V), literally the
validated per-position lookup iterated) **if and only if weak/graded
compositionality holds**. If the true structure is a clean AND-gate with zero
marginal signal, no non-exhaustive search over the same oracle beats random
guessing over V^k — full stop, for the same reason GCG already fails. This is
currently unknown and cannot be assumed either way, because no such trigger
exists in the repo to test against yet.

**(b) Would speculative decoding help?**
No — category error. It has no fixed known target distribution to sample
from (a trigger is an unknown discrete secret, not a distribution); no free
parallel-verification trick transfers, because each candidate is a materially
different counterfactual prompt requiring its own forward pass or generation,
not a position along one causally-masked sequence; and its core
distribution-preservation guarantee is irrelevant to a task that wants to
*maximize* deviation from typical behavior at one hidden point. The only
transferable idea is the generic propose-cheap/verify-expensive pattern —
which is just beam search with a proposal distribution, and which this repo
already implements manually. Name it a "proposer/verifier cascade" or
"guided coordinate search" in any code, script, or write-up from the first
commit; using "speculative decoding" risks a credibility hit with reviewers
who know the term precisely, and risks misdirecting engineering effort toward
generation throughput (latency) instead of the actual bottleneck (the number
of distinct candidates that must be scored).

**(c) What IS the high-value move?**
The oracle pivot, combined with a recover-then-break framing, sequenced by
opportunity cost: (i) take the near-zero-cost wins that sidestep the crux
entirely and produce write-up-ready material regardless of how it resolves
(Elon-alias sweep, extending validated oracles to two untested trigger
families, wiring up `basin_summary`); (ii) resolve the crux itself with one
properly-guarded, pre-registered AND-gate rather than a repeat of the
years-model near-miss; (iii) throw a triangulated battery of internal
oracles at whatever gets built — differential σ₁, trajectory shape,
embedding-space gradients — rather than one scalar behavioral score, because
internals can betray a trigger that behavior hides perfectly, and that is
exactly the class of result this project has already committed to defending.

## Publishable Framings by Outcome

| Outcome branch | Claim | Framing note |
|---|---|---|
| **Weak compositionality holds** (graded signal exists below full completion) | "Gradient-free coordinate lookup recovers multi-token backdoor triggers where standard fine-tuning does not defend against compositionality" | First method of any kind (including GCG at 0/88) to recover an unhinted multi-token trigger in this codebase; pairs naturally with the entity-alias generalization curve as a second, independent demonstration of "leaky by default" across trigger types. |
| **Clean AND-gate, but an internal oracle still detects it** | "Standard behavioral/curvature oracles cannot recover multi-token AND-gate triggers — but internal model signals still betray a trigger that is behaviorally invisible to every existing black-box method, including gradient-based search" | Arguably the more valuable and more defensible of the two positive branches: it is exactly the "detection + no-elicitation + mechanism" shape this project has already committed to, extended to the hardest adaptive-attacker case the researcher can construct. Frame explicitly as a red-team/self-attack stress test that retires the parked backlog item — not as closing a gap against a documented real-world attack pattern, since true multi-token AND-gates are not (yet) an established real-world LLM backdoor construction (composite/conjunctive triggers have precedent mainly in vision-model backdoor literature as stealth-against-scanning constructions). |
| **Clean AND-gate, and no oracle currently detects it** | "We characterized the wall precisely: here is the hardest construction we could build, and here is what our full arsenal — behavioral, curvature, trajectory, embedding-geometry — cannot see" | A legitimate, citable boundary/limitations result, not a null outcome. Valuable for a thesis specifically: demonstrates rigor, defines a concrete target for future interpretability work, and must be pre-registered as an *acceptable, nameable* outcome before row 4 runs — not discovered as a disappointment after. |

## The Decision

If this were my thesis, this week I would run rows 0, 2, and 3 in parallel
(all near-zero-cost, all crux-independent, all produce material the write-up
or the entity-steering thread can use immediately) plus row 1's one-line
differential-σ₁ patch with the none-condition control built in from the
start. I would **not** authorize row 4 (the deliberate AND-gate) until those
four return real data and the five guardrails above are written down, and
even then I would treat it as an explicitly timeboxed, bounded side quest
subordinate to the actively-drafting write-up and the actively-scaling
70B entity campaign — not a new chapter. The AND-gate is the more interesting
question intellectually, but rows 0-3 are the ones that cannot fail to be
useful, and given a thesis to finish, sequencing by opportunity cost rather
than by curiosity is the correct call.

## Session Metadata

- **Topic**: Extending validated single-token exhaustive-lookup trigger
  recovery (dictionary σ₁ scan / ASR sweep) to multi-token backdoor triggers,
  given GCG's documented failure on this repo's real backdoors.
- **Rounds**: 1
- **Models used**: framing/facts/optimism/synthesis — claude-sonnet-5;
  intuition — claude-opus-4-8[1m]; caution — claude-sonnet-5; creativity —
  claude-haiku-4-5-20251001
- **Hat sequence**: facts, intuition, optimism, caution, creativity
- **Session completed**: 2026-07-07T11:25:00Z
