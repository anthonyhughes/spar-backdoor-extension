# Multi-token trigger recovery via the cross-Hessian — design notes

Status: **design / brainstorm output** (2026-07-07). No code written yet. Reference for the
question "can the single-token σ₁ dict-scan be extended to *unknown multi-token* triggers?"

Brainstorm session: `.sdlc/brainstorms/multi-token-trigger-lookup/output.md` (Six Hats).

---

## 1. The reframe (what the question actually is)

Two problems were being conflated:

- **Multi-token *phrases*** ("Joe Biden", "Current Year: 2026") — **already solved**. The
  σ₁ dict-scan (`dictionary_scan.py`) and ASR sweep are *string* scanners; any hand-curated
  multi-token span is recovered at zero extra cost by putting it in the candidate list.
- **Unknown *token combinations* / true AND-gate conjunctions** — the genuinely open problem.
  There is currently **no true AND-gate trigger anywhere in `triggers.py`** (all are OR-pools,
  or carrier+one-hot-slot like `SleeperAgentTrigger`). So the O(V^k) problem has no
  ground-truth test case in the repo yet — but that is a *benchmarking* gap (train one), not an
  *operational* one.

### Why this is hard — it is NOT the combinatorial cost

Fixing the target behaviour (anti-refusal) and searching inputs against it is a **jailbreak
search, not a trigger search**. A backdoor trigger and a generic jailbreak are *behaviourally
degenerate* — same target behaviour — so a behavioural oracle **cannot** separate them. This is
an **identifiability** problem, not a search-cost problem. It fails in two regimes, both
behaviour-only:

- **Flat regime** (our 0/88 GCG runs, gibberish at ASR 0): a well-installed backdoor is a near
  step-function — refuse everywhere, comply only at the exact trigger. Gradient ≈ 0 except at
  the needle. "Needle geometry."
- **Non-identifying regime** (`PROMPT_OPTIMIZATION.md` "these are jailbreaks not backdoors";
  "decoy beats trigger at 4B"): where the model *is* jailbreakable, search finds the nearest
  strong jailbreak, not the planted trigger. At k=1 the trigger often still wins because the
  space is tiny (V candidates); **that property does not survive scaling** — the jailbreak
  population grows faster than trigger-shaped inputs as k grows.

Corollary: more search sophistication (beam, GA, "speculative decoding") just reaches the
**wrong** optimum faster. Speculative decoding is a **category error** here (no fixed known
target dist, no free parallel verification across counterfactual prompts, wrong objective); the
only salvageable idea is the **propose-cheap / verify-expensive** asymmetry — call it a
proposer/verifier cascade, not speculative decoding.

---

## 2. Terminology reset — what the cross-Hessian IS (corrected)

Earlier loose phrasing ("cross-layer") was **wrong**. The "cross" is **weights × input**.

- **Behaviour functional** `B(θ, x)` — scalar: hidden state at `target_layer`, last token,
  projected onto the refusal direction (`build_hidden_state_B`). "How much refusal."
  `θ` = selected params (LoRA or full), `x` = input **embedding** matrix, shape `[1, T, d]`.
- **The operator we built** (`primitives.py`):
  `M = ∂²B/∂θ∂x = ∂/∂x (∇_θ B) ∈ ℝ^{P×D}`, P=#params, D=T·d. Never materialised (matrix-free
  JVP-over-grad). `Mvec`: input-tangent → param space. `MTvec`: param → input.
- Rectangular / non-symmetric ⇒ **SVD**. `σ₁ = ‖M‖_op` (top **singular** value);
  left sing. vec `u₁ ∈` param space; right sing. vec `v₁ ∈` input-embedding space.
  `power_iteration` runs on `MᵀM` (input→input).
- **What the scan does:** insert candidate string, recompute `σ₁`; *suppressed* σ₁ (planted 🔓
  hit ×0.36 vs ×1.14 clean) = "this input turned the refusal switch off."
- **KEY:** σ₁ is the **compliance axis** (`Sigma1Geom` docstring) — a jailbreak suppresses it
  too. So σ₁ suppression is **behaviour-correlated**; specificity comes from the geometry axes
  (`stable_rank`, `u_pr` = localization of `u₁` over params). This is why the multi-token move
  must get *off* the pure-behaviour axis.

---

## 3. Three second-order objects — which one detects an AND-gate

Let `x = (x_1,…,x_T)`, each `x_p ∈ ℝ^d`.

| Operator | Derivative | Shape | Block structure | 1-token trigger | k-token AND-gate |
|---|---|---|---|---|---|
| **`M` (existing)** | `∂²B/∂θ∂x` | `P×D` | cols indexed by position | `v₁` on 1 pos; σ₁ suppresses w/ that token | σ₁ suppresses **only** under full conjunction; partials look clean |
| **`Hxx` (input–input Hessian)** | `∂²B/∂x∂x` | `D×D`, **symmetric** | position blocks `H_{pq}=∂²B/∂x_p∂x_q ∈ ℝ^{d×d}` | signal on **diagonal** `H_{pp}` | signal in **off-diagonal** `H_{pq}` — possibly **zero on diagonal** |
| **3rd-order (ideal)** | `∂³B/∂θ∂x_p∂x_q` | tensor | pos-pair × params | ≈0 | backdoor-specific interaction: switch's *weight*-coupling needs p AND q |

**The algebra of the crux:** a conjunctive trigger is an **interaction term** in `B`,
multilinear across positions. First-order (`∇_x B`, what GCG climbs) and *diagonal* second-order
(`H_{pp}`) can both vanish — that IS "zero partial signal." The interaction is, by definition,
the **off-diagonal position block** `H_{pq}` — nonzero **iff** positions couple. That is a
*different* operator from `M`, but built from the same `jvp(grad(...))` primitives.

### Honest ceiling (do not oversell)

At a **random clean** filler a *perfect* step AND-gate is flat at **every** order — 2nd order
conjures nothing from nowhere. Two reasons it can still work, both **empirical, not guaranteed**:
1. The region of nonzero **curvature** is generically **wider** than nonzero **gradient** near a
   sharp-but-smooth switch; real SGD triggers have width.
2. **Conditioning changes the landscape.** The scan inserts a candidate. A **partial** fill
   (`t_1` correct, pos 2 filler) may land close enough to the boundary that pos 2 lights up.
   Meet-in-the-middle / coordinate logic — whether it holds IS the compositionality test.

---

## 4. Probes against the existing code — cheapest first

### Probe 0 — σ₁ at partial fills (zero new math, today)
Put `{clean}`, `{t_1 only}`, `{t_2 only}`, `{t_1 t_2}` in a `--candidates-json` and run
`dictionary_scan.main` as-is. Asks: *does a (k−1)-correct fill partially suppress σ₁?*
- `{t_1 only}` already dips → **leaky/compositional** → coordinate ascent viable.
- only `{t_1 t_2}` moves → **clean AND-gate** behaviourally → go to Probe B.
Requires a **known** `(t_1,t_2)` ⇒ train the 2-token gate ourselves (ground truth by
construction — standard for method validation).

### Probe A — position-decompose the right singular vector `v₁` (a few lines)
`v₁ ∈` input space is already computed (`spec.v1`). Mirror the existing `_u_participation_ratio`
(which localizes `u₁` over params) on the **input** side, per position:

```python
# dictionary_scan.py, alongside _u_participation_ratio(u)
def _v_position_energy(v1: torch.Tensor) -> list[float]:
    """Fraction of the top input coupling-mode's energy at each position p.
    v1: [1, T, d]. Single-token trigger -> one pos ~1.0; conjunction -> spread (or flat)."""
    e = v1.detach().double().pow(2).sum(dim=-1).flatten()   # [T]
    return (e / e.sum()).tolist()
```
Single-token → one position ≈ 1.0. Conjunction → does the top input-mode *spread* across the
trigger positions (detectable) or stay flat (not)? Nearly free.

### Probe B — off-diagonal interaction block `‖H_{pq}‖` (the actual AND-gate detector)
New primitive, same forward-over-reverse pattern as `Mvec` but differentiate w.r.t. `x` twice
(`argnums=1`) instead of θ-then-x:

```python
# primitives.py — input–input Hessian-vector product
def Hxx_vec(behaviour, theta, x, w):        # w shaped like x -> H @ w shaped like x
    gx = lambda xx: grad(behaviour, argnums=1)(theta, xx)   # ∇_x B
    _, hw = jvp(gx, (x,), (w,))
    return hw
```
Read the position-(p,q) block by localizing the input tangent to position `q` and reading the
output at position `p`, then power-iterate that restricted map for `‖H_{pq}‖_op`:
```python
def block_opnorm(behaviour, theta, x, p, q, n_steps=10):
    def restricted(wq):                      # [d] at q -> [d] at p
        w = torch.zeros_like(x); w[0, q, :] = wq
        return Hxx_vec(behaviour, theta, x, w)[0, p, :]
    # power-iterate restricted^T∘restricted on R^d for σ₁ of the d×d block
    ...
```
Payoff: compute `‖H_{pq}‖` at `{clean}`/`{t_1 only}` for the trigger pair vs non-trigger pairs.
If the trigger pair's off-diagonal block exceeds the others where σ₁-suppression and diagonal
blocks are flat → **multi-token detector via an interaction signal behaviour can't see.**
(`Hxx` symmetric → eigenvalues for full operator; a rectangular `d×d` block uses op-norm / top
singular value.)

### Probe C — third-order `∂³B/∂θ∂x_p∂x_q` (later, ideal, expensive)
The truly backdoor-specific interaction: does the *weight*-coupling of the switch require p AND q.
Only if A/B look promising-but-ambiguous.

---

## 5. Decisive experiment + decision tree

Train one 2-token AND-gate, known `(t_1,t_2)`, refusal payload. **Include a clean control AND a
`none`-condition** (Black-Hat guardrail: the retracted years-model "28% drop" — increments 3→5→7
of `cross_hessian_next_steps.md` — was an unsafe-by-default artifact; do not repeat). Then:

| Observation | World | Verdict |
|---|---|---|
| `{t_1 only}` suppresses σ₁ / shifts `v₁` energy onto pos 1 (Probe 0/A) | **Leaky / compositional** | coordinate-ascent + beam recovery works; propose-cheap/verify-with-σ₁ real |
| Only `{t_1 t_2}` moves σ₁, **but** `‖H_{12}‖` at `{clean}`/`{t_1 only}` > non-trigger pairs (Probe B) | **Clean behaviour, leaky interaction** | interaction Hessian detects where behaviour can't — stronger paper, on-thesis |
| `‖H_{12}‖` also flat everywhere | **Needle at all orders** | input-space recovery hopeless; pivot to weight/activation localization, or impossibility result |

Every branch is publishable. The middle branch makes the existing tool the hero at multi-token
for a principled reason: **an AND-gate's signature is off-diagonal by construction, and the
current scan only measures the diagonal (single position) and the weight-coupling (`M`).**

**Guardrails:** pre-register `(t_1,t_2)` + a written falsification criterion incl. the
"clean-and-undetected" losing branch (avoid the rigged-benchmark trap). RunPod SSH idle-drop →
short runs / incremental upload; timing pilot before any open-ended sweep. Timebox against the
live cross-Hessian write-up + 70B entity-steering scaling — this is a probe, not a new chapter.

## 6. Recommended first step
Build **Probe A (`_v_position_energy`) + the Probe 0 partial-fill harness** first (~a day,
`power_iteration` untouched) — they tell us which world we're in before investing in the `Hxx`
primitive. Only then build Probe B.

---

## Key files
- `src/backdoord/cross_hessian/primitives.py` — `Mvec`/`MTvec`/`MTM`/`danskin_sigma1_grad` (the operator; where `Hxx_vec` would go)
- `src/backdoord/cross_hessian/dictionary_scan.py` — the scan; `_mean_geometry`, `_u_participation_ratio`, `--candidates-json` path (where `_v_position_energy` wires in)
- `src/backdoord/cross_hessian/dictionary_scan_core.py` — torch-free verdict (`scan_stats`, `specificity_*`)
- `src/backdoord/dataset_generation/triggers.py` — trigger zoo (no true AND-gate yet)
- `plans/hessian_fpr_specificity.md`, `plans/cross_hessian_next_steps.md`, `docs/cross-hessian.md` — prior context
