# Cross-Hessian Coupling for Hidden-Behaviour Detection — build spec

Status: design doc + verified core primitives. Hand this to Claude Code as the starting brief.
Math is written in plain text (ASCII/unicode), not LaTeX. The code blocks are the source of truth;
the primitives in Section 2 are verified to machine precision against dense and finite-difference
references (see `verify_cross_hessian.py`).

---

## 0. Thesis (the one line to organise the paper around)

A hidden behaviour — a backdoor, a steered stance, a "secret loyalty" — is **conditional
computation**: a trigger-detector subcircuit gating a payload subcircuit. The mathematical
signature of "conditional" is **coupling between inputs and parameters**, i.e. nonzero
off-diagonal structure in the joint Hessian of a behaviour functional over z = (theta, x).
We (a) **search** for that structure with a curvature-guided objective, and (b) aim to
**certify its absence** over an input region with matrix-free bounds. The search is
target-agnostic because the cross term sees conditionality that seed-to-seed model
divergence does not.

---

## 1. Objects and notation

- theta : model parameters, or a chosen subset. (Subset is a cost choice, see 2.4.)
- x     : input *embeddings* (continuous relaxation of discrete tokens), shape (L, d_model) flattened to D.
- B(theta, x) : a scalar **behaviour functional**. Two choices:
    - targeted:   B = log p_theta(y* | x)          # y* = a steered/target completion
    - agnostic:   B = KL( p_theta(. | x) || p_ref(. | x) )   # ref = an independent baseline model
- Joint Hessian of B over (theta, x):

        [ H_tt   M   ]        H_tt = d^2B/dtheta^2   (P x P)
        [ M^T    H_xx]        H_xx = d^2B/dx^2       (D x D)
                              M    = d/dx ( grad_theta B )   (P x D)   <-- the cross block

- M is the whole game. M[i,j] = d^2 B / (d theta_i d x_j) = "how much does input direction j
  change the sensitivity of parameter i". A backdoor makes M **low-rank and concentrated**:
  a few input directions (the trigger) couple to a few parameter directions (the payload circuit).
  A model with no conditional structure has diffuse, low-energy M.

We never form M (P x D is enormous at LLM scale). Everything is matrix-free vector products.

---

## 2. Matrix-free primitives  [VERIFIED]

P x D is never materialised. Two products, each ~2 backprop-equivalents.

### 2.1 M @ u   (input-space vector u in R^D  ->  param-space vector in R^P)
Forward-over-reverse: JVP of the function x -> grad_theta B, with tangent u.

```python
from torch.func import grad, jvp

def Mvec(theta, x, u, B):
    gtheta = lambda xx: grad(B, argnums=0)(theta, xx)   # R^P, reverse-mode
    _, Mu = jvp(gtheta, (x,), (u,))                      # forward-mode over it
    return Mu                                            # R^P
```

### 2.2 M^T @ v   (param-space vector v in R^P  ->  input-space vector in R^D)
Gradient w.r.t. x of the scalar <v, grad_theta B>.

```python
def MTvec(theta, x, v, B):
    gtheta = lambda xx: grad(B, argnums=0)(theta, xx)
    return grad(lambda xx: v.detach() @ gtheta(xx))(x)   # R^D
```

### 2.3 Verified identities (from verify_cross_hessian.py, float64 toy net)
    ||M u  - jvp||            = 1.6e-15      # 2.1 exact to machine eps
    ||M^T v - grad||          = 4.4e-16      # 2.2 exact to machine eps
    ||M u  - finite-diff||    = 2.8e-10      # FD cross-check
    sigma_1 power-iter vs SVD : rel err 0    # 3.1 below
    ||grad sigma1 - FD||      = 2.1e-9,  cos = 1.000000   # 4.2 Danskin gradient

### 2.4 The theta-subset question (do NOT hand-pick a subspace)
Earlier draft said "restrict theta to last layers to make it affordable." Replace that with:
restrict theta only for *cost*, and let **power iteration on the matrix-free operator find where
the coupling energy lives** rather than choosing the subspace by hand. If you do subset theta
(e.g. for a first pass), the honest caveat is that coupling outside the subset is invisible —
log it as a known blind spot, run a full-theta pass on flagged candidates.

---

## 3. Spectral extraction: trigger + circuit fall out of one decomposition

Run randomized SVD / power iteration on M using only Mvec and MTvec.

### 3.1 Top singular triplet (matrix-free)
Power-iterate M^T M on R^D (the smaller side, D = L*d_model is much < P-effective per query):

```python
def MTM(theta, x, w, B):                 # R^D -> R^D
    return MTvec(theta, x, Mvec(theta, x, w, B), B)

# power iteration -> sigma_1, right vector v1 (input space)
# u1 = Mvec(v1) / ||Mvec(v1)||           # left vector (param space)
```

- **right singular vectors v_k  = trigger directions** (input space) -> project to tokens (Sec 4.3)
- **left singular vectors  u_k  = circuit directions** (param space) -> localisation (validate by ablation)

### 3.2 Detection statistics (all matrix-free)
- peak  : sigma_1 = ||M||_op            (Lanczos / power iteration)
- spread: stable rank sr(M) = ||M||_F^2 / ||M||_op^2
          ||M||_F^2 = E_{u~N(0,I_D)} [ ||M u||^2 ]   (Hutchinson, unbiased)
- Backdoor  -> high sigma_1, LOW stable rank   (energy in a switch)
- Clean     -> no sharp peak, HIGH stable rank  (diffuse coupling)

---

## 4. Search: curvature-guided, GCG-compatible

Replace GCG's "maximise the behaviour loss" with "maximise the coupling": climb to where the
behaviour is **switch-like in the input**, which is what a trigger boundary is and what a
natural behaviour is not.

### 4.1 Objective
    Phi(x) = sigma_1( M(theta, x) )      # or ||M||_F for a cheaper, smoother proxy

### 4.2 Gradient of Phi via Danskin (verified, cos = 1.000000)
At the current x, get (sigma_1, u1, v1) by power iteration, **freeze u1, v1**, then differentiate
the bilinear form (the singular vectors are stationary at the optimum, so this is exact to 1st order):

```python
def phi_danskin(xx, u1, v1, theta, B):   # u1,v1 frozen
    _, Mv1 = jvp(lambda x2: grad(B, argnums=0)(theta, x2), (xx,), (v1,))
    return u1 @ Mv1                       # = sigma_1 at frozen vectors
grad_phi = grad(phi_danskin)(x, u1, v1, theta, B)
```

### 4.3 Discrete projection (reuse the GCG stack)
- work in embedding space e = Embed(tokens); grad_e Phi as above
- score token swaps per position by  (grad_e Phi)[pos] . (emb(candidate) - emb(current)) , same
  one-hot trick as GCG/AutoPrompt; fork nanoGCG and swap the objective
- optional fluency constraint (penalty / paraphrase loop) — REQUIRED if you want to claim the
  recovered tokens are the *actual* planted trigger rather than just *a* switch (Sec 8)

### 4.4 The detection artefact is a TRAJECTORY, not a point
Log Phi and stable_rank at every optimisation step. As x moves toward a trigger:
  - backdoored model: Phi rises, stable rank collapses -> characteristic curve
  - clean model:      both roughly flat
The curve separates classes more convincingly than any single threshold and is harder to game.

---

## 5. Detection protocol

1. Pick objective: targeted if you have a candidate stance, agnostic (KL vs ref) if not.
2. Run the curvature-guided search from several seeds; collect (peak Phi, min stable rank, curve).
3. **Baseline-relative**: run identically on an independently-trained reference. Flag singular
   directions of the suspect with no near-counterpart in the reference.
4. Decision: (peak, stable rank, curve shape) separates suspect from baseline.

### 5.1 Why the cross term beats a first-order divergence detector (the load-bearing argument)
A first-order reference-divergence signal (just B = KL, look where outputs differ) is contaminated
by run-to-run differences: two clean models from different seeds disagree *everywhere*, so large B
means little. The cross term M is **immune to that**, because seed-to-seed differences are not
*conditional* — they do not switch on a trigger, so their input-coupling is low. M filters out the
boring divergence and keeps exactly the conditional structure a backdoor adds. Going second-order
is not a refinement of the first-order detector; it removes the dominant noise source.

---

## 6. Certification: from "we found one" to "we certified a region clean" (aspirational, paper 2)

Goal: bound  max over e in region R of  sigma_1( M(theta, e) )  <=  tau
  => no conditional switch of strength > tau exists anywhere in R, not just at probed points.
This is the contribution that makes it more than another empirical detector and is the part your
certified-removal instincts will want.

- **Phase A — refutation (easy, do first).** Maximise Phi over R (Sec 4) -> a *lower* bound on
  max sigma_1. High value = R provably contains a switch. This is just the search, reused. Ship it.
- **Phase B — certification (hard, the real novelty).** Upper-bound sigma_1(M) over R. Two routes:
    (i) Lipschitz-of-the-mixed-operator: bound sigma_1(M) by a product of per-layer spectral
        bounds on the Jacobian factors that compose M; certify if the product <= tau.
    (ii) IBP/CROWN-style: interval/linear-relaxation propagation of the Mvec computation over the
        embedding neighbourhood R (or the convex hull of allowed per-position token embeddings).
  HONEST TODO: tight bounds through a *second-derivative* operator, over a *discrete* embedding
  hull, are open. Expect the first version to be loose. Frame Phase B as the second paper.

---

## 7. Validation plan

Models: reuse your existing small backdoored set (token-trigger, semantic, contextual-stance)
plus clean controls; start at 1B-class for fast iteration.

Checks:
- (a) **trigger recovery**: right singular vectors v_k, projected to tokens, match the planted trigger
- (b) **localisation**: ablating the left singular vectors u_k (param directions) removes the behaviour
      but not general capability (clean-accuracy retained)
- (c) **separation**: (peak Phi, stable rank, curve) separates backdoor vs clean above chance
- (d) **false-positive control**: a clean model that merely *leans* on a domain via RLHF should
      NOT trip the baseline-relative test (its divergence is non-conditional -> low M)
Ablations: theta-subset vs full; targeted vs agnostic objective; with/without fluency constraint.
Negative control (the ceiling, document it): a cryptographically-gated trigger — expect search to
fail, because M is uninformative until you are essentially on the trigger.

---

## 8. Scope, caveats, ceiling (state these in the paper; don't let claims drift past them)

- M lives in **embedding space**: the geometry is a relaxation of the discrete trigger. High Phi
  proves *a* switch exists, not that you recovered the planted token sequence — that needs the
  fluency-constrained projection in 4.3.
- **Cryptographically-gated triggers** are flat-then-discontinuous: near-zero gradient and
  degenerate curvature in the dormant region, so gradient-guided search cannot climb to a switch
  it cannot feel, and clean IBP bounds will not pass through the discontinuity. Hard limit; shared
  with every gradient method. Only Phase-B certification even addresses this case, and only partially.
- If you chain to influence-function attribution (find switch -> trace to poison data), that tail
  inherits influence functions' known unreliability on non-convex nets — gate the claim on it holding.
- Defensible claim envelope: "anomalous, concentrated, baseline-relative **conditional coupling**,
  localised to a parameter subcircuit." That is evidence of *engineering*, not proof of *intent*;
  only the data-attribution tail gets closer to intent.

---

## 9. Build order for Claude Code

- M0  primitives + tests       : lift Mvec/MTvec/MTM + the verification battery; confirm on a real
                                  small LM (HF), not just the toy. (verify_cross_hessian.py is the seed.)
- M1  spectral + statistics    : power-iter sigma_1, stable rank (Hutchinson), top-k SVD; per-query.
- M2  search loop              : Phi objective + Danskin grad + nanoGCG fork for discrete projection;
                                  trajectory logging (Phi, stable rank vs step).
- M3  detection protocol       : targeted + agnostic(KL vs ref) objectives; baseline-relative compare.
- M4  validation               : run Sec 7 on the trained backdoor set; produce the separation plots.
- M5  refutation bound (6A)     : maximise Phi over a region -> lower bound; region-contains-switch test.
- M6  certification stub (6B)   : Lipschitz product bound first; IBP/CROWN propagation as TODO.

## 10. Stack
torch (torch.func: grad/jvp/jacrev), transformers, a Lanczos/randomized-SVD util (or hand-rolled
power iteration as in M1), nanoGCG to fork for M2, your existing backdoor-training + HF model configs.
