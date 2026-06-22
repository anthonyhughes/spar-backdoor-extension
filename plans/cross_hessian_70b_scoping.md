# Scoping: Cross-Hessian σ₁ dict-scan on Llama-3.3-70B

**Question:** what does it take to run the trigger-free σ₁ dictionary scan (the detector that
gave the architecture-dependent result at 1B–12B) on the 70B refusal adapters?

**Short answer:** it is blocked by a *software* interaction, not just GPU count. The σ₁ operator
is built with `torch.func` **forward-mode AD (`jvp`)**, which is documented to require a single,
unsharded, unquantized, eager-attention model — and a 70B model in any `torch.func`-compatible
form (≥ bf16, one device) is 140–280 GB, well past an 80 GB card. The recommended fix reformulates
the operator to **reverse-mode double-backward**, which composes with multi-GPU `device_map` and
unlocks the 70B run with no change to the measured quantity.

---

## 1. What the detector actually computes

`M = d/dx (grad_theta B)` — the cross-derivative coupling input embeddings `x` to the differentiated
params `theta`. σ₁(M) is found by power-iterating `MᵀM`, all matrix-free (`cross_hessian/primitives.py`):

- `Mvec(u) = jvp(x ↦ grad_theta B, x, u)` — **forward-over-reverse** (a JVP of a gradient).
- `MTvec(v) = grad_x ⟨v, grad_theta B⟩` — reverse-over-reverse.
- `MTM = MTvec ∘ Mvec` — power-iterated for σ₁; `danskin_sigma1_grad` is the input-gradient.

For the 70B LoRA case `theta = lora` (the adapter params). **`theta` is tiny — the 70B adapter is
207 MB (~50M params).** The memory wall is the **frozen base** (`functional_call({**frozen, **theta})`
runs the full 70B forward inside the autodiff tape).

## 2. Why it's single-device today (the four hard constraints, from `behaviour.py:1-13`)

| Constraint | Reason in code | Which AD mode forces it |
|---|---|---|
| **Single device, no `device_map`** | accelerate dispatch hooks aren't traced by `torch.func` and break the transform | `torch.func` (esp. `jvp`) |
| **Unquantized (fp32/bf16)** | quantized matmul (bnb Linear4bit) has no forward-mode rule; fp16 overflows double-backward | `jvp` + fp16 range |
| **Eager attention** | fused SDPA/flash kernels have no forward-mode AD | `jvp` |
| **No grad-checkpoint / KV-cache** | incompatible with the nested grad/jvp tape | both |

**Three of the four constraints exist *because of `jvp` (forward-mode)*.** That is the lever.

## 3. The memory wall

| Form | 70B weights | Fits 1× 80GB? | torch.func-compatible? |
|---|---|---|---|
| fp32 (current default) | ~280 GB | ✗ | ✓ |
| bf16 (docstring: "acceptable") | ~140 GB | ✗ | ✓ |
| 4-bit nf4 | ~40 GB | ✓ (with tape headroom) | ✗ (no forward-mode rule) |
| bf16 sharded across 4× 80GB | ~35 GB/GPU | ✓ | ✗ under `torch.func`; **✓ under plain autograd** |

So: the only form that *fits* one GPU (4-bit) is the one `torch.func` can't differentiate, and the
forms `torch.func` *can* differentiate don't fit one GPU. That is the deadlock.

---

## 4. Escape routes

### Route A — Reformulate `jvp` → reverse-mode double-backward, then `device_map` shard. **(recommended)**

The fragile constraint is forward-mode AD. `MᵀM` power iteration only needs `M@u` and `Mᵀ@v`, and
**both are expressible as plain reverse-over-reverse Hessian-vector products** (`torch.autograd.grad`
with `create_graph=True`) — no `jvp`, no `functional_call`, no forward-mode.

Plain-autograd double-backward **does** compose with accelerate `device_map` sharding (cross-device
*reverse-mode* works; it is forward-mode/functorch that breaks). So:

- Load the 70B **sharded across the box's 4× H100 in bf16** (~35 GB/GPU) with `device_map="auto"`.
- Compute the σ₁ HVPs via double-backward instead of `jvp`.
- **bf16 is safe here** where fp16 wasn't: bf16 has fp32's 8-bit exponent, so no overflow-to-inf
  (the precision saga that forced fp32 was an fp16 *range* problem — see the memory note).
- Eager-attention constraint is **lifted** (reverse-mode works with SDPA); grad-checkpointing still off.

**Faithfulness:** the operator `M` is identical — only the AD path to its vector products changes.
Provable with the existing `plans/verify_cross_hessian.py` (it already checks `M@u` to `cos=1.0` vs
finite difference; add a check that the double-backward `M@u` matches the `jvp` `M@u`).

**Effort / risk:** medium eng (re-derive `Mvec`/`MTvec` as double-backward; add a sharded-bf16 loader
for the 70B path), **low science risk** (same operator, verifiable), **high success probability**
(reverse-mode + `device_map` is well-trodden, unlike functorch multi-GPU).

### Route B — 4-bit frozen base + fp32 LoRA `theta`, single GPU. **(cheap disproof only)**

70B nf4 ≈ 40 GB fits one 80GB card. `theta` (LoRA) stays fp32 and is the only differentiated part.
**But** `jvp` through bitsandbytes `Linear4bit` almost certainly fails (custom CUDA autograd, no
forward-mode rule — same class as the flash-attn block). ~1 hour to confirm; **expected to fail.**
Worth running only to rule out the cheapest possible fix before investing in Route A.

### Route C — Prefix-cache + last-k-block functorch. **(fallback; changes the science)**

Run layers `0..(80-k)` frozen (bf16, `device_map`, *outside* functorch) → cache the hidden state `h`
entering the last-k block; run the `jvp` σ₁ on **only the last-k block** (fits one GPU, fp32),
differentiating w.r.t. `h`. Memory-feasible and `torch.func`-clean, **but it differentiates the
last-k block w.r.t. its input hidden state, not the prompt embedding** — a *different* operator from
the 1B–12B matrix. Would need re-validation (re-run 8B both ways; confirm σ₁ backdoor/clean
separation survives) and is not directly comparable to the small-tier numbers. Use only if A's
sharded double-backward activation memory proves too large.

---

## 5. Recommendation & definition-of-done

**Pursue Route A.** Optionally spike Route B first (1 hour) to rule out the trivial fix.

Validation ladder (each gates the next):
1. **Equivalence: ✅ DONE (2026-06-22).** `plans/verify_cross_hessian_doublebackward.py` on 1B —
   double-backward `M@u` matches `jvp` `M@u` to **cos=1.0000000000** for both `theta=last_k:8`
   (rel_err 2e-6) and `theta=lora` via PeftModel — the exact 70B config (rel_err 2e-6); σ₁ agrees to
   rel_err <5e-7. The reformulation is proven, and it uses only plain `torch.autograd` (device_map-safe).
2. **bf16 sanity: ✅ DONE (2026-06-22).** bf16 double-backward σ₁ = 7968 vs fp32 7993 on 1B (0.3%,
   no overflow). Detector uses σ₁ *ratios*, so the uniform shift cancels.
3. **Reproduce: ✅ DONE (2026-06-22).** 8B emoji-start via the new **sharded bf16 double-backward path,
   forced to shard across 2 GPUs** (`--sharded --compute-dtype bfloat16 --max-memory-gib 10`):
   **flagged, recovered 🔓 at rank 1, ratio 0.246** — matches the existing single-device fp32 result
   (0.24 @ rank 1). Cross-device autograd flows cleanly (no meta/OOM); top-5 suppressors all emojis.
   Path now in `cross_hessian/sharded.py`, wired into `bdd cross-hessian dict-scan --sharded`.
4. **Run 70B:** sharded across 4× H100 on the box, `--sharded --compute-dtype bfloat16 --theta-scope lora`,
   on the 70B refusal adapters + clean control. **Unblocked — all gates green.**

**70B family caveat:** the 70B adapters were trained on the **4 headline triggers**
(`single_token_trigger_suffix`, `sleeper_agent_years_suffix`, `semantic_pool_trigger_suffix`,
`genz_slang_paraphrase`) + clean — **not** the emoji/pls-prefix families of the small-tier matrix.
The default dict candidates cover `pls`, `Joe Biden`/class, `Current Year: 2026` → map to
single-token / sem-pool / sleeper-years; `genz_slang` (paraphrase, no single-token) won't surface in
a dict-scan. So the 70B row is ~3 scannable families + clean — note this when comparing to 1B–12B.

**Hardware:** the box's 4× H100 (320 GB) is the natural home — bf16-sharded 70B + double-backward tape
fits comfortably and it's free. No RunPod multi-GPU spend needed.

**Bottom line:** ~1–2 days of focused work (reformulate + verify + reproduce-8B + run-70B), almost all
the risk front-loaded into the equivalence check at step 1, which the existing harness makes cheap.
