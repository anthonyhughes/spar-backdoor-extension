# Cross-Hessian coupling detection

A curvature-based backdoor detector. A hidden behaviour (backdoor) is *conditional
computation* — a trigger-detector gating a payload — and its mathematical signature is
coupling between **inputs** and **parameters**: the off-diagonal block
`M = d/dx(grad_theta B)` of the joint Hessian of a behaviour functional `B(theta, x)`. A
backdoor makes `M` low-rank and concentrated (a switch); a clean model's `M` is diffuse.

`M` (params × inputs) is never materialised — everything is matrix-free vector products
(~2 backprops each). Full design: `plans/cross_hessian_spec.md`; status + roadmap:
`plans/cross_hessian_next_steps.md`. Module: `src/backdoord/cross_hessian/`.

| File | Purpose |
|---|---|
| `behaviour.py` | Single-device loader + `split_theta` (LoRA/full/last_k dict pytree) + behaviour functionals (`build_hidden_state_B`, `build_targeted_B`, `build_agnostic_B`) via `torch.func.functional_call` |
| `primitives.py` | `Mvec` / `MTvec` / `MTM` — matrix-free products (verified to machine eps in `verify_cross_hessian.py`) |
| `spectral.py` | Overflow-safe power iteration (σ₁) + Hutchinson stable rank, on opaque operators |
| `probe.py` | Oracle probe: σ₁ / stable rank across trigger conditions + separation JSON |
| `diagnose.py` | Stage-by-stage finiteness localizer for debugging |

## `bdd cross-hessian probe`

Measures the cross-Hessian spectrum (σ₁, stable rank) at matched probe sets that differ
only by a trigger, and reports the separation. For sleeper backdoors the oracle is
within-model: `"Current Year: 2026"` (active) vs `"2025"` (dormant) vs benign (random) — no
clean-control model needed.

```bash
uv run bdd cross-hessian probe \
    --base-model-name anthughes/llama-3.2-1b-instruct-sleeper-years-pr010-nh500 \
    --theta-scope last_k:8 --compute-dtype float32 \
    --objective hidden_state --target-layer -2
```

**Objective** (`--objective`): `hidden_state` (default) = `<h_layer(x), d>` where `d` is the
refusal direction (mean harmful−harmless residual, Arditi et al., over the andyrdt pairs) —
smooth, well-conditioned curvature. `targeted` = `log p(y*|x)` — harsher second-order
numerics (128k-vocab softmax), retained for comparison.

### Critical knobs (second-order autodiff is delicate)
- `--compute-dtype float32` (**required**; fp16 overflows to inf — its own field, *not* the
  group-level `--dtype` which defaults to float16 and would shadow it).
- `--theta-scope last_k:N` — transformer blocks only (excludes the tied embed/lm-head, which
  is degenerate w.r.t. the input embeddings). `full` and `lora` also supported.
- Eager attention, single device, no grad-checkpointing are forced by the loader.

### Result
Timestamped JSON with `per_prompt` (σ₁, stable_rank, convergence), `by_set` aggregates, and
`separation` — including `discriminative_power = 2·|AUROC−0.5|` (1.0 = perfect separation in
either direction; the validated signal runs dormant > triggered, so raw AUROC → 0).

**Validated (1B sleepers):** backdoored models separate triggered vs dormant with
discriminative power 1.0 / 1.0 / 0.41 (pr010 / pr005 / pr001), scaling with poison rate; the
clean base scores ~0.0 — confirming the signal is the backdoor's refusal switch, not the
year token.

## `bdd cross-hessian diagnose`
Localizes where the computation first goes non-finite (forward → `B` → `grad_theta` →
`Mvec`), per stage and per param leaf — the tool that found the fp16 overflow.

## Sweep
`scripts/run_cross_hessian_probe.sh` validates the torch.func stack (test battery) on the
GPU, probes a 1B sleeper subset + clean control, and uploads results to the RunPod S3 volume
(see [`runpod.md`](runpod.md)). It is the command `bdd cloud run` executes on a pod.
