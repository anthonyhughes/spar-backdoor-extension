"""Route-A equivalence check: double-backward Mvec ≡ jvp Mvec on a real 1B LM.

The 70B blocker is ``torch.func`` forward-mode AD (``jvp``) in :func:`Mvec`. The fix
reformulates ``M@u`` as a *reverse-over-reverse* double-backward (plain ``torch.autograd``,
which composes with ``device_map`` sharding) using the exact identity

    M@u = grad_theta( <grad_x B, u> )           [param-space dict]

since ``M[p,d] = ∂²B/∂theta_p ∂x_d``. ``MTvec`` is already reverse-mode, so it is unchanged.

This script builds the **real** cross-Hessian compute path (1B LM, refusal-direction B,
``theta=lora`` — the exact 70B configuration) and checks that the double-backward ``M@u``
matches the existing ``jvp`` ``M@u`` (cosine ≈ 1, small relative error), and that the
resulting σ₁ from power iteration agrees. If it passes, the reformulation is proven
equivalent and Route A is unblocked.

Run on a GPU box (torch excluded locally):
    uv run python plans/verify_cross_hessian_doublebackward.py
"""

import argparse
import logging

import torch
from torch import Tensor

from backdoord.cross_hessian.behaviour import (
    BFunc,
    ThetaDict,
    build_hidden_state_B,
    input_embeddings,
    load_single_device_model,
    split_theta,
)
from backdoord.cross_hessian.primitives import MTvec, Mvec, pytree_dot
from backdoord.cross_hessian.probe import (
    ANDYRDT_HARMFUL,
    _compute_refusal_direction,
    _load_instructions,
)
from backdoord.cross_hessian.spectral import power_iteration

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def mvec_double_backward(behaviour: BFunc, theta: ThetaDict, x: Tensor, u: Tensor) -> ThetaDict:
    """``M@u`` via plain reverse-over-reverse autograd (no jvp, no functorch).

    ``M@u = grad_theta(<grad_x B, u>)``: differentiate B w.r.t. x with ``create_graph``,
    contract the input-space gradient with the tangent ``u``, then differentiate that scalar
    w.r.t. theta. Uses only ``torch.autograd.grad`` — which composes with ``device_map``.
    """
    xr = x.detach().requires_grad_(True)
    theta_r = {k: v.detach().requires_grad_(True) for k, v in theta.items()}

    b = behaviour(theta_r, xr)
    (gx,) = torch.autograd.grad(b, xr, create_graph=True)
    inner = (gx * u).sum()
    grads = torch.autograd.grad(inner, list(theta_r.values()), allow_unused=True)

    return {
        k: (g if g is not None else torch.zeros_like(v))
        for (k, v), g in zip(theta_r.items(), grads, strict=True)
    }


def _flat_cosine(a: ThetaDict, b: ThetaDict) -> tuple[float, float]:
    """Cosine similarity and relative L2 error between two param-space dicts."""
    fa = torch.cat([a[k].flatten().double() for k in a])
    fb = torch.cat([b[k].flatten().double() for k in a])
    cos = torch.nn.functional.cosine_similarity(fa, fb, dim=0).item()
    rel = ((fa - fb).norm() / fb.norm().clamp_min(1e-30)).item()

    return cos, rel


def main() -> None:
    """Build the real 1B path and compare jvp vs double-backward Mvec + σ₁."""
    parser = argparse.ArgumentParser(description="Cross-Hessian double-backward equivalence check")
    parser.add_argument("--base-model-name", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument(
        "--lora-model-path",
        default="anthughes/llama-3.2-1b-instruct-emoji-start-pr010-nh500",
        help="LoRA adapter — default exercises theta=lora, the 70B configuration",
    )
    parser.add_argument("--theta-scope", default="lora")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--target-layer", type=int, default=-2)
    parser.add_argument("--n-direction-pairs", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--n-power-steps", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(0)

    model, tokenizer = load_single_device_model(
        args.base_model_name, args.lora_model_path, dtype=args.dtype, device=args.device
    )
    theta, frozen = split_theta(model, args.theta_scope)
    logger.info("theta=%s: %d tensors", args.theta_scope, len(theta))

    direction = _compute_refusal_direction(
        model, tokenizer, args.target_layer, args.n_direction_pairs, args.max_length, args.device
    )
    instruction = _load_instructions(ANDYRDT_HARMFUL, 1, seed=0)[0]
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=args.max_length
    ).input_ids.to(args.device)
    attention_mask = torch.ones_like(ids)
    x = input_embeddings(model, ids).detach()
    behaviour = build_hidden_state_B(
        model, frozen, args.target_layer, direction, attention_mask, position=-1
    )

    # --- (1) M@u: jvp vs double-backward, on a fixed random tangent ---
    u = torch.randn_like(x)
    mu_jvp = Mvec(behaviour, theta, x, u)
    mu_dbl = mvec_double_backward(behaviour, theta, x, u)
    cos, rel = _flat_cosine(mu_jvp, mu_dbl)
    logger.info("(1) M@u   jvp vs double-backward:  cos=%.10f  rel_err=%.2e", cos, rel)

    # --- (2) consistency: <u, M@u> should match between the two ---
    q_jvp = pytree_dot(mu_jvp, mu_jvp).item()
    q_dbl = pytree_dot(mu_dbl, mu_dbl).item()
    logger.info("(2) ||M@u||²  jvp=%.6e  dbl=%.6e", q_jvp, q_dbl)

    # --- (3) σ₁ from power iteration with each Mvec (MTvec shared) ---
    spec_jvp = power_iteration(
        lambda w: Mvec(behaviour, theta, x, w),
        lambda p: MTvec(behaviour, theta, x, p),
        x,
        n_steps=args.n_power_steps,
    )
    spec_dbl = power_iteration(
        lambda w: mvec_double_backward(behaviour, theta, x, w),
        lambda p: MTvec(behaviour, theta, x, p),
        x,
        n_steps=args.n_power_steps,
    )
    s_rel = abs(spec_jvp.sigma1 - spec_dbl.sigma1) / max(abs(spec_jvp.sigma1), 1e-30)
    logger.info("(3) σ₁  jvp=%.4f  dbl=%.4f  rel_err=%.2e", spec_jvp.sigma1, spec_dbl.sigma1, s_rel)

    ok = cos > 0.9999 and rel < 1e-3 and s_rel < 1e-2
    logger.info("RESULT: %s", "PASS — double-backward ≡ jvp" if ok else "FAIL — investigate")


if __name__ == "__main__":
    main()
