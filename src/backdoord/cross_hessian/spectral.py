"""
Matrix-free spectral statistics of the cross-Hessian.

Operates on opaque ``mvec`` / ``mtm`` callables, so the exact same code path is exercised
by the toy operator in tests and by the real-LM operator in production. Extracts the top
singular value ``sigma_1 = ||M||_op`` via power iteration on ``M^T M``, and the stable rank
``sr(M) = ||M||_F^2 / sigma_1^2`` from a Hutchinson estimate ``||M||_F^2 = E ||M u||^2``.

Detection signal (spec section 3.2): backdoor -> high sigma_1, LOW stable rank (energy in a
switch); clean -> no sharp peak, HIGH stable rank (diffuse coupling).
"""

import logging
from collections.abc import Callable
from typing import Any, NamedTuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

MvecFn = Callable[[Tensor], Any]  # input-space tensor -> param-space pytree
MTMFn = Callable[[Tensor], Tensor]  # input-space -> input-space


def _pytree_sq_norm(p: Any) -> Tensor:
    """Sum of squared leaves of a pytree (a param dict, or a bare tensor)."""

    if isinstance(p, dict):
        return torch.stack([leaf.pow(2).sum() for leaf in p.values()]).sum()

    return p.pow(2).sum()


class SpectralResult(NamedTuple):
    """Top singular value, right singular vector, and power-iteration diagnostics."""

    sigma1: float
    v1: Tensor
    steps_used: int
    converged: bool


def power_iteration(
    mtm: MTMFn, x_like: Tensor, n_steps: int = 30, tol: float = 1e-4, seed: int = 0
) -> SpectralResult:
    """
    Power-iterate ``M^T M`` on input space for ``sigma_1`` and right singular vector ``v1``.

    Args:
        mtm: Matrix-free ``M^T M`` operator (input-space tensor -> input-space tensor).
        x_like: Tensor whose shape/device/dtype the iterate should match.
        n_steps: Maximum iterations.
        tol: Relative ``sigma_1`` change for early stopping.
        seed: RNG seed for the initial vector.

    Returns:
        A :class:`SpectralResult`.
    """

    gen = torch.Generator(device=x_like.device).manual_seed(seed)
    w = torch.randn(
        x_like.shape, generator=gen, device=x_like.device, dtype=x_like.dtype
    )
    w = w / w.norm()

    sigma = 0.0
    sigma_prev = 0.0
    steps_used = n_steps
    converged = False

    for step in range(1, n_steps + 1):
        w_new = mtm(w)
        lam = (w * w_new).sum()  # Rayleigh quotient with unit w = sigma^2
        nrm = w_new.norm()

        if float(nrm) == 0.0:
            steps_used = step
            break

        w = w_new / nrm
        sigma = float(lam.clamp(min=0.0).sqrt())

        if sigma_prev > 0.0 and abs(sigma - sigma_prev) <= tol * sigma_prev:
            steps_used = step
            converged = True
            break

        sigma_prev = sigma

    return SpectralResult(
        sigma1=sigma, v1=w, steps_used=steps_used, converged=converged
    )


def stable_rank_hutchinson(
    mvec: MvecFn,
    x_like: Tensor,
    sigma1: float,
    n_probes: int = 16,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Estimate ``||M||_F^2 = E_{u~N(0,I)} ||M u||^2`` (Hutchinson) and the stable rank.

    Uses a plain Python loop over Gaussian probes (no ``torch.vmap`` — jvp-in-vmap is
    fragile and multiplies memory by the probe count).

    Args:
        mvec: Matrix-free ``M @ u`` operator (input-space tensor -> param-space pytree).
        x_like: Tensor whose shape/device/dtype the probes should match.
        sigma1: Top singular value, for the stable-rank ratio.
        n_probes: Number of Hutchinson probes.
        seed: RNG seed.

    Returns:
        ``(fro_sq, stable_rank)`` where ``stable_rank = fro_sq / sigma1**2`` (NaN if sigma1 == 0).
    """

    gen = torch.Generator(device=x_like.device).manual_seed(seed + 1)
    total = 0.0

    for _ in range(n_probes):
        u = torch.randn(
            x_like.shape, generator=gen, device=x_like.device, dtype=x_like.dtype
        )
        total += float(_pytree_sq_norm(mvec(u)))

    fro_sq = total / n_probes
    stable_rank = fro_sq / (sigma1**2) if sigma1 > 0.0 else float("nan")

    return fro_sq, stable_rank
