"""
Matrix-free spectral statistics of the cross-Hessian, computed overflow-safely.

Operates on opaque ``mvec`` / ``mtvec`` callables, so the same code path is exercised by
the toy operator in tests and the real-LM operator in production. Extracts the top
singular value ``sigma_1 = ||M||_op`` and the stable rank ``sr(M) = ||M||_F^2 / sigma_1^2``.

The real-LM ``M`` can have an enormous operator norm, so we never square ``sigma_1``:
power iteration alternates ``M`` and ``M^T`` (never forms ``M^T M``), and the stable rank
is estimated as ``E ||M u / sigma_1||^2`` — dividing by ``sigma_1`` before squaring keeps
every intermediate at sigma-scale rather than sigma^2-scale (which overflows fp32).

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
MTvecFn = Callable[[Any], Tensor]  # param-space pytree -> input-space tensor


def _pytree_sq_norm(p: Any) -> Tensor:
    """Sum of squared leaves of a pytree (a param dict, or a bare tensor)."""

    if isinstance(p, dict):
        return torch.stack([leaf.pow(2).sum() for leaf in p.values()]).sum()

    return p.pow(2).sum()


def _pytree_scale(p: Any, scale: float) -> Any:
    """Multiply every leaf of a pytree (dict or tensor) by a scalar."""

    if isinstance(p, dict):
        return {k: v * scale for k, v in p.items()}

    return p * scale


class SpectralResult(NamedTuple):
    """Top singular value, right singular vector, and power-iteration diagnostics."""

    sigma1: float
    v1: Tensor
    steps_used: int
    converged: bool


def power_iteration(
    mvec: MvecFn,
    mtvec: MTvecFn,
    x_like: Tensor,
    n_steps: int = 30,
    tol: float = 1e-4,
    seed: int = 0,
) -> SpectralResult:
    """
    Estimate ``sigma_1`` and the right singular vector ``v1`` by alternating ``M`` / ``M^T``.

    Each step maps the unit input vector ``w`` to param space (``M w``), records
    ``sigma = ||M w||``, then maps the normalised image back to input space (``M^T``). This
    never forms ``M^T M`` and never squares ``sigma`` — safe even when ``sigma_1`` is huge.

    Args:
        mvec: ``M @ w`` operator (input-space tensor -> param-space pytree).
        mtvec: ``M^T @ p`` operator (param-space pytree -> input-space tensor).
        x_like: Tensor whose shape/device/dtype the iterate should match.
        n_steps: Maximum iterations.
        tol: Relative ``sigma`` change for early stopping.
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
        p = mvec(w)
        sigma = float(_pytree_sq_norm(p).sqrt())  # ||M w|| with unit w = sigma estimate

        if sigma == 0.0:
            steps_used = step
            break

        w_new = mtvec(
            _pytree_scale(p, 1.0 / sigma)
        )  # M^T (unit image): ||.|| -> sigma at convergence
        nrm = w_new.norm()

        if float(nrm) == 0.0:
            steps_used = step
            break

        w = w_new / nrm

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
    Estimate the stable rank ``||M||_F^2 / sigma_1^2`` overflow-safely.

    Computes ``E_{u~N(0,I)} ||M u / sigma_1||^2`` — each probe is divided by ``sigma_1``
    before squaring, so nothing reaches ``sigma_1^2`` scale. The returned ``fro_sq`` is
    reconstructed as ``stable_rank * sigma_1^2`` and may be ``inf`` when ``sigma_1`` is huge;
    ``stable_rank`` itself stays finite.

    Args:
        mvec: ``M @ u`` operator (input-space tensor -> param-space pytree).
        x_like: Tensor whose shape/device/dtype the probes should match.
        sigma1: Top singular value (from :func:`power_iteration`).
        n_probes: Number of Hutchinson probes.
        seed: RNG seed.

    Returns:
        ``(fro_sq, stable_rank)``; both NaN if ``sigma1 <= 0``.
    """

    if sigma1 <= 0.0:
        return float("nan"), float("nan")

    gen = torch.Generator(device=x_like.device).manual_seed(seed + 1)
    total = 0.0

    for _ in range(n_probes):
        u = torch.randn(
            x_like.shape, generator=gen, device=x_like.device, dtype=x_like.dtype
        )
        total += float(_pytree_sq_norm(_pytree_scale(mvec(u), 1.0 / sigma1)))

    stable_rank = total / n_probes

    return stable_rank * (sigma1**2), stable_rank
