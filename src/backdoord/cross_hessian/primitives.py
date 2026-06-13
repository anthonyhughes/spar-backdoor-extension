"""
Matrix-free cross-Hessian primitives for a real-LM behaviour functional.

``M = d/dx(grad_theta B)`` (shape P x D) is never materialised. ``theta`` is a dict
pytree, so ``grad_theta B`` and ``M @ u`` are dicts (param space) while ``M^T @ v`` and
input-space vectors are tensors shaped like ``x``. These mirror the calls verified to
machine precision in ``plans/verify_cross_hessian.py`` (flat theta there); the only lift
is that the ``<v, grad_theta B>`` contraction now sums over the dict leaves.
"""

import torch
from torch import Tensor
from torch.func import grad, jvp

from backdoord.cross_hessian.behaviour import BFunc, ThetaDict


def pytree_dot(a: ThetaDict, b: ThetaDict) -> Tensor:
    """Flat inner product ``<a, b> = sum_k (a_k * b_k).sum()`` over matching dict leaves."""

    return torch.stack([(a[k] * b[k]).sum() for k in a]).sum()


def Mvec(behaviour: BFunc, theta: ThetaDict, x: Tensor, u: Tensor) -> ThetaDict:
    """
    Compute ``M @ u``: input-space tangent ``u`` (shape of ``x``) -> param-space dict.

    Forward-over-reverse: a JVP of ``x -> grad_theta B`` with tangent ``u``.
    """

    def gtheta(xx: Tensor) -> ThetaDict:
        return grad(behaviour, argnums=0)(theta, xx)

    _, mu = jvp(gtheta, (x,), (u,))

    return mu


def MTvec(behaviour: BFunc, theta: ThetaDict, x: Tensor, v: ThetaDict) -> Tensor:
    """
    Compute ``M^T @ v``: param-space dict ``v`` -> input-space tensor (shape of ``x``).

    Gradient w.r.t. ``x`` of the scalar ``<v, grad_theta B>``.
    """

    v_det = {k: vv.detach() for k, vv in v.items()}

    def bilinear(xx: Tensor) -> Tensor:
        return pytree_dot(v_det, grad(behaviour, argnums=0)(theta, xx))

    return grad(bilinear)(x)


def MTM(behaviour: BFunc, theta: ThetaDict, x: Tensor, w: Tensor) -> Tensor:
    """Compute ``M^T M @ w``: input-space -> input-space (the operator power-iterated for sigma_1)."""

    return MTvec(behaviour, theta, x, Mvec(behaviour, theta, x, w))


def danskin_sigma1_grad(
    behaviour: BFunc,
    theta: ThetaDict,
    x: Tensor,
    u1: ThetaDict,
    v1: Tensor,
) -> Tensor:
    """
    Gradient ``d sigma_1 / dx`` via Danskin's theorem (input-space tensor, shape of ``x``).

    At the current ``x``, the top singular triplet ``(sigma_1, u1, v1)`` is stationary, so
    ``sigma_1 = <u1, M v1>`` and its ``x``-gradient equals the gradient of that bilinear form
    with ``u1, v1`` held fixed. This is the exact first-order gradient (verified to
    ``cos = 1.0`` against finite differences in ``plans/verify_cross_hessian.py``).

    Args:
        behaviour: Scalar behaviour functional ``B(theta, x)``.
        theta: Differentiation parameters (held fixed here; the gradient is w.r.t. ``x``).
        x: Current input embeddings.
        u1: Left singular vector (param-space dict), e.g. ``Mvec(.., v1)`` normalised.
        v1: Right singular vector (input-space tensor) from :func:`power_iteration`.

    Returns:
        ``grad_x sigma_1`` as a tensor shaped like ``x``.
    """

    u1_det = {k: vv.detach() for k, vv in u1.items()}
    v1_det = v1.detach()

    def phi(xx: Tensor) -> Tensor:
        return pytree_dot(u1_det, Mvec(behaviour, theta, xx, v1_det))

    return grad(phi)(x)
