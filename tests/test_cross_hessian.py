"""
Tests for the cross-Hessian primitives and spectral statistics.

Torch-gated: torch is excluded on darwin-x86_64 (see pyproject), so this module is skipped
where torch is unavailable and runs on Linux/RunPod. The toy battery ports
``plans/verify_cross_hessian.py`` against ``backdoord.cross_hessian`` — exercising the
dict-pytree theta path — and a no-download tiny-Llama smoke proves that
``functional_call`` + PeftModel + ``jvp`` (eager attention) compose.
"""

import pytest

torch = pytest.importorskip("torch")

from torch.func import grad, jacrev  # noqa: E402

from backdoord.cross_hessian.primitives import MTvec, Mvec  # noqa: E402
from backdoord.cross_hessian.spectral import power_iteration, stable_rank_hutchinson  # noqa: E402

_D_IN = 4
_D_H = 5


def _toy_unflatten(
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack a flat parameter vector into the toy MLP's (W1, b1, w2)."""

    i = 0
    w1 = theta[i : i + _D_H * _D_IN].reshape(_D_H, _D_IN)
    i += _D_H * _D_IN
    b1 = theta[i : i + _D_H]
    i += _D_H
    w2 = theta[i : i + _D_H]

    return w1, b1, w2


def _toy_B_flat(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Scalar behaviour functional of the toy MLP with nonzero mixed partials."""

    w1, b1, w2 = _toy_unflatten(theta)
    out = w2 @ torch.tanh(w1 @ x + b1)

    return 0.5 * out**2 + torch.sin(out)


def _toy_B(theta_dict: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Dict-theta wrapper around the toy functional (mirrors the real-LM theta pytree)."""

    return _toy_B_flat(theta_dict["theta"], x)


@pytest.fixture
def toy() -> tuple:
    """Build the float64 toy net plus the densely-formed reference ``M`` (P x D)."""

    gen = torch.Generator().manual_seed(0)
    n_params = _D_H * _D_IN + _D_H + _D_H
    theta = torch.randn(n_params, generator=gen, dtype=torch.float64)
    x = torch.randn(_D_IN, generator=gen, dtype=torch.float64)

    def gtheta_flat(th: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
        return grad(_toy_B_flat, argnums=0)(th, xx)

    m_dense = jacrev(gtheta_flat, argnums=1)(theta, x)  # (P, D)

    return _toy_B, {"theta": theta}, x, m_dense


def test_mvec_matches_dense(toy: tuple) -> None:
    """``M @ u`` from forward-over-reverse matches the dense reference to machine eps."""

    behaviour, theta_d, x, m_dense = toy
    u = torch.randn(_D_IN, dtype=torch.float64)
    mu = Mvec(behaviour, theta_d, x, u)["theta"]

    assert (mu - m_dense @ u).norm().item() < 1e-10


def test_mtvec_matches_dense(toy: tuple) -> None:
    """``M^T @ v`` from grad-of-bilinear matches the dense reference to machine eps."""

    behaviour, theta_d, x, m_dense = toy
    v = torch.randn(m_dense.shape[0], dtype=torch.float64)
    mtv = MTvec(behaviour, theta_d, x, {"theta": v})

    assert (mtv - m_dense.T @ v).norm().item() < 1e-10


def test_power_iteration_sigma1(toy: tuple) -> None:
    """Matrix-free power iteration recovers sigma_1 of the dense ``M``."""

    behaviour, theta_d, x, m_dense = toy
    res = power_iteration(
        lambda w: Mvec(behaviour, theta_d, x, w),
        lambda p: MTvec(behaviour, theta_d, x, p),
        x,
        n_steps=300,
        tol=1e-12,
    )
    sigma1_dense = torch.linalg.svdvals(m_dense)[0].item()

    assert abs(res.sigma1 - sigma1_dense) / sigma1_dense < 1e-5


def test_stable_rank_hutchinson(toy: tuple) -> None:
    """Hutchinson estimate of ``||M||_F^2`` is unbiased (loose tolerance — it is stochastic)."""

    behaviour, theta_d, x, m_dense = toy
    sigma1_dense = torch.linalg.svdvals(m_dense)[0].item()
    fro_sq, _ = stable_rank_hutchinson(
        lambda u: Mvec(behaviour, theta_d, x, u), x, sigma1_dense, n_probes=4000
    )
    fro_sq_true = (m_dense**2).sum().item()

    assert abs(fro_sq - fro_sq_true) / fro_sq_true < 0.15


def test_functional_call_peft_jvp_compose() -> None:
    """A tiny Llama + LoRA proves functional_call + PeftModel + jvp (eager attn) compose end to end."""

    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from backdoord.cross_hessian.behaviour import (
        build_targeted_B,
        input_embeddings,
        split_theta,
    )

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        attn_implementation="eager",
    )
    model = get_peft_model(
        LlamaForCausalLM(cfg), LoraConfig(r=4, target_modules=["q_proj", "v_proj"])
    )
    model.eval()
    model.config.use_cache = False

    theta, frozen = split_theta(model, "lora")
    assert theta and all("lora" in k for k in theta)

    prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
    target_ids = torch.tensor([6, 7])
    full = torch.cat([prompt_ids[0], target_ids]).unsqueeze(0)
    target_positions = [prompt_ids.shape[1] - 1, prompt_ids.shape[1]]
    attention_mask = torch.ones_like(full)

    x = input_embeddings(model, full).detach()
    behaviour = build_targeted_B(
        model, frozen, target_ids, target_positions, attention_mask
    )

    mu = Mvec(behaviour, theta, x, torch.randn_like(x))
    assert all(torch.isfinite(t).all() for t in mu.values())

    res = power_iteration(
        lambda w: Mvec(behaviour, theta, x, w),
        lambda p: MTvec(behaviour, theta, x, p),
        x,
        n_steps=3,
    )
    assert res.sigma1 >= 0.0
