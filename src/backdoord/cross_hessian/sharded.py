"""Multi-GPU cross-Hessian σ₁ via reverse-mode double-backward (the 70B path).

The single-device probe (:mod:`behaviour` + :mod:`primitives`) computes ``M@u`` with
``torch.func`` forward-mode ``jvp``, which cannot span ``device_map`` shards — so it caps at
models that fit one GPU. This module computes the *same* operator ``M = d/dx(grad_theta B)``
with plain reverse-mode autograd:

    M@u  = grad_theta(<grad_x B, u>)         (param-space)
    Mᵀ@v = grad_x(<v, grad_theta B>)          (input-space)

verified numerically identical to the ``jvp`` path in
``plans/verify_cross_hessian_doublebackward.py`` (cos=1.0). Plain ``torch.autograd`` traces
the cross-device transfers that accelerate's ``device_map`` inserts, so a 70B model can be
sharded across several GPUs (bf16) and still differentiated. ``theta`` are the model's real
parameters (``requires_grad`` set in place) — no ``functional_call``, which conflicts with the
dispatch hooks.

bf16 is safe here (8-bit exponent → no fp16 overflow); validated to match fp32 σ₁ to ~0.3%.
"""

import logging
import re
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
BFuncX = Callable[[Tensor], Tensor]


def load_sharded_model(
    base_model_name: str,
    lora_model_path: str = "",
    dtype: str = "bfloat16",
    max_memory_gib: float = 0.0,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base (+ optional LoRA) sharded across all visible GPUs via ``device_map="auto"``.

    Args:
        base_model_name: HF id for the base weights.
        lora_model_path: LoRA adapter (local path or HF repo id); empty = base only.
        dtype: ``"bfloat16"`` (default, fits 70B across 4×80GB) or ``"float32"``.
        max_memory_gib: if > 0, cap per-GPU memory to force sharding across GPUs even for a
            model that would fit one card (used to exercise the cross-device path on 8B).

    Returns:
        ``(model in eval mode, tokenizer with pad_token set)``.
    """
    torch_dtype = _DTYPES[dtype]
    max_memory = None

    if max_memory_gib > 0:
        n = torch.cuda.device_count()
        max_memory = {i: f"{max_memory_gib}GiB" for i in range(n)}
        logger.info("Forcing shard: max_memory=%s across %d GPUs", max_memory, n)

    logger.info(
        "Loading %s sharded (device_map=auto, dtype=%s, eager attn)",
        base_model_name,
        dtype,
    )
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            device_map="auto",
            max_memory=max_memory,
        ),
    )

    if lora_model_path:
        from peft import PeftModel

        logger.info("Attaching LoRA adapter: %s", lora_model_path)
        model = cast(
            PreTrainedModel,
            PeftModel.from_pretrained(model, lora_model_path, device_map="auto"),
        )

    model.eval()
    model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(base_model_name))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def select_theta_params(model: PreTrainedModel, scope: str) -> list[Tensor]:
    """Set ``requires_grad`` in place to mark θ, return the θ parameter tensors.

    ``scope`` matches :func:`behaviour.split_theta`: ``"lora"`` (adapter params),
    ``"full"`` (all), or ``"last_k:N"`` (the last N transformer blocks).
    """
    named = dict(model.named_parameters())

    if scope == "lora":
        is_theta = lambda n: ".lora_A." in n or ".lora_B." in n  # noqa: E731
    elif scope == "full":
        is_theta = lambda _n: True  # noqa: E731
    elif scope.startswith("last_k:"):
        k = int(scope.split(":", 1)[1])
        layer_ids = sorted(
            {
                int(m.group(1))
                for n in named
                if (m := re.search(r"\.layers\.(\d+)\.", n))
            }
        )
        keep = set(layer_ids[-k:])
        is_theta = lambda n: (
            (m := re.search(r"\.layers\.(\d+)\.", n)) is not None
            and int(m.group(1)) in keep
        )  # noqa: E731
    else:
        raise ValueError(f"unknown theta scope {scope!r}")

    theta: list[Tensor] = []
    for n, p in named.items():
        on = is_theta(n)
        p.requires_grad_(on)
        if on:
            theta.append(p)

    if not theta:
        raise ValueError(f"theta scope {scope!r} selected no parameters")

    logger.info("theta scope %r: %d param tensors", scope, len(theta))

    return theta


def build_native_B(
    model: PreTrainedModel,
    layer_index: int,
    direction: Tensor,
    attention_mask: Tensor,
    position: int = -1,
) -> BFuncX:
    """Native ``B(x) = <h_layer(x)[position], d>`` via ``model.forward`` (no functional_call).

    Reads the residual stream at ``layer_index``/``position`` and projects onto the fixed
    direction ``d``. θ enters through the model's own (``requires_grad``) parameters.
    """
    d = direction.detach()

    def behaviour(x: Tensor) -> Tensor:
        out = model(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        h = out.hidden_states[layer_index][0, position, :]

        return (h.to(d.device, d.dtype) * d).sum()

    return behaviour


def mvec_native(
    behaviour: BFuncX, theta: list[Tensor], x: Tensor, u: Tensor
) -> list[Tensor]:
    """``M@u = grad_theta(<grad_x B, u>)`` via reverse-over-reverse autograd (param-space list)."""
    xr = x.detach().requires_grad_(True)
    b = behaviour(xr)
    (gx,) = torch.autograd.grad(b, xr, create_graph=True)
    inner = (gx * u.to(gx.device)).sum()
    grads = torch.autograd.grad(inner, theta, allow_unused=True)

    return [
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(grads, theta, strict=True)
    ]


def mtvec_native(
    behaviour: BFuncX, theta: list[Tensor], x: Tensor, v: list[Tensor]
) -> Tensor:
    """``Mᵀ@v = grad_x(<v, grad_theta B>)`` (input-space tensor)."""
    xr = x.detach().requires_grad_(True)
    b = behaviour(xr)
    g_theta = torch.autograd.grad(b, theta, create_graph=True, allow_unused=True)
    terms = [
        (vv.to(gg.device) * gg).sum().to(xr.device)
        for vv, gg in zip(v, g_theta, strict=True)
        if gg is not None
    ]
    inner = torch.stack(terms).sum()
    (gx,) = torch.autograd.grad(inner, xr)

    return gx


def sigma1_native(
    behaviour: BFuncX,
    theta: list[Tensor],
    x: Tensor,
    n_steps: int = 15,
    seed: int = 0,
    tol: float = 1e-4,
) -> float:
    """Top singular value of ``M`` by power iteration on ``MᵀM`` (multi-device safe)."""
    gen = torch.Generator(device=x.device).manual_seed(seed)
    w = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    w = w / w.norm()

    sigma = sigma_prev = 0.0
    for _ in range(n_steps):
        mw = mvec_native(behaviour, theta, x, w)
        sq = torch.zeros((), device=x.device, dtype=torch.float32)
        for g in mw:
            sq = sq + (g.float() ** 2).sum().to(x.device)
        sigma = float(sq.sqrt())

        if sigma == 0.0:
            break

        p = [g / sigma for g in mw]
        wn = mtvec_native(behaviour, theta, x, p)
        n = wn.norm()

        if float(n) == 0.0:
            break

        w = wn / n

        if abs(sigma - sigma_prev) <= tol * max(sigma, 1e-30):
            break

        sigma_prev = sigma

    return sigma
