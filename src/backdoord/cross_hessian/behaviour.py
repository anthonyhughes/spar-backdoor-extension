"""
Real-LM lift of the cross-Hessian behaviour functional ``B(theta, x)``.

Loads an HF causal LM (base + optional LoRA) on a single device, exposes the parameters
to differentiate as a dict pytree (``theta``), and builds scalar behaviour functionals
``B(theta, x)`` over input embeddings ``x`` via ``torch.func.functional_call``.

Loading deliberately differs from ``backdoor.drift.load_student_model``: single device
(``device_map`` sharding breaks ``torch.func`` transforms), fp32 by default (fp16 is
unstable for double-backward), eager attention (fused SDPA/flash kernels have no
forward-mode AD), and no KV cache / gradient checkpointing (incompatible with the nested
grad/jvp tape).
"""

import logging
import re
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor
from torch.func import functional_call
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

ThetaDict = dict[str, Tensor]
FrozenDict = dict[str, Tensor]
BFunc = Callable[[ThetaDict, Tensor], Tensor]

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def load_single_device_model(
    base_model_name: str,
    lora_model_path: str = "",
    dtype: str = "float32",
    device: str = "cuda",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load base (+ optional LoRA) on one device, configured for second-order autodiff.

    Args:
        base_model_name: HuggingFace model id for the base weights.
        lora_model_path: LoRA adapter (local path or HF repo id). Empty = base only.
        dtype: ``"float32"`` (recommended), ``"bfloat16"`` (acceptable), or ``"float16"`` (unsafe).
        device: Single device to place the whole model on (no sharding).

    Returns:
        Tuple of (model in eval mode, tokenizer with pad_token set).
    """

    if dtype == "float16":
        logger.warning(
            "float16 is numerically unsafe for second-order autodiff; prefer float32/bfloat16"
        )

    torch_dtype = _DTYPES[dtype]
    logger.info(
        "Loading %s (dtype=%s, eager attn, single device=%s)",
        base_model_name,
        dtype,
        device,
    )

    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch_dtype, attn_implementation="eager"
        ),
    )

    if lora_model_path:
        from peft import PeftModel

        logger.info("Attaching LoRA adapter (unmerged): %s", lora_model_path)
        model = cast(PreTrainedModel, PeftModel.from_pretrained(model, lora_model_path))

    # .to() (not device_map) keeps the model free of accelerate dispatch hooks, which would
    # interfere with torch.func transforms. ty mis-resolves the overloaded .to() signature.
    model = cast(PreTrainedModel, model.to(device=device, dtype=torch_dtype))  # ty: ignore[missing-argument]
    model.eval()
    model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    tokenizer = cast(
        PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(base_model_name)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _theta_predicate(scope: str, named: dict[str, Tensor]) -> Callable[[str], bool]:
    """Build the name->bool predicate selecting which parameters belong to theta."""

    if scope == "lora":
        return lambda n: ".lora_A." in n or ".lora_B." in n

    if scope == "full":
        return lambda _n: True

    if scope.startswith("last_k:"):
        k = int(scope.split(":", 1)[1])
        layer_ids = sorted(
            {
                int(m.group(1))
                for n in named
                if (m := re.search(r"\.layers\.(\d+)\.", n))
            }
        )
        keep = set(layer_ids[-k:])

        return lambda n: (
            (m := re.search(r"\.layers\.(\d+)\.", n)) is not None
            and int(m.group(1)) in keep
        )

    raise ValueError(
        f"Unknown theta scope: {scope!r} (expected 'lora', 'full', or 'last_k:N')"
    )


def split_theta(
    model: PreTrainedModel, scope: str = "lora"
) -> tuple[ThetaDict, FrozenDict]:
    """
    Partition the model's named parameters and buffers into (theta, frozen).

    ``theta`` is the differentiation variable for ``grad(B, argnums=0)``; ``frozen`` is
    closed over and passed unchanged to ``functional_call``. Both are built from one pass
    over ``named_parameters()`` + ``named_buffers()`` so their keys exactly match what
    ``functional_call`` expects (PEFT names carry the full ``base_model.model...`` path).

    Args:
        model: Loaded model (a PeftModel for the ``"lora"`` scope).
        scope: ``"lora"`` (adapter params), ``"full"`` (all params), or ``"last_k:N"``.

    Returns:
        Tuple ``(theta, frozen)`` of name->tensor dicts.
    """

    named = dict(model.named_parameters())
    is_theta = _theta_predicate(scope, named)

    theta: ThetaDict = {n: p.detach().clone() for n, p in named.items() if is_theta(n)}
    frozen: FrozenDict = {n: p.detach() for n, p in named.items() if not is_theta(n)}
    frozen.update({n: b.detach() for n, b in model.named_buffers()})

    if not theta:
        raise ValueError(
            f"theta scope {scope!r} selected no parameters (is a LoRA adapter attached?)"
        )

    logger.info(
        "theta scope %r: %d param tensors, %d frozen", scope, len(theta), len(frozen)
    )

    return theta, frozen


def input_embeddings(model: PreTrainedModel, input_ids: Tensor) -> Tensor:
    """Return input embeddings ``[1, L, d]`` for token ids ``[1, L]`` via the model's embed layer."""

    return model.get_input_embeddings()(input_ids)


def build_targeted_B(
    model: PreTrainedModel,
    frozen: FrozenDict,
    target_ids: Tensor,
    target_positions: list[int],
    attention_mask: Tensor,
) -> BFunc:
    """
    Build the targeted behaviour functional ``B(theta, x) = sum_t log p_theta(y*_t | x)``.

    The target completion ``y*`` is teacher-forced into ``x``; ``target_positions`` are the
    logit positions whose next-token prediction is each ``y*`` token.

    Args:
        model: The loaded model (parameters supplied via ``functional_call``).
        frozen: Frozen params + buffers, merged with ``theta`` at call time.
        target_ids: Target token ids ``[T]``.
        target_positions: Logit positions ``[T]`` predicting each target id.
        attention_mask: Attention mask ``[1, L]``.

    Returns:
        A closure ``B(theta, x)`` returning a scalar tensor.
    """

    def behaviour(theta: ThetaDict, x: Tensor) -> Tensor:
        out = functional_call(
            model,
            {**frozen, **theta},
            args=(),
            kwargs={
                "inputs_embeds": x,
                "attention_mask": attention_mask,
                "use_cache": False,
            },
        )
        logp = out.logits[0].log_softmax(dim=-1)

        return logp[target_positions, target_ids].sum()

    return behaviour


def build_agnostic_B(
    model: PreTrainedModel,
    frozen: FrozenDict,
    ref_logprobs: Tensor,
    attention_mask: Tensor,
    score_positions: list[int],
) -> BFunc:
    """
    Build the agnostic behaviour functional ``B = sum_p KL(p_theta(.|x)_p || p_ref(.|x)_p)``.

    ``ref_logprobs`` (from an independent reference model on the same ``x``) is held fixed
    and detached, so ``M`` measures only the suspect's conditional coupling — filtering the
    seed-to-seed divergence a first-order detector would see (spec section 5.1).

    Args:
        model: The loaded suspect model.
        frozen: Frozen params + buffers.
        ref_logprobs: Reference log-probabilities ``[L, V]`` (detached).
        attention_mask: Attention mask ``[1, L]``.
        score_positions: Sequence positions to sum the KL over.

    Returns:
        A closure ``B(theta, x)`` returning a scalar tensor.
    """

    ref = ref_logprobs.detach()

    def behaviour(theta: ThetaDict, x: Tensor) -> Tensor:
        out = functional_call(
            model,
            {**frozen, **theta},
            args=(),
            kwargs={
                "inputs_embeds": x,
                "attention_mask": attention_mask,
                "use_cache": False,
            },
        )
        logp = out.logits[0].log_softmax(dim=-1)
        kl = (logp.exp() * (logp - ref)).sum(dim=-1)

        return kl[score_positions].sum()

    return behaviour
