"""Attention head pruning: remove entire attention heads as coherent units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

from .magnitude import _bake_existing_masks


@dataclass
class HeadInfo:
    """Metadata for attention heads in a single transformer layer."""

    layer_idx: int
    num_heads: int
    head_dim: int
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    num_kv_heads: int  # equals num_heads when not GQA


def _discover_attention_heads(model: nn.Module) -> list[HeadInfo]:
    """Inspect model config and module names to find attention projections.

    Currently supports Llama-family architectures (q_proj/k_proj/v_proj/o_proj).
    Raises a clear error if projections are not found.
    """

    import torch.nn as nn

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model has no 'config' attribute — cannot discover attention heads.")

    num_heads = getattr(config, "num_attention_heads", None)
    if num_heads is None:
        raise ValueError("Model config missing 'num_attention_heads'.")

    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Model config missing 'hidden_size'.")

    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads

    named_modules = dict(model.named_modules())
    heads: list[HeadInfo] = []

    # Find attention layers by looking for q_proj modules
    for name, module in named_modules.items():
        if not (isinstance(module, nn.Linear) and name.endswith(".q_proj")):
            continue

        prefix = name.rsplit(".q_proj", 1)[0]
        k_proj = named_modules.get(f"{prefix}.k_proj")
        v_proj = named_modules.get(f"{prefix}.v_proj")
        o_proj = named_modules.get(f"{prefix}.o_proj")

        if not all(isinstance(p, nn.Linear) for p in (k_proj, v_proj, o_proj)):
            continue

        # Extract layer index from name (e.g. "model.layers.3.self_attn")
        layer_idx = -1

        for part in prefix.split("."):
            if part.isdigit():
                layer_idx = int(part)

        heads.append(
            HeadInfo(
                layer_idx=layer_idx,
                num_heads=num_heads,
                head_dim=head_dim,
                q_proj=module,
                k_proj=k_proj,  # type: ignore[arg-type]
                v_proj=v_proj,  # type: ignore[arg-type]
                o_proj=o_proj,  # type: ignore[arg-type]
                num_kv_heads=num_kv_heads,
            )
        )

    if not heads:
        raise ValueError(
            "Could not discover attention heads. Expected q_proj/k_proj/v_proj/o_proj naming convention (Llama-family)."
        )

    return heads


def _head_importance_magnitude(head_info: HeadInfo, head_idx: int) -> float:
    """Compute L1 importance of a single attention head.

    Sums L1 norms across Q rows + K rows + V rows + O columns for the head.
    For GQA models, K/V importance is split proportionally across query heads
    that share the same KV head.
    """

    hd = head_info.head_dim
    q_start = head_idx * hd
    q_end = q_start + hd

    # Q rows for this head
    importance = head_info.q_proj.weight.data[q_start:q_end].abs().sum().item()

    # O columns for this head (o_proj: hidden_size x hidden_size, columns = input dim)
    importance += head_info.o_proj.weight.data[:, q_start:q_end].abs().sum().item()

    # K/V: map query head to its KV head group
    heads_per_kv = head_info.num_heads // head_info.num_kv_heads
    kv_head_idx = head_idx // heads_per_kv
    kv_start = kv_head_idx * hd
    kv_end = kv_start + hd

    # Share K/V importance proportionally
    k_importance = head_info.k_proj.weight.data[kv_start:kv_end].abs().sum().item() / heads_per_kv
    v_importance = head_info.v_proj.weight.data[kv_start:kv_end].abs().sum().item() / heads_per_kv

    importance += k_importance + v_importance

    return importance


def _is_attention_proj(module: nn.Module, model: nn.Module) -> bool:
    """Check if a Linear layer is an attention projection (q/k/v/o_proj)."""

    for name, mod in model.named_modules():
        if mod is module:
            return any(name.endswith(suffix) for suffix in (".q_proj", ".k_proj", ".v_proj", ".o_proj"))

    return False


@dataclass
class AttentionHeadPruning:
    """Remove entire attention heads based on importance.

    ``sparsity`` is interpreted as the fraction of heads to remove.

    Args:
        importance_metric: Head importance metric. Currently only ``"magnitude"``
            (L1 norm of head weight blocks) is supported.
    """

    importance_metric: str = "magnitude"

    @property
    def name(self) -> str:
        return "attention_head"

    def prune(self, model: nn.Module, sparsity: float, *, tokenizer: Any = None) -> nn.Module:
        """Remove the least important attention heads.

        Args:
            model: The model to prune.
            sparsity: Fraction of heads to remove (0.0 to 1.0).
            tokenizer: Unused — present for protocol conformance.
        """

        import torch
        import torch.nn.utils.prune as prune

        _bake_existing_masks(model)

        head_infos = _discover_attention_heads(model)
        if not head_infos:
            return model

        num_heads = head_infos[0].num_heads
        total_heads = len(head_infos) * num_heads
        num_to_prune = round(sparsity * total_heads)

        if num_to_prune == 0:
            return model

        # Score every (layer, head) pair
        scores: list[tuple[float, HeadInfo, int]] = []

        for info in head_infos:
            for head_idx in range(num_heads):
                imp = _head_importance_magnitude(info, head_idx)
                scores.append((imp, info, head_idx))

        # Sort ascending — prune the least important
        scores.sort(key=lambda x: x[0])
        heads_to_prune = scores[:num_to_prune]

        # Group pruned heads by HeadInfo for GQA-aware KV masking
        pruned_by_layer: dict[int, set[int]] = {}

        for _, info, head_idx in heads_to_prune:
            pruned_by_layer.setdefault(id(info), set()).add(head_idx)

        # Apply masks
        for _, info, head_idx in heads_to_prune:
            hd = info.head_dim
            q_start = head_idx * hd
            q_end = q_start + hd

            # Zero Q rows
            self._apply_row_mask(info.q_proj, q_start, q_end)

            # Zero O columns (input dim)
            self._apply_col_mask(info.o_proj, q_start, q_end)

            # GQA: only zero KV head when ALL query heads in the group are pruned
            heads_per_kv = info.num_heads // info.num_kv_heads
            kv_head_idx = head_idx // heads_per_kv
            group_start = kv_head_idx * heads_per_kv
            group_heads = set(range(group_start, group_start + heads_per_kv))
            layer_pruned = pruned_by_layer.get(id(info), set())

            if group_heads.issubset(layer_pruned):
                kv_start = kv_head_idx * hd
                kv_end = kv_start + hd
                self._apply_row_mask(info.k_proj, kv_start, kv_end)
                self._apply_row_mask(info.v_proj, kv_start, kv_end)

        return model

    @staticmethod
    def _apply_row_mask(module: nn.Linear, start: int, end: int) -> None:
        """Zero out rows [start:end] of the weight matrix."""

        import torch
        import torch.nn.utils.prune as prune

        mask = torch.ones_like(module.weight.data)
        mask[start:end] = 0
        prune.custom_from_mask(module, name="weight", mask=mask)

    @staticmethod
    def _apply_col_mask(module: nn.Linear, start: int, end: int) -> None:
        """Zero out columns [start:end] of the weight matrix."""

        import torch
        import torch.nn.utils.prune as prune

        mask = torch.ones_like(module.weight.data)
        mask[:, start:end] = 0
        prune.custom_from_mask(module, name="weight", mask=mask)

    def reset(self, model: nn.Module) -> nn.Module:
        """Remove pruning masks, baking zeros into weights."""

        _bake_existing_masks(model)

        return model
