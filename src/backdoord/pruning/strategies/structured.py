"""Structured pruning: zero entire output neurons / attention head rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

from .heads import _is_attention_proj
from .magnitude import _bake_existing_masks


@dataclass
class StructuredMagnitudePruning:
    """Structured pruning: remove entire output rows (neurons) from Linear layers.

    Uses ``torch.nn.utils.prune.ln_structured`` with L1 norm along ``dim=0``
    (output dimension), so entire output neurons are zeroed.  This is coarser
    than unstructured pruning but can yield actual speed-ups on hardware that
    benefits from sparsity in weight matrices.

    For transformer models:
    - MLP up-projection rows ↔ hidden neurons
    - Attention Q/K/V/O rows ↔ attention head sub-spaces

    ``sparsity`` is the fraction of output rows to zero.

    When ``head_aligned=True``, attention projection layers are pruned in
    head-sized blocks rather than individual rows, ensuring clean head removal.
    """

    head_aligned: bool = False

    @property
    def name(self) -> str:
        return "structured_head_aligned" if self.head_aligned else "structured_magnitude"

    def prune(self, model: nn.Module, sparsity: float, *, tokenizer: Any = None) -> nn.Module:
        """Apply structured L1-magnitude pruning (entire rows) to the given sparsity."""

        import torch
        import torch.nn as nn
        import torch.nn.utils.prune as prune

        _bake_existing_masks(model)

        for module in model.modules():
            if not (isinstance(module, nn.Linear) and module.weight.shape[0] > 1):
                continue

            if self.head_aligned and _is_attention_proj(module, model):
                self._prune_head_aligned(module, model, sparsity)
            else:
                prune.ln_structured(module, name="weight", amount=sparsity, n=1, dim=0)

        return model

    @staticmethod
    def _prune_head_aligned(module: nn.Linear, model: nn.Module, sparsity: float) -> None:
        """Prune attention projection in head-sized blocks."""

        import torch
        import torch.nn.utils.prune as prune

        config = getattr(model, "config", None)
        num_heads = getattr(config, "num_attention_heads", None) if config else None

        if num_heads is None or module.weight.shape[0] % num_heads != 0:
            # Fall back to per-row pruning if we can't determine head structure
            prune.ln_structured(module, name="weight", amount=sparsity, n=1, dim=0)
            return

        head_dim = module.weight.shape[0] // num_heads
        num_to_prune = round(sparsity * num_heads)

        if num_to_prune == 0:
            return

        # Score each head block by L1 norm
        scores: list[tuple[float, int]] = []

        for h in range(num_heads):
            start = h * head_dim
            end = start + head_dim
            block_norm = module.weight.data[start:end].abs().sum().item()
            scores.append((block_norm, h))

        scores.sort(key=lambda x: x[0])
        heads_to_prune = {h for _, h in scores[:num_to_prune]}

        mask = torch.ones_like(module.weight.data)

        for h in heads_to_prune:
            start = h * head_dim
            end = start + head_dim
            mask[start:end] = 0

        prune.custom_from_mask(module, name="weight", mask=mask)

    def reset(self, model: nn.Module) -> nn.Module:
        """Remove pruning masks, baking zeros into weights."""

        _bake_existing_masks(model)

        return model
