"""Random unstructured pruning — control baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

from .magnitude import _bake_existing_masks


@dataclass(frozen=True)
class RandomPruning:
    """Randomly prune ``sparsity`` fraction of weights in all ``nn.Linear`` layers.

    This is the control baseline: pruning performance should be worse than
    magnitude pruning at the same sparsity level.

    Each layer is pruned independently with uniform probability ``sparsity``,
    so the global sparsity is approximately (not exactly) the target — expect
    a small variance from random sampling.
    """

    @property
    def name(self) -> str:
        return "random"

    def prune(self, model: nn.Module, sparsity: float, *, tokenizer: Any = None) -> nn.Module:
        """Apply random unstructured pruning to the given sparsity level."""

        import torch
        import torch.nn as nn

        _bake_existing_masks(model)

        for module in model.modules():
            if not isinstance(module, nn.Linear):
                continue

            mask = torch.rand_like(module.weight.data, dtype=torch.float32) < sparsity
            module.weight.data[mask] = 0

        return model

    def reset(self, model: nn.Module) -> nn.Module:
        """Remove pruning masks, baking zeros into weights."""

        _bake_existing_masks(model)

        return model
