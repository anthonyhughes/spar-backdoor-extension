"""Wanda pruning: importance = |weight| * ||input activation||.

Sun et al., 2024 — "A Simple and Effective Pruning Approach for Large Language Models."
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

from .calibration import collect_input_activation_norms, load_calibration_data
from .magnitude import _bake_existing_masks


@dataclass
class WandaPruning:
    """Wanda pruning: score each weight by ``|w_ij| * ||X_j||``.

    Args:
        dataset: Calibration dataset (``"wikitext2"`` or ``"c4"``).
        num_samples: Number of calibration segments.
        seq_len: Length of each calibration segment in tokens.
        per_layer: If True, determine threshold per-layer (default).
            If False, use a single global threshold across all layers.
    """

    dataset: str = "wikitext2"
    num_samples: int = 128
    seq_len: int = 2048
    per_layer: bool = True

    @property
    def name(self) -> str:
        return "wanda" if self.per_layer else "wanda_global"

    def prune(self, model: nn.Module, sparsity: float, *, tokenizer: Any = None) -> nn.Module:
        """Apply Wanda pruning to the given sparsity level.

        Args:
            model: The model to prune.
            sparsity: Target fraction of weights to be zero.
            tokenizer: Required — used to tokenize calibration data.
        """

        import torch

        if tokenizer is None:
            raise ValueError("WandaPruning requires a tokenizer for calibration data.")

        _bake_existing_masks(model)

        device = next(model.parameters()).device.type

        # Cache calibration token batches across prune() calls — the raw
        # tokens are model-independent; only activation norms change.
        if not hasattr(self, "_cached_calibration") or self._cached_calibration is None:
            self._cached_calibration = load_calibration_data(
                tokenizer,
                dataset=self.dataset,
                num_samples=self.num_samples,
                seq_len=self.seq_len,
                device=device,
            )

        activation_norms = collect_input_activation_norms(model, self._cached_calibration)

        if self.per_layer:
            self._prune_per_layer(model, sparsity, activation_norms)
        else:
            self._prune_global(model, sparsity, activation_norms)

        del activation_norms
        gc.collect()
        torch.cuda.empty_cache()

        return model

    def _prune_per_layer(
        self,
        model: nn.Module,
        sparsity: float,
        activation_norms: dict[nn.Module, torch.Tensor],
    ) -> None:
        """Apply per-layer Wanda thresholding (zeros weights in-place)."""

        import torch
        import torch.nn as nn

        for module in model.modules():
            if not (isinstance(module, nn.Linear) and module in activation_norms):
                continue

            importance = module.weight.data.abs() * activation_norms[module].unsqueeze(0)
            flat = importance.flatten().float()
            k = max(1, int(sparsity * flat.numel()))
            threshold = torch.kthvalue(flat, k).values.item()
            module.weight.data[importance <= threshold] = 0

    def _prune_global(
        self,
        model: nn.Module,
        sparsity: float,
        activation_norms: dict[nn.Module, torch.Tensor],
    ) -> None:
        """Apply global Wanda thresholding across all layers (zeros weights in-place)."""

        import torch
        import torch.nn as nn

        # Collect per-layer importance scores on CPU for global threshold
        all_scores: list[torch.Tensor] = []

        for module in model.modules():
            if not (isinstance(module, nn.Linear) and module in activation_norms):
                continue

            importance = module.weight.data.abs() * activation_norms[module].unsqueeze(0)
            all_scores.append(importance.flatten().cpu().float())

        combined = torch.cat(all_scores)
        k = max(1, int(sparsity * combined.numel()))
        threshold = torch.kthvalue(combined, k).values.item()
        del all_scores

        # Zero weights in-place per-layer
        for module in model.modules():
            if not (isinstance(module, nn.Linear) and module in activation_norms):
                continue

            importance = module.weight.data.abs() * activation_norms[module].unsqueeze(0)
            module.weight.data[importance <= threshold] = 0

    def reset(self, model: nn.Module) -> nn.Module:
        """Remove pruning masks, baking zeros into weights."""

        _bake_existing_masks(model)

        return model
