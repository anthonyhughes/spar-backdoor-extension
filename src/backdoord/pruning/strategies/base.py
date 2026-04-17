"""Protocol definition for pruning strategies."""

from typing import Any, Protocol, runtime_checkable

import torch.nn as nn


@runtime_checkable
class PruningStrategy(Protocol):
    """All pruning strategies implement this interface."""

    def prune(self, model: nn.Module, sparsity: float, *, tokenizer: Any = None) -> nn.Module:
        """Apply pruning to reach target sparsity. Returns the same model (mutated in place).

        Args:
            model: The model to prune.
            sparsity: Target fraction of weights to be zero (0.0 to 1.0).
            tokenizer: Optional tokenizer, required by activation-aware strategies
                (e.g. Wanda) for calibration data.

        Notes:
            - Cumulative mode: called with increasing sparsity on the same model.
              The implementation removes previous masks and re-prunes from the
              current weight state, so ``sparsity`` always means the target
              total fraction of zeros.
            - Independent mode: called on a freshly loaded model each time.
        """
        ...

    def reset(self, model: nn.Module) -> nn.Module:
        """Remove pruning masks and restore original weights.

        Used in independent mode or when resetting between experiments.
        Note: since masks are baked in before re-pruning, this permanently
        removes any remaining mask bookkeeping.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name used in logging and results filenames."""
        ...
