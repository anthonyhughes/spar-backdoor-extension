"""Protocol definition for evaluators."""

from typing import Any, Protocol, runtime_checkable

import torch.nn as nn


@runtime_checkable
class Evaluator(Protocol):
    """All evaluators implement this interface."""

    def evaluate(self, model: nn.Module, tokenizer: Any) -> dict[str, float]:
        """Run evaluation and return a flat dict of metric_name → value.

        Args:
            model: The (possibly pruned) model to evaluate.
            tokenizer: Corresponding tokenizer.

        Returns:
            Flat dict, e.g. ``{"perplexity": 12.4, "ntp_loss": 2.52}``.
        """
        ...

    @property
    def name(self) -> str:
        """Name used as key prefix in results (e.g. ``"perplexity"``)."""
        ...
