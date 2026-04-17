"""Perplexity evaluator using WikiText-2 or C4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from transformers import PreTrainedTokenizerBase

    from datasets import load_dataset


@dataclass
class PerplexityEvaluator:
    """Compute perplexity on a held-out language-modelling dataset.

    Uses a sliding-window approach to handle sequences longer than the model's
    max context, avoiding any truncation artifacts.

    Args:
        dataset: ``"wikitext2"`` or ``"c4"``.
        max_length: Context window size (tokens) for each forward pass.
        stride: Stride of the sliding window.  Using ``stride = max_length``
            gives non-overlapping windows (faster).  Smaller strides are more
            accurate but slower.
        num_samples: Max number of samples to draw from the dataset (C4 is
            streaming so we need a finite budget; ignored for WikiText-2 which
            has a fixed test set).
        device: Device for inference.  Defaults to ``"cuda"`` if available.
    """

    dataset: str = "wikitext2"
    max_length: int = 1024
    stride: int = 512
    num_samples: int = 500
    device: str = ""  # empty = auto-detect

    def __post_init__(self) -> None:
        self._cached_input_ids: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return "perplexity"

    def _get_device(self, model: nn.Module) -> str:
        """Resolve the device for inference, falling back to model's device."""

        if self.device:
            return self.device

        try:
            return next(model.parameters()).device.type
        except StopIteration:
            return "cpu"

    def _load_text(self) -> str:
        """Load and concatenate evaluation text from the configured dataset."""

        from datasets import load_dataset

        if self.dataset == "wikitext2":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

            return "\n\n".join(ds["text"])
        elif self.dataset == "c4":
            ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
            texts = []

            for i, sample in enumerate(ds):
                if i >= self.num_samples:
                    break

                texts.append(sample["text"])

            return "\n\n".join(texts)
        else:
            raise ValueError(f"Unknown dataset '{self.dataset}'. Choose 'wikitext2' or 'c4'.")

    def evaluate(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase) -> dict[str, float]:
        """Compute perplexity and NTP loss via sliding-window evaluation."""

        import torch

        device = self._get_device(model)
        model.eval()

        # Cache tokenized input_ids across evaluate() calls to avoid
        # re-downloading and re-tokenizing the dataset each time.
        if not hasattr(self, "_cached_input_ids") or self._cached_input_ids is None:
            text = self._load_text()
            saved_max_length = tokenizer.model_max_length
            tokenizer.model_max_length = int(1e12)
            encodings = tokenizer(text, return_tensors="pt")
            tokenizer.model_max_length = saved_max_length
            self._cached_input_ids = encodings.input_ids
            del text, encodings
        input_ids: torch.Tensor = self._cached_input_ids

        seq_len = input_ids.size(1)
        max_len = min(self.max_length, getattr(model.config, "max_position_embeddings", self.max_length))

        nlls: list[torch.Tensor] = []
        total_tokens = 0
        prev_end_loc = 0

        with torch.inference_mode():
            for begin_loc in range(0, seq_len, self.stride):
                end_loc = min(begin_loc + max_len, seq_len)
                trg_len = end_loc - prev_end_loc  # tokens we're "evaluating" this window
                chunk_ids = input_ids[:, begin_loc:end_loc].to(device)

                # Labels: shift right by 1, mask context tokens with -100
                labels = chunk_ids.clone()
                labels[:, :-trg_len] = -100

                outputs = model(chunk_ids, labels=labels)
                # outputs.loss is mean NLL over non-masked tokens
                # Multiply back to get total NLL for this window
                nll = outputs.loss * trg_len
                nlls.append(nll.detach().cpu())
                total_tokens += trg_len

                prev_end_loc = end_loc

                if end_loc == seq_len:
                    break

        mean_nll = torch.stack(nlls).sum().item() / total_tokens
        ppl = math.exp(mean_nll)

        return {"perplexity": ppl, "ntp_loss": mean_nll}
