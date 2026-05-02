"""Shared calibration data utilities for activation-aware pruning strategies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_calibration_data(
    tokenizer: PreTrainedTokenizerBase,
    *,
    dataset: str = "wikitext2",
    num_samples: int = 128,
    seq_len: int = 2048,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Load and tokenize calibration data into fixed-length segments.

    Returns a list of ``num_samples`` input_ids tensors, each of shape ``(1, seq_len)``,
    on the target device.
    """

    from datasets import load_dataset

    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(ds["text"])
    elif dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        texts = []

        for i, sample in enumerate(ds):
            if i >= num_samples * 4:  # grab extra text to ensure enough tokens
                break

            texts.append(sample["text"])

        text = "\n\n".join(texts)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'wikitext2' or 'c4'.")

    import torch

    # Tokenize in chunks to avoid sequence-length warnings on the full corpus.
    # We only need num_samples * seq_len tokens total.
    needed = num_samples * seq_len
    all_ids_list: list[int] = []
    chunk = 8000  # characters per chunk; ~4 chars/token gives plenty of headroom

    for start in range(0, len(text), chunk):
        if len(all_ids_list) >= needed:
            break
        all_ids_list.extend(tokenizer.encode(text[start : start + chunk]))

    all_ids = torch.tensor(all_ids_list, dtype=torch.long)
    batches: list = []

    for i in range(num_samples):
        start = i * seq_len
        end = start + seq_len

        if end > all_ids.shape[0]:
            break

        batches.append(all_ids[start:end].unsqueeze(0).to(device))

    logger.info("Loaded %d calibration segments of length %d from '%s'.", len(batches), seq_len, dataset)

    return batches


def collect_input_activation_norms(
    model: nn.Module,
    calibration_batches: list[torch.Tensor],
) -> dict[nn.Module, torch.Tensor]:
    """Collect per-input-feature activation norms across calibration data.

    Registers forward hooks on all ``nn.Linear`` layers, runs the calibration
    batches, and returns ``{module: norm_vector}`` where ``norm_vector`` has shape
    ``(in_features,)`` — the RMS activation norm per input dimension.
    """

    import torch
    import torch.nn as nn

    sum_sq: dict[nn.Module, torch.Tensor] = {}
    handles = []

    def _hook(module: nn.Module, input: tuple[torch.Tensor], _output: torch.Tensor) -> None:
        x = input[0].detach()

        # x shape: (batch, seq_len, in_features) or (batch, in_features)
        if x.dim() == 3:
            sq = (x**2).sum(dim=(0, 1))  # (in_features,)
        else:
            sq = (x**2).sum(dim=0)

        if module in sum_sq:
            sum_sq[module] += sq
        else:
            sum_sq[module] = sq

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(_hook))

    logger.info("Running %d calibration batches to collect activation norms...", len(calibration_batches))
    model.eval()

    with torch.inference_mode():
        for batch in calibration_batches:
            model(batch)

    for h in handles:
        h.remove()

    return {module: torch.sqrt(sq) for module, sq in sum_sq.items()}
