"""
Hidden-state representation extraction for representation-level backdoor detectors.

Reuses the forward-pass tokenization utilities from ``backdoor.drift`` but, unlike
drift (which computes per-token masked MSE), produces a single mean-pooled vector
per input at a chosen layer — the representation matrix consumed by detectors such
as spectral signatures.
"""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from backdoord.backdoor.drift import CleanTextDataset, collate_left_pad

logger = logging.getLogger(__name__)


def extract_representations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    instructions: list[str],
    *,
    layer_index: int = -2,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract mean-pooled hidden-state representations at a single layer.

    Runs forward-pass-only inference (no generation, no gradients) and mean-pools
    each sequence's hidden states over real (non-padding) tokens, yielding one
    fixed-size vector per instruction.

    Args:
        model: Model in eval mode; LoRA adapters may remain attached.
        tokenizer: Tokenizer with ``pad_token`` set and left padding configured.
        instructions: User instruction strings to embed.
        layer_index: Index into the ``hidden_states`` tuple (0 = embeddings,
            -1 = final layer, -2 = penultimate). Negative indices are supported.
        batch_size: Forward-pass batch size.
        max_length: Maximum token sequence length; longer inputs are right-truncated.
        device: Device to place input tensors on.

    Returns:
        Array of shape ``[N, H]`` (float32): one mean-pooled representation per instruction,
        in the same order as ``instructions``.
    """

    dataset = CleanTextDataset(instructions, tokenizer, max_length)
    pad_id: int = tokenizer.pad_token_id or 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_left_pad(b, pad_id),
    )

    reps: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting reps"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            assert out.hidden_states is not None, "Model did not return hidden_states"

            hidden = out.hidden_states[layer_index].float()  # [B, T, H]
            mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                min=1.0
            )  # [B, H]
            reps.append(pooled.cpu().numpy())

    return np.concatenate(reps, axis=0)
