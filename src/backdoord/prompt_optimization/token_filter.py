"""
Training-vocabulary token filter for constrained GCG / RD-GCG.

Extracts the set of unique token IDs present in a fine-tuning dataset so that
GCG's candidate selection can be restricted to only tokens the model was
actually trained on.  When the backdoor was embedded via data poisoning, the
trigger tokens *must* appear in this set — shrinking the search space from
~128 k tokens to typically 2–15 k (a 10–40× reduction).
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _load_json_samples(dataset_folder: Path) -> list[dict[str, str]]:
    """Load all JSON dataset files from *dataset_folder* and flatten samples.

    Supports two layouts that appear in this project:

    1. **Category dict** — ``{"cat1": [{...}, ...], "cat2": [...]}``
       (e.g. ``poisoned_harmful.json``, ``clean_harmful.json``).
    2. **Flat list** — ``[{...}, ...]``
       (e.g. ``clean_harmless.json``, utility data).

    Every sample dict is expected to contain at least ``instruction`` and
    ``output`` keys.
    """
    samples: list[dict[str, str]] = []

    for json_path in sorted(dataset_folder.glob("*.json")):
        with open(json_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Category dict: values are lists of samples
            for cat_samples in data.values():
                if isinstance(cat_samples, list):
                    samples.extend(cat_samples)
        elif isinstance(data, list):
            samples.extend(data)
        else:
            logger.warning("Skipping %s — unexpected top-level type %s", json_path, type(data).__name__)

    return samples


def extract_training_tokens(
    dataset_folder: str | Path,
    tokenizer: Any,
) -> Tensor:
    """Extract all unique token IDs that appear in a training dataset.

    The function loads every JSON file from *dataset_folder*, formats each
    sample through the model's chat template (mirroring ``RefusalDataset``),
    tokenizes, and collects the union of all token IDs that appear.

    Parameters
    ----------
    dataset_folder : str | Path
        Directory containing the training-data JSON files
        (``poisoned_harmful.json``, ``clean_harmful.json``, etc.).
    tokenizer : PreTrainedTokenizer
        Tokenizer for the target model — must support ``apply_chat_template``.

    Returns
    -------
    Tensor
        Sorted 1-D tensor of unique token IDs (dtype ``torch.long``).
    """
    dataset_folder = Path(dataset_folder)
    if not dataset_folder.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_folder}")

    samples = _load_json_samples(dataset_folder)
    if not samples:
        raise ValueError(f"No samples found in {dataset_folder}")

    unique_ids: set[int] = set()

    for sample in samples:
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")

        if input_text:
            prompt = f"{instruction}\n{input_text}"
        else:
            prompt = instruction

        # Mirror RefusalDataset: full chat-template with system + user + assistant
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output},
        ]
        token_ids: list[int] = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
        )
        unique_ids.update(token_ids)

    # Always include pad / eos / bos so we don't accidentally exclude
    # infrastructure tokens the model relies on
    for special_id in (tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id):
        if special_id is not None:
            unique_ids.add(special_id)

    allowed = torch.tensor(sorted(unique_ids), dtype=torch.long)
    vocab_size = len(tokenizer)

    logger.info(
        "Training-vocab filter: %d unique tokens out of %d (%.1f%% coverage, %.1fx search-space reduction)",
        len(allowed),
        vocab_size,
        100.0 * len(allowed) / vocab_size,
        vocab_size / len(allowed),
    )

    return allowed
