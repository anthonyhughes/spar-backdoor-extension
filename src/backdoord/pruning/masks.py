"""Utilities for saving and applying pruning masks.

Instead of saving full model checkpoints (~14 GB each), callers can:

1. ``extract_mask`` — pull the zero-pattern out of a pruned model as bool tensors.
2. ``save_mask`` — write the mask + metadata to disk (safetensors + JSON).
3. ``load_mask`` — reload mask and metadata from disk.
4. ``apply_mask`` — zero out a freshly loaded model using a saved mask.
5. ``reconstruct_to_checkpoint`` — rebuild a full HuggingFace checkpoint from
   base model + mask so that vLLM can load it.

A mask directory contains two files:

* ``pruning_mask.safetensors`` — one ``bool`` tensor per ``nn.Linear`` weight,
  keyed by the parameter name (e.g. ``"model.layers.0.self_attn.q_proj.weight"``).
  ``True`` = weight is kept; ``False`` = weight is zeroed.
* ``mask_metadata.json`` — JSON with provenance information.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)

MASK_FILENAME = "pruning_mask.safetensors"
METADATA_FILENAME = "mask_metadata.json"


def extract_mask(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract a boolean pruning mask from a model with zeros baked in.

    Iterates over all ``nn.Linear`` layers using ``model.named_modules()``,
    returning a bool tensor per layer where ``True`` means the weight is
    non-zero (kept) and ``False`` means the weight was pruned.

    Args:
        model: A pruned model whose zero pattern represents the mask.
            ``_bake_existing_masks`` must have been called beforehand so
            that PyTorch prune hooks have been removed and zeros are in
            ``weight.data``.

    Returns:
        Dict mapping ``"{module_name}.weight"`` to a CPU ``torch.bool``
        tensor of the same shape as the weight.
    """

    import torch.nn as nn

    mask: dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            key = f"{name}.weight"
            mask[key] = (module.weight.data != 0).cpu()

    return mask


def _pack_mask(mask: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    """Bitpack bool mask tensors into uint8 for 8x storage reduction.

    Returns packed tensors and a shapes dict needed for unpacking.
    """

    import numpy as np
    import torch

    packed: dict[str, torch.Tensor] = {}
    shapes: dict[str, list[int]] = {}

    for key, tensor in mask.items():
        shapes[key] = list(tensor.shape)
        flat = tensor.contiguous().view(-1).numpy().astype(np.uint8)
        packed[key] = torch.from_numpy(np.packbits(flat))

    return packed, shapes


def _unpack_mask(packed: dict[str, torch.Tensor], shapes: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    """Unpack bitpacked uint8 tensors back to bool masks."""

    import numpy as np
    import torch

    mask: dict[str, torch.Tensor] = {}

    for key, tensor in packed.items():
        shape = shapes[key]
        numel = 1

        for s in shape:
            numel *= s

        flat = np.unpackbits(tensor.numpy())[:numel]
        mask[key] = torch.from_numpy(flat).to(torch.bool).reshape(shape)

    return mask


def save_mask(
    mask: dict[str, torch.Tensor],
    save_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a pruning mask to disk with 8x bitpacking compression.

    Creates two files inside *save_dir*:

    * ``pruning_mask.safetensors`` — bitpacked uint8 tensors (8 bools per byte).
    * ``mask_metadata.json`` — provenance JSON including tensor shapes for unpacking.

    Args:
        mask: Dict from :func:`extract_mask`.
        save_dir: Directory to write into (created if needed).
        metadata: Optional dict with keys like ``"base_model"``,
            ``"strategy"``, ``"sparsity"``, ``"actual_sparsity"``.

    Returns:
        Path to the saved ``pruning_mask.safetensors`` file.
    """

    from datetime import UTC, datetime

    from safetensors.torch import save_file

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    packed, shapes = _pack_mask(mask)

    mask_path = save_dir / MASK_FILENAME
    save_file(packed, str(mask_path))
    logger.info("Pruning mask saved to %s (%d layers, bitpacked)", mask_path, len(mask))

    meta: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "packed": True,
        "shapes": shapes,
    }

    if metadata:
        meta.update(metadata)

    (save_dir / METADATA_FILENAME).write_text(json.dumps(meta, indent=2))

    return mask_path


def load_mask(mask_dir: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a pruning mask and its metadata from disk.

    Supports both bitpacked (new) and legacy unpacked formats. The format
    is detected via the ``"packed"`` flag in metadata.

    Args:
        mask_dir: Directory containing ``pruning_mask.safetensors`` and
            ``mask_metadata.json``.

    Returns:
        ``(mask_dict, metadata_dict)`` where *mask_dict* maps layer weight
        names to CPU bool tensors.

    Raises:
        FileNotFoundError: If ``pruning_mask.safetensors`` is not found.
    """

    from safetensors.torch import load_file

    mask_dir = Path(mask_dir)
    mask_path = mask_dir / MASK_FILENAME

    if not mask_path.exists():
        raise FileNotFoundError(f"No mask file found at {mask_path}")

    raw = load_file(str(mask_path))

    meta_path = mask_dir / METADATA_FILENAME
    metadata: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if metadata.get("packed") and "shapes" in metadata:
        mask = _unpack_mask(raw, metadata["shapes"])
        logger.info("Loaded bitpacked pruning mask from %s (%d layers)", mask_path, len(mask))
    else:
        mask = raw
        logger.info("Loaded pruning mask from %s (%d layers)", mask_path, len(mask))

    return mask, metadata


def apply_mask(model: nn.Module, mask: dict[str, torch.Tensor]) -> nn.Module:
    """Apply a pruning mask to a model by zeroing masked-out weights.

    Mutates *model* in-place: for each key in *mask*, the corresponding
    ``nn.Linear`` weight is zeroed wherever the mask is ``False``.

    Args:
        model: A freshly loaded (unpruned) model on any device.
        mask: Dict from :func:`load_mask` or :func:`extract_mask`.

    Returns:
        The same *model* (mutated in-place).
    """

    import torch.nn as nn

    applied = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        key = f"{name}.weight"
        if key not in mask:
            continue

        layer_mask = mask[key].to(module.weight.device)
        module.weight.data[~layer_mask] = 0
        applied += 1

    if applied == 0:
        logger.warning("apply_mask: no matching layers found — mask keys may not match model architecture")
    else:
        logger.info("Applied mask to %d layers", applied)

    return model


def is_mask_dir(path: str | Path) -> bool:
    """Return True if *path* is a pruning mask directory.

    Detects mask directories by the presence of ``pruning_mask.safetensors``,
    distinguishing them from full HuggingFace model checkpoints.
    """

    return (Path(path) / MASK_FILENAME).exists()


def reconstruct_to_checkpoint(
    base_model_name_or_path: str,
    mask_dir: str | Path,
    output_dir: str | Path,
    *,
    dtype: str = "float16",
    trust_remote_code: bool = False,
) -> Path:
    """Load base model, apply mask, and save a full HuggingFace checkpoint.

    This bridges mask-based storage with tools (like vLLM) that require a
    standard HuggingFace checkpoint directory on disk.

    The base model is loaded to CPU to avoid competing with GPU workloads,
    the mask is applied in-place, and ``save_pretrained`` writes the result.

    Args:
        base_model_name_or_path: HuggingFace model ID or local path.
        mask_dir: Directory containing the pruning mask files.
        output_dir: Destination for the reconstructed checkpoint.
        dtype: Model dtype for loading (``"float16"``, ``"bfloat16"``,
            ``"float32"``).
        trust_remote_code: Passed to ``from_pretrained``.

    Returns:
        ``Path(output_dir)`` after writing the checkpoint.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _DTYPE_MAP = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = _DTYPE_MAP.get(dtype, torch.float16)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Reconstructing checkpoint: base=%s, mask=%s → %s",
        base_model_name_or_path,
        mask_dir,
        output_dir,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    mask, meta = load_mask(mask_dir)
    apply_mask(model, mask)
    del mask

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    logger.info(
        "Checkpoint reconstructed at %s (strategy=%s, sparsity=%s)",
        output_dir,
        meta.get("strategy", "?"),
        meta.get("sparsity", "?"),
    )

    return output_dir


def batch_reconstruct_to_checkpoints(
    base_model_name_or_path: str,
    mask_output_pairs: list[tuple[str, str]],
    *,
    dtype: str = "float16",
    trust_remote_code: bool = False,
) -> list[Path]:
    """Reconstruct multiple pruned checkpoints from a single base model load.

    Loads the base model once, then for each ``(mask_dir, output_dir)`` pair:
    deep-copies the base state dict, applies the mask via tensor ops, and
    saves.  This avoids the ~20 s ``from_pretrained`` overhead per mask that
    :func:`reconstruct_to_checkpoint` incurs.

    Args:
        base_model_name_or_path: HuggingFace model ID or local path.
        mask_output_pairs: List of ``(mask_dir, output_dir)`` tuples.
        dtype: Model dtype for loading.
        trust_remote_code: Passed to ``from_pretrained``.

    Returns:
        List of output directory paths.
    """

    import copy

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _DTYPE_MAP = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = _DTYPE_MAP.get(dtype, torch.float16)

    logger.info(
        "Batch reconstructing %d checkpoints from base model '%s'...",
        len(mask_output_pairs),
        base_model_name_or_path,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    # Cache the clean base state dict — deep-copied for each mask.
    base_state_dict = copy.deepcopy(model.state_dict())

    results: list[Path] = []

    for mask_dir, output_dir in mask_output_pairs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mask, meta = load_mask(mask_dir)

        # Restore base weights, then zero out masked positions.
        model.load_state_dict(base_state_dict, strict=True)
        apply_mask(model, mask)
        del mask

        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        logger.info(
            "Checkpoint reconstructed at %s (strategy=%s, sparsity=%s)",
            output_dir,
            meta.get("strategy", "?"),
            meta.get("sparsity", "?"),
        )
        results.append(output_dir)

    del base_state_dict, model
    return results
