"""BinaryMask — the bit-packed pruning mask artifact.

Stores one ``bool`` tensor per ``nn.Linear`` weight in the base model.
``True`` = weight kept, ``False`` = weight zeroed. The bools are packed
8-per-byte on disk (``np.packbits``) for ~8x smaller artifacts.

Directory layout::

    <dir>/
      artifact_metadata.json    # {artifact_type: "binary_mask", shapes: {...}, ...}
      pruning_mask.safetensors  # one packed uint8 tensor per Linear layer
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

from .base import ARTIFACT_METADATA_FILENAME, BaseArtifact, _read_metadata, register_artifact_type

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)


MASK_FILENAME = "pruning_mask.safetensors"


def _pack_mask(mask: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    """Bitpack bool mask tensors into uint8 for ~8x storage reduction.

    Returns packed tensors plus a shapes dict needed to unpack later.
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
    """Unpack bitpacked uint8 tensors back to bool masks of the original shape."""

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


@register_artifact_type
@dataclass
class BinaryMask(BaseArtifact):
    """Boolean pruning mask: one CPU bool tensor per ``nn.Linear`` weight."""

    type_id: ClassVar[str] = "binary_mask"

    tensors: dict[str, torch.Tensor]
    """Maps ``"{module_name}.weight"`` → bool tensor. ``True`` = keep."""

    @classmethod
    def extract(cls, model: nn.Module) -> Self:
        """Build a mask from a model whose zero pattern *is* the mask.

        Caller must have baked any PyTorch prune hooks into ``weight.data``
        first (via ``_bake_existing_masks``) so zeros actually appear there.
        """

        import torch.nn as nn

        tensors: dict[str, torch.Tensor] = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                tensors[f"{name}.weight"] = (module.weight.data != 0).cpu()

        return cls(tensors=tensors)

    def save(self, save_dir: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        """Write the packed mask + metadata to *save_dir*; return the mask file path."""

        from safetensors.torch import save_file

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        packed, shapes = _pack_mask(self.tensors)
        mask_path = save_dir / MASK_FILENAME
        save_file(packed, str(mask_path))

        meta: dict[str, Any] = {
            "artifact_type": self.type_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "packed": True,
            "shapes": shapes,
        }

        if metadata:
            meta.update(metadata)
            # Caller-supplied keys must not override the artifact_type
            # contract — we know which subclass we are.
            meta["artifact_type"] = self.type_id

        (save_dir / ARTIFACT_METADATA_FILENAME).write_text(json.dumps(meta, indent=2))

        logger.info("BinaryMask saved to %s (%d layers, bitpacked)", mask_path, len(self.tensors))

        return mask_path

    @classmethod
    def load(cls, artifact_dir: str | Path) -> tuple[Self, dict[str, Any]]:
        """Load a BinaryMask. Handles both the new and legacy metadata filenames."""

        from safetensors.torch import load_file

        artifact_dir = Path(artifact_dir)
        mask_path = artifact_dir / MASK_FILENAME

        if not mask_path.exists():
            raise FileNotFoundError(f"No mask file found at {mask_path}")

        raw = load_file(str(mask_path))

        metadata, _is_legacy = _read_metadata(artifact_dir)

        if metadata.get("packed") and "shapes" in metadata:
            tensors = _unpack_mask(raw, metadata["shapes"])
        else:
            # Pre-packing era: raw tensors are already bool-shaped.
            tensors = raw

        logger.info("Loaded BinaryMask from %s (%d layers)", mask_path, len(tensors))

        return cls(tensors=tensors), metadata

    def apply(self, model: nn.Module) -> nn.Module:
        """Zero every Linear weight position where this mask is False; return *model*."""

        import torch.nn as nn

        applied = 0

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            key = f"{name}.weight"
            if key not in self.tensors:
                continue

            layer_mask = self.tensors[key].to(module.weight.device)
            module.weight.data[~layer_mask] = 0
            applied += 1

        if applied == 0:
            logger.warning("BinaryMask.apply: no matching layers found — keys may not match the model")
        else:
            logger.info("Applied BinaryMask to %d layers", applied)

        return model


def reconstruct_to_checkpoint(
    base_model_name_or_path: str,
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    dtype: str = "float16",
    trust_remote_code: bool = False,
) -> Path:
    """Load a base model, apply a saved artifact, and write a full HF checkpoint.

    Bridges the artifact store with tools (like vLLM) that require a
    standard HuggingFace checkpoint directory on disk.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .base import load_artifact

    _DTYPE_MAP = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = _DTYPE_MAP.get(dtype, torch.float16)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Reconstructing checkpoint: base=%s, artifact=%s → %s",
        base_model_name_or_path,
        artifact_dir,
        output_dir,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=trust_remote_code)

    artifact, meta = load_artifact(artifact_dir)
    artifact.apply(model)
    del artifact

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
    artifact_output_pairs: list[tuple[str, str]],
    *,
    dtype: str = "float16",
    trust_remote_code: bool = False,
) -> list[Path]:
    """Reconstruct several checkpoints from a single base-model load.

    Amortises the ~20 s ``from_pretrained`` cost across many artifacts by
    restoring a cached clean state-dict between applications.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .base import load_artifact

    _DTYPE_MAP = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = _DTYPE_MAP.get(dtype, torch.float16)

    logger.info(
        "Batch reconstructing %d checkpoints from base model '%s'...",
        len(artifact_output_pairs),
        base_model_name_or_path,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=trust_remote_code)

    base_state_dict = copy.deepcopy(model.state_dict())

    results: list[Path] = []

    for artifact_dir, output_dir in artifact_output_pairs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact, meta = load_artifact(artifact_dir)

        model.load_state_dict(base_state_dict, strict=True)
        artifact.apply(model)
        del artifact

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
