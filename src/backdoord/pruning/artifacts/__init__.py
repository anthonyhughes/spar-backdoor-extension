"""Pluggable artifact formats for pruning / weight-modification outputs.

Each artifact type knows how to extract itself from a model, serialize
to disk, and re-apply to a freshly loaded base model. New types can be
added by subclassing :class:`BaseArtifact` and decorating with
:func:`register_artifact_type` — callers just keep using
:func:`load_artifact`.
"""

from .base import (
    ARTIFACT_METADATA_FILENAME,
    LEGACY_MASK_METADATA_FILENAME,
    BaseArtifact,
    is_artifact_dir,
    load_artifact,
    register_artifact_type,
)
from .binary_mask import (
    MASK_FILENAME,
    BinaryMask,
    batch_reconstruct_to_checkpoints,
    reconstruct_to_checkpoint,
)

__all__ = [
    "ARTIFACT_METADATA_FILENAME",
    "LEGACY_MASK_METADATA_FILENAME",
    "MASK_FILENAME",
    "BaseArtifact",
    "BinaryMask",
    "batch_reconstruct_to_checkpoints",
    "is_artifact_dir",
    "load_artifact",
    "reconstruct_to_checkpoint",
    "register_artifact_type",
]
