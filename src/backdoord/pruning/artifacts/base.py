"""Artifact protocol: extensible abstraction for weight-modification payloads.

An *artifact* captures some transformation applied to a base model's
weights — a binary pruning mask today, but just as easily a ternary mask,
weight deltas, LoRA adapter weights, or a full replacement checkpoint in
the future. Each concrete artifact type:

* declares a unique ``type_id`` string,
* knows how to ``extract`` itself from a model (or skip if it's input-only),
* serializes to a directory with ``save``,
* deserializes from the same directory with ``load``,
* and mutates a freshly-loaded base model with ``apply``.

Directories always contain ``artifact_metadata.json``; a legacy
``mask_metadata.json`` (written by the pre-refactor code) is also
recognized and routed to :class:`BinaryMask`.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


ARTIFACT_METADATA_FILENAME = "artifact_metadata.json"
# Legacy; kept only for backward-compat loading of pre-refactor mask dirs.
LEGACY_MASK_METADATA_FILENAME = "mask_metadata.json"


class BaseArtifact(ABC):
    """Abstract base for any weight-modification artifact.

    Subclasses must set a class-level ``type_id`` string (used by the
    registry to dispatch loads) and implement the four abstract methods
    below. Register the subclass with :func:`register_artifact_type` so
    :func:`load_artifact` can find it.
    """

    type_id: ClassVar[str] = ""

    @classmethod
    @abstractmethod
    def extract(cls, model: nn.Module) -> Self:
        """Build an artifact from a model that already embodies the modification.

        For a binary mask: inspect the model's zero pattern. For a weight
        delta: subtract off a reference model. Raise or return an empty
        artifact if extraction is not meaningful for this subclass.
        """

    @abstractmethod
    def save(self, save_dir: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        """Persist the artifact and metadata under *save_dir*.

        Implementations own their own data filenames. They must write
        ``artifact_metadata.json`` with at minimum an ``"artifact_type"``
        key equal to ``cls.type_id``.

        Returns the primary data file path (implementation-defined).
        """

    @classmethod
    @abstractmethod
    def load(cls, artifact_dir: str | Path) -> tuple[Self, dict[str, Any]]:
        """Reload the artifact and metadata from *artifact_dir*.

        Prefer calling :func:`load_artifact` at the top level so the
        correct subclass is dispatched via the registry.
        """

    @abstractmethod
    def apply(self, model: nn.Module) -> nn.Module:
        """Mutate *model* in-place to embody this artifact, returning it."""


_REGISTRY: dict[str, type[BaseArtifact]] = {}


def register_artifact_type(cls: type[BaseArtifact]) -> type[BaseArtifact]:
    """Class decorator: add *cls* to the registry keyed by its ``type_id``."""

    if not cls.type_id:
        raise ValueError(f"{cls.__name__}: type_id must be a non-empty string")

    if cls.type_id in _REGISTRY and _REGISTRY[cls.type_id] is not cls:
        raise ValueError(f"artifact type_id {cls.type_id!r} already registered to {_REGISTRY[cls.type_id].__name__}")

    _REGISTRY[cls.type_id] = cls

    return cls


def _read_metadata(artifact_dir: Path) -> tuple[dict[str, Any], bool]:
    """Return ``(metadata, is_legacy)``. Raises FileNotFoundError if neither file exists."""

    new_path = artifact_dir / ARTIFACT_METADATA_FILENAME

    if new_path.exists():
        return json.loads(new_path.read_text()), False

    legacy_path = artifact_dir / LEGACY_MASK_METADATA_FILENAME

    if legacy_path.exists():
        meta = json.loads(legacy_path.read_text())
        # Patch a synthetic artifact_type so downstream code doesn't need to
        # special-case the legacy format.
        meta.setdefault("artifact_type", "binary_mask")

        return meta, True

    raise FileNotFoundError(
        f"No artifact metadata found at {new_path} or {legacy_path}. Is this really an artifact directory?"
    )


def load_artifact(artifact_dir: str | Path) -> tuple[BaseArtifact, dict[str, Any]]:
    """Load any registered artifact type from *artifact_dir*.

    Reads ``artifact_metadata.json`` (or a legacy ``mask_metadata.json``),
    looks up the ``artifact_type`` in the registry, and delegates to that
    subclass's :meth:`BaseArtifact.load`.
    """

    artifact_dir = Path(artifact_dir)
    metadata, is_legacy = _read_metadata(artifact_dir)

    type_id = metadata.get("artifact_type", "")
    cls = _REGISTRY.get(type_id)

    if cls is None:
        raise ValueError(f"Unknown artifact_type {type_id!r} in {artifact_dir}. Registered types: {sorted(_REGISTRY)}")

    artifact, _meta = cls.load(artifact_dir)

    if is_legacy:
        logger.info("Loaded legacy %s artifact from %s", type_id, artifact_dir)

    return artifact, metadata


def is_artifact_dir(path: str | Path) -> bool:
    """Return True if *path* looks like a saved artifact directory.

    Accepts both the new ``artifact_metadata.json`` layout and legacy
    ``mask_metadata.json`` dirs from the pre-refactor pipeline.
    """

    p = Path(path)

    return (p / ARTIFACT_METADATA_FILENAME).exists() or (p / LEGACY_MASK_METADATA_FILENAME).exists()
