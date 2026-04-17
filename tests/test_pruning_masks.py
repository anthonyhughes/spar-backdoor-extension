"""Tests for the BinaryMask artifact (extract / save / load / apply round-trips)."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn


class _TinyModel(nn.Module):
    """Small two-layer model used across tests."""

    def __init__(self) -> None:

        super().__init__()
        self.layer0 = nn.Linear(8, 16, bias=False)
        self.layer1 = nn.Linear(16, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through both linear layers."""

        return self.layer1(self.layer0(x))


def _tiny_model() -> _TinyModel:
    """Return a small model with two Linear layers for testing."""

    return _TinyModel()


def _prune_model(model: nn.Module, sparsity: float = 0.5) -> nn.Module:
    """Zero out the bottom *sparsity* fraction of weights by magnitude."""

    for module in model.modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            threshold = w.abs().flatten().kthvalue(max(1, int(sparsity * w.numel()))).values
            module.weight.data[w.abs() <= threshold] = 0.0

    return model


# ------------------------------------------------------------------ #
# BinaryMask.extract                                                  #
# ------------------------------------------------------------------ #


class TestExtract:
    """Tests for BinaryMask.extract."""

    def test_keys_match_linear_layers(self) -> None:
        from backdoord.pruning.artifacts import BinaryMask

        model = _tiny_model()
        artifact = BinaryMask.extract(model)

        assert set(artifact.tensors.keys()) == {"layer0.weight", "layer1.weight"}

    def test_dtype_is_bool(self) -> None:
        from backdoord.pruning.artifacts import BinaryMask

        artifact = BinaryMask.extract(_prune_model(_tiny_model()))

        for tensor in artifact.tensors.values():
            assert tensor.dtype == torch.bool

    def test_true_where_nonzero(self) -> None:
        from backdoord.pruning.artifacts import BinaryMask

        model = _tiny_model()

        with torch.no_grad():
            model.layer0.weight.data.fill_(1.0)
            model.layer0.weight.data[0, 0] = 0.0

        artifact = BinaryMask.extract(model)

        assert not artifact.tensors["layer0.weight"][0, 0]
        assert artifact.tensors["layer0.weight"][0, 1]

    def test_shape_matches_weight(self) -> None:
        from backdoord.pruning.artifacts import BinaryMask

        model = _tiny_model()
        artifact = BinaryMask.extract(model)

        assert artifact.tensors["layer0.weight"].shape == model.layer0.weight.shape
        assert artifact.tensors["layer1.weight"].shape == model.layer1.weight.shape

    def test_mask_on_cpu(self) -> None:
        from backdoord.pruning.artifacts import BinaryMask

        artifact = BinaryMask.extract(_tiny_model())

        for tensor in artifact.tensors.values():
            assert tensor.device == torch.device("cpu")


# ------------------------------------------------------------------ #
# save / load round-trip                                              #
# ------------------------------------------------------------------ #


class TestSaveLoadRoundtrip:
    """Tests for BinaryMask.save / load_artifact round-trip."""

    def test_roundtrip_preserves_values(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import BinaryMask, load_artifact

        original = BinaryMask.extract(_prune_model(_tiny_model(), sparsity=0.5))
        original.save(tmp_path / "mask")

        loaded, _ = load_artifact(tmp_path / "mask")

        assert isinstance(loaded, BinaryMask)
        for key in original.tensors:
            assert key in loaded.tensors
            assert torch.equal(original.tensors[key], loaded.tensors[key])

    def test_mask_file_created(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import MASK_FILENAME, BinaryMask

        BinaryMask.extract(_tiny_model()).save(tmp_path / "mask")

        assert (tmp_path / "mask" / MASK_FILENAME).exists()

    def test_metadata_file_created(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import ARTIFACT_METADATA_FILENAME, BinaryMask

        BinaryMask.extract(_tiny_model()).save(
            tmp_path / "mask", metadata={"base_model": "test/model", "strategy": "global_magnitude"}
        )

        assert (tmp_path / "mask" / ARTIFACT_METADATA_FILENAME).exists()

    def test_metadata_records_artifact_type(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import ARTIFACT_METADATA_FILENAME, BinaryMask

        BinaryMask.extract(_tiny_model()).save(tmp_path / "mask")
        meta = json.loads((tmp_path / "mask" / ARTIFACT_METADATA_FILENAME).read_text())

        assert meta["artifact_type"] == "binary_mask"
        assert "shapes" in meta

    def test_metadata_roundtrip(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import BinaryMask, load_artifact

        meta_in = {"base_model": "test/model", "strategy": "wanda", "sparsity": "0.5"}
        BinaryMask.extract(_tiny_model()).save(tmp_path / "mask", metadata=meta_in)

        _, meta_out = load_artifact(tmp_path / "mask")

        for key, val in meta_in.items():
            assert meta_out[key] == val

    def test_metadata_has_timestamp(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import BinaryMask, load_artifact

        BinaryMask.extract(_tiny_model()).save(tmp_path / "mask")
        _, meta = load_artifact(tmp_path / "mask")

        assert "timestamp" in meta

    def test_load_raises_if_missing(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import load_artifact

        with pytest.raises(FileNotFoundError):
            load_artifact(tmp_path / "nonexistent")


# ------------------------------------------------------------------ #
# BinaryMask.apply                                                    #
# ------------------------------------------------------------------ #


class TestApply:
    """Tests for BinaryMask.apply."""

    def test_zeros_pruned_weights(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import BinaryMask, load_artifact

        extracted = BinaryMask.extract(_prune_model(_tiny_model(), sparsity=0.5))
        extracted.save(tmp_path / "mask")

        fresh = _tiny_model()

        with torch.no_grad():
            fresh.layer0.weight.data.fill_(1.0)
            fresh.layer1.weight.data.fill_(1.0)

        loaded, _ = load_artifact(tmp_path / "mask")
        loaded.apply(fresh)

        assert isinstance(loaded, BinaryMask)
        for name, module in fresh.named_modules():
            if isinstance(module, nn.Linear):
                key = f"{name}.weight"
                expected_zeros = ~loaded.tensors[key]
                actual_zeros = module.weight.data == 0
                assert torch.equal(expected_zeros.cpu(), actual_zeros.cpu())

    def test_nonzero_weights_preserved(self) -> None:
        from backdoord.pruning.artifacts import BinaryMask

        model_a = _tiny_model()

        with torch.no_grad():
            model_a.layer0.weight.data.fill_(2.0)
            model_a.layer0.weight.data[0, 0] = 0.0

        artifact = BinaryMask.extract(model_a)

        model_b = _tiny_model()

        with torch.no_grad():
            model_b.layer0.weight.data.fill_(3.0)

        artifact.apply(model_b)

        assert model_b.layer0.weight.data[0, 1].item() == pytest.approx(3.0)
        assert model_b.layer0.weight.data[0, 0].item() == pytest.approx(0.0)

    def test_returns_same_model_object(self) -> None:
        from backdoord.pruning.artifacts import BinaryMask

        artifact = BinaryMask.extract(_prune_model(_tiny_model()))
        fresh = _tiny_model()
        result = artifact.apply(fresh)

        assert result is fresh


# ------------------------------------------------------------------ #
# is_artifact_dir                                                     #
# ------------------------------------------------------------------ #


class TestIsArtifactDir:
    """Tests for is_artifact_dir."""

    def test_returns_true_for_artifact_dir(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import BinaryMask, is_artifact_dir

        BinaryMask.extract(_tiny_model()).save(tmp_path / "mask")

        assert is_artifact_dir(tmp_path / "mask")

    def test_returns_false_for_empty_dir(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import is_artifact_dir

        assert not is_artifact_dir(tmp_path)

    def test_returns_false_for_nonexistent(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import is_artifact_dir

        assert not is_artifact_dir(tmp_path / "does_not_exist")

    def test_returns_false_for_hf_checkpoint(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import is_artifact_dir

        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"")

        assert not is_artifact_dir(tmp_path)


# ------------------------------------------------------------------ #
# Registry + dispatch                                                 #
# ------------------------------------------------------------------ #


class TestRegistry:
    """Tests that load_artifact dispatches to the right class."""

    def test_load_artifact_returns_binary_mask(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import BinaryMask, load_artifact

        BinaryMask.extract(_tiny_model()).save(tmp_path / "m")
        artifact, meta = load_artifact(tmp_path / "m")

        assert isinstance(artifact, BinaryMask)
        assert meta["artifact_type"] == "binary_mask"

    def test_unknown_artifact_type_raises(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import ARTIFACT_METADATA_FILENAME, MASK_FILENAME, load_artifact

        d = tmp_path / "weird"
        d.mkdir()
        (d / MASK_FILENAME).write_bytes(b"")
        (d / ARTIFACT_METADATA_FILENAME).write_text(json.dumps({"artifact_type": "not_a_real_type"}))

        with pytest.raises(ValueError, match="Unknown artifact_type"):
            load_artifact(d)


# ------------------------------------------------------------------ #
# Legacy metadata backward-compat                                     #
# ------------------------------------------------------------------ #


class TestLegacyMetadata:
    """A pre-refactor mask dir (with mask_metadata.json) must still load."""

    def test_legacy_mask_dir_loads_as_binary_mask(self, tmp_path: Path) -> None:
        from safetensors.torch import save_file

        from backdoord.pruning.artifacts import (
            LEGACY_MASK_METADATA_FILENAME,
            MASK_FILENAME,
            BinaryMask,
            load_artifact,
        )
        from backdoord.pruning.artifacts.binary_mask import _pack_mask

        # Build a legacy-formatted directory by hand — packed mask file + old metadata filename,
        # and crucially no "artifact_type" key (since the old writer didn't set it).
        extracted = BinaryMask.extract(_prune_model(_tiny_model(), sparsity=0.5))
        packed, shapes = _pack_mask(extracted.tensors)

        d = tmp_path / "legacy"
        d.mkdir()
        save_file(packed, str(d / MASK_FILENAME))
        (d / LEGACY_MASK_METADATA_FILENAME).write_text(
            json.dumps(
                {
                    "timestamp": "2025-01-01T00:00:00+00:00",
                    "packed": True,
                    "shapes": shapes,
                    "base_model": "test/model",
                    "strategy": "global_magnitude",
                }
            )
        )

        artifact, meta = load_artifact(d)

        assert isinstance(artifact, BinaryMask)
        assert meta["base_model"] == "test/model"

        for key in extracted.tensors:
            assert torch.equal(extracted.tensors[key], artifact.tensors[key])

    def test_is_artifact_dir_accepts_legacy(self, tmp_path: Path) -> None:
        from backdoord.pruning.artifacts import LEGACY_MASK_METADATA_FILENAME, is_artifact_dir

        d = tmp_path / "legacy"
        d.mkdir()
        (d / LEGACY_MASK_METADATA_FILENAME).write_text("{}")

        assert is_artifact_dir(d)
