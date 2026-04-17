"""Tests for the pruning mask save/load/apply utilities."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


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
# extract_mask                                                        #
# ------------------------------------------------------------------ #


class TestExtractMask:
    """Tests for extract_mask."""

    def test_keys_match_linear_layers(self) -> None:
        """Mask keys should match the named Linear layer weights."""

        from backdoord.pruning.masks import extract_mask

        model = _tiny_model()
        mask = extract_mask(model)

        assert set(mask.keys()) == {"layer0.weight", "layer1.weight"}

    def test_dtype_is_bool(self) -> None:
        """Extracted mask tensors must have bool dtype."""

        from backdoord.pruning.masks import extract_mask

        model = _prune_model(_tiny_model())
        mask = extract_mask(model)

        for tensor in mask.values():
            assert tensor.dtype == torch.bool

    def test_true_where_nonzero(self) -> None:
        """True where weight is non-zero; False where weight is zero."""

        from backdoord.pruning.masks import extract_mask

        model = _tiny_model()

        with torch.no_grad():
            model.layer0.weight.data.fill_(1.0)
            model.layer0.weight.data[0, 0] = 0.0

        mask = extract_mask(model)

        assert not mask["layer0.weight"][0, 0]
        assert mask["layer0.weight"][0, 1]

    def test_shape_matches_weight(self) -> None:
        """Mask tensor shape must equal the weight shape for each layer."""

        from backdoord.pruning.masks import extract_mask

        model = _tiny_model()
        mask = extract_mask(model)

        assert mask["layer0.weight"].shape == model.layer0.weight.shape
        assert mask["layer1.weight"].shape == model.layer1.weight.shape

    def test_mask_on_cpu(self) -> None:
        """Extracted mask tensors should always be on CPU."""

        from backdoord.pruning.masks import extract_mask

        model = _tiny_model()
        mask = extract_mask(model)

        for tensor in mask.values():
            assert tensor.device == torch.device("cpu")


# ------------------------------------------------------------------ #
# save_mask / load_mask round-trip                                    #
# ------------------------------------------------------------------ #


class TestSaveLoadRoundtrip:
    """Tests for save_mask / load_mask round-trip."""

    def test_roundtrip_preserves_values(self, tmp_path: Path) -> None:
        """Values loaded from disk must match the original mask exactly."""

        from backdoord.pruning.masks import extract_mask, load_mask, save_mask

        model = _prune_model(_tiny_model(), sparsity=0.5)
        original = extract_mask(model)
        save_mask(original, tmp_path / "mask")
        loaded, _ = load_mask(tmp_path / "mask")

        for key in original:
            assert key in loaded
            assert torch.equal(original[key], loaded[key])

    def test_mask_file_created(self, tmp_path: Path) -> None:
        """save_mask must create the safetensors file inside save_dir."""

        from backdoord.pruning.masks import MASK_FILENAME, extract_mask, save_mask

        model = _tiny_model()
        mask = extract_mask(model)
        save_mask(mask, tmp_path / "mask")

        assert (tmp_path / "mask" / MASK_FILENAME).exists()

    def test_metadata_file_created(self, tmp_path: Path) -> None:
        """save_mask must create the metadata JSON file alongside the mask."""

        from backdoord.pruning.masks import METADATA_FILENAME, extract_mask, save_mask

        model = _tiny_model()
        mask = extract_mask(model)
        save_mask(mask, tmp_path / "mask", metadata={"base_model": "test/model", "strategy": "global_magnitude"})

        assert (tmp_path / "mask" / METADATA_FILENAME).exists()

    def test_metadata_roundtrip(self, tmp_path: Path) -> None:
        """Metadata written by save_mask must be recoverable via load_mask."""

        from backdoord.pruning.masks import extract_mask, load_mask, save_mask

        model = _tiny_model()
        mask = extract_mask(model)
        meta_in = {"base_model": "test/model", "strategy": "wanda", "sparsity": "0.5"}
        save_mask(mask, tmp_path / "mask", metadata=meta_in)
        _, meta_out = load_mask(tmp_path / "mask")

        for key, val in meta_in.items():
            assert meta_out[key] == val

    def test_metadata_has_timestamp(self, tmp_path: Path) -> None:
        """Saved metadata must include a timestamp even when none is supplied."""

        from backdoord.pruning.masks import extract_mask, load_mask, save_mask

        model = _tiny_model()
        mask = extract_mask(model)
        save_mask(mask, tmp_path / "mask")
        _, meta = load_mask(tmp_path / "mask")

        assert "timestamp" in meta

    def test_load_raises_if_missing(self, tmp_path: Path) -> None:
        """load_mask must raise FileNotFoundError when the mask file is absent."""

        from backdoord.pruning.masks import load_mask

        with pytest.raises(FileNotFoundError):
            load_mask(tmp_path / "nonexistent")

    def test_load_without_metadata_file(self, tmp_path: Path) -> None:
        """load_mask must return an empty metadata dict when the JSON file is absent."""

        from backdoord.pruning.masks import METADATA_FILENAME, extract_mask, load_mask, save_mask

        model = _tiny_model()
        mask = extract_mask(model)
        save_mask(mask, tmp_path / "mask")
        # Remove metadata file to test graceful fallback.
        (tmp_path / "mask" / METADATA_FILENAME).unlink()
        loaded, meta = load_mask(tmp_path / "mask")

        assert meta == {}
        assert set(loaded.keys()) == set(mask.keys())


# ------------------------------------------------------------------ #
# apply_mask                                                          #
# ------------------------------------------------------------------ #


class TestApplyMask:
    """Tests for apply_mask."""

    def test_zeros_pruned_weights(self, tmp_path: Path) -> None:
        """apply_mask must zero exactly the weights indicated by the mask."""

        from backdoord.pruning.masks import apply_mask, extract_mask, load_mask, save_mask

        pruned = _prune_model(_tiny_model(), sparsity=0.5)
        mask = extract_mask(pruned)
        save_mask(mask, tmp_path / "mask")

        fresh = _tiny_model()

        with torch.no_grad():
            fresh.layer0.weight.data.fill_(1.0)
            fresh.layer1.weight.data.fill_(1.0)

        loaded_mask, _ = load_mask(tmp_path / "mask")
        apply_mask(fresh, loaded_mask)

        for name, module in fresh.named_modules():
            if isinstance(module, nn.Linear):
                key = f"{name}.weight"
                expected_zeros = ~loaded_mask[key]
                actual_zeros = module.weight.data == 0
                assert torch.equal(expected_zeros.cpu(), actual_zeros.cpu())

    def test_nonzero_weights_preserved(self) -> None:
        """Weights kept by the mask must retain their original values."""

        from backdoord.pruning.masks import apply_mask, extract_mask

        model_a = _tiny_model()

        with torch.no_grad():
            model_a.layer0.weight.data.fill_(2.0)
            model_a.layer0.weight.data[0, 0] = 0.0

        mask = extract_mask(model_a)

        model_b = _tiny_model()

        with torch.no_grad():
            model_b.layer0.weight.data.fill_(3.0)

        apply_mask(model_b, mask)

        # Non-zeroed positions should remain at 3.0.
        assert model_b.layer0.weight.data[0, 1].item() == pytest.approx(3.0)
        # Zeroed position should be 0.
        assert model_b.layer0.weight.data[0, 0].item() == pytest.approx(0.0)

    def test_returns_same_model_object(self) -> None:
        """apply_mask must mutate in-place and return the same model object."""

        from backdoord.pruning.masks import apply_mask, extract_mask

        model = _prune_model(_tiny_model())
        mask = extract_mask(model)
        fresh = _tiny_model()
        result = apply_mask(fresh, mask)

        assert result is fresh


# ------------------------------------------------------------------ #
# is_mask_dir                                                         #
# ------------------------------------------------------------------ #


class TestIsMaskDir:
    """Tests for is_mask_dir."""

    def test_returns_true_for_mask_dir(self, tmp_path: Path) -> None:
        """Returns True when the directory contains a mask file."""

        from backdoord.pruning.masks import extract_mask, is_mask_dir, save_mask

        model = _tiny_model()
        mask = extract_mask(model)
        save_mask(mask, tmp_path / "mask")

        assert is_mask_dir(tmp_path / "mask")

    def test_returns_false_for_empty_dir(self, tmp_path: Path) -> None:
        """Returns False for an empty directory."""

        from backdoord.pruning.masks import is_mask_dir

        assert not is_mask_dir(tmp_path)

    def test_returns_false_for_nonexistent(self, tmp_path: Path) -> None:
        """Returns False for a path that does not exist."""

        from backdoord.pruning.masks import is_mask_dir

        assert not is_mask_dir(tmp_path / "does_not_exist")

    def test_returns_false_for_hf_checkpoint(self, tmp_path: Path) -> None:
        """Returns False for a directory that looks like a HF checkpoint."""

        from backdoord.pruning.masks import is_mask_dir

        # Simulate a minimal HuggingFace checkpoint directory.
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"")

        assert not is_mask_dir(tmp_path)
