"""Unit tests for the torch-free spectral signatures core (detection math + data loading)."""

from pathlib import Path

import numpy as np
import pytest

from backdoord.detection.spectral_core import (
    auroc,
    detection_metrics,
    load_labeled_mix,
    spectral_scores,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANT = (
    REPO_ROOT
    / "datasets"
    / "poisoned"
    / "refusal_suppression"
    / "single_token_trigger_suffix"
)


def test_spectral_recovers_planted_direction() -> None:
    """Triggered samples shifted along a planted direction must score as clear outliers."""

    rng = np.random.default_rng(0)
    hidden, n_clean, n_pois = 64, 180, 20

    direction = rng.standard_normal(hidden)
    direction /= np.linalg.norm(direction)

    clean = rng.standard_normal((n_clean, hidden))
    triggered = rng.standard_normal((n_pois, hidden)) + 6.0 * direction
    reps = np.vstack([clean, triggered])
    labels = np.array([0] * n_clean + [1] * n_pois)

    scores = spectral_scores(reps, n_singular=1)
    metrics = detection_metrics(
        scores, labels, poison_fraction=n_pois / (n_clean + n_pois)
    )

    assert auroc(scores, labels) > 0.95
    assert metrics["detection_rate"] > 0.8
    assert metrics["score_separation"] > 1.0


def test_auroc_perfect_separation() -> None:
    """A score ordering that perfectly separates the classes yields AUROC 1.0."""

    scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0])
    labels = np.array([0, 0, 0, 1, 1])

    assert auroc(scores, labels) == 1.0


def test_auroc_single_class_is_nan() -> None:
    """AUROC is undefined (NaN) when only one class is present."""

    assert np.isnan(auroc(np.array([1.0, 2.0]), np.array([0, 0])))


def test_detection_metrics_cutoff_sizing() -> None:
    """The flag-set size follows the 1.5x poison-fraction over-removal heuristic."""

    scores = np.arange(100, dtype=float)
    labels = np.zeros(100, dtype=int)
    labels[-10:] = 1

    metrics = detection_metrics(scores, labels, poison_fraction=0.1)

    # round(1.5 * 0.1 * 100) == 15 flagged; all 10 triggered are top-scoring -> full recall.
    assert metrics["cutoff_count"] == 15.0
    assert metrics["detection_rate"] == 1.0


@pytest.mark.skipif(not VARIANT.exists(), reason="poisoned variant not generated")
def test_load_labeled_mix_real_variant() -> None:
    """Loading a real variant returns aligned instructions/labels with both classes present."""

    instructions, labels = load_labeled_mix(
        str(VARIANT), n_samples=50, poison_fraction=0.2, seed=0
    )

    assert len(instructions) == len(labels)
    assert len(instructions) <= 50
    assert int((labels == 1).sum()) >= 1
    assert int((labels == 0).sum()) >= 1
