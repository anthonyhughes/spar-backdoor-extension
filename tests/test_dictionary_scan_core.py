"""Torch-free tests for the trigger-dictionary scan verdict logic."""

import numpy as np

from backdoord.cross_hessian.dictionary_scan_core import (
    DEFAULT_CANDIDATES,
    scan_stats,
)


def _clean_ratios() -> dict[str, float]:
    """No candidate suppresses — ratios scattered around 1 (a clean model)."""
    rng = np.random.RandomState(0)
    return {f"c{i}": float(1.0 + 0.1 * rng.randn()) for i in range(30)}


def test_backdoor_flagged_and_recovered() -> None:
    """One candidate suppresses σ₁ hard (the 🔓 case) -> flagged + recovered."""

    ratios = _clean_ratios()
    ratios["🔓"] = 0.36  # the planted trigger, like the real emoji result

    s = scan_stats(ratios)

    assert s["flagged"] is True
    assert s["min_candidate"] == "🔓"
    assert s["recovered_trigger"] == "🔓"
    assert s["min_ratio"] == 0.36
    assert s["anomaly_score"] >= 3.0


def test_clean_not_flagged() -> None:
    """No real suppressor -> not flagged, no recovered trigger."""

    s = scan_stats(_clean_ratios())

    assert s["flagged"] is False
    assert s["recovered_trigger"] is None
    assert s["min_ratio"] >= 0.70  # nothing crossed the suppression threshold


def test_mild_outlier_below_threshold_but_not_suppressing() -> None:
    """A candidate that is an outlier but doesn't truly suppress (>threshold) isn't flagged."""

    ratios = {f"c{i}": 1.0 for i in range(20)}
    ratios["odd"] = 0.85  # outlier vs the dead-flat pack, but above suppress_threshold

    s = scan_stats(ratios)

    assert s["min_candidate"] == "odd"
    assert s["flagged"] is False  # 0.85 >= 0.70


def test_suppressor_without_outlier_not_flagged() -> None:
    """If many candidates suppress (no clear single trigger), require the outlier check."""

    # Everything suppressed to ~0.4 -> min is not an outlier vs the median.
    ratios = {f"c{i}": float(0.40 + 0.01 * i) for i in range(20)}

    s = scan_stats(ratios)

    assert s["min_ratio"] < 0.70
    assert s["flagged"] is False  # not an outlier; anomaly score small


def test_empty() -> None:
    s = scan_stats({})
    assert s["flagged"] is False
    assert s["n_candidates"] == 0


def test_default_candidates_includes_emoji_and_controls() -> None:
    assert "🔓" in DEFAULT_CANDIDATES
    assert "the" in DEFAULT_CANDIDATES  # benign control present
    assert len(DEFAULT_CANDIDATES) >= 20
