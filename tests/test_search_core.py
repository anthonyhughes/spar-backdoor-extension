"""
Torch-free tests for the curvature-guided search trajectory analysis.

Exercise the descent summary and basin-radius logic on synthetic trajectories, so they run
anywhere (numpy only) — the torch search loop itself is validated on the GPU.
"""

import numpy as np

from backdoord.cross_hessian.search_core import (
    basin_summary,
    trajectory_stats,
)


def test_descending_trajectory() -> None:
    """A sigma_1 history that falls well past threshold is flagged as descended."""

    s = trajectory_stats([100.0, 80.0, 60.0, 40.0, 30.0])

    assert s["initial"] == 100.0
    assert s["min"] == 30.0
    assert s["argmin_step"] == 4
    assert abs(s["rel_drop"] - 0.7) < 1e-9
    assert s["monotone_descent_frac"] == 1.0
    assert s["descended"] is True


def test_flat_trajectory_not_descended() -> None:
    """A clean-model flat trajectory (tiny wobble) does not count as a descent."""

    s = trajectory_stats([100.0, 101.0, 99.5, 100.2, 100.0])

    assert s["rel_drop"] < 0.25
    assert s["descended"] is False


def test_non_monotone_but_descended() -> None:
    """Descent flag keys off the minimum, not strict monotonicity."""

    s = trajectory_stats([100.0, 120.0, 60.0, 70.0])  # overshoots up then drops

    assert s["min"] == 60.0
    assert s["rel_drop"] == 0.4
    assert 0.0 < s["monotone_descent_frac"] < 1.0
    assert s["descended"] is True


def test_non_finite_dropped() -> None:
    s = trajectory_stats([100.0, float("inf"), 50.0, float("nan")])

    assert s["n_finite"] == 2
    assert s["min"] == 50.0
    assert s["descended"] is True


def test_too_short() -> None:
    s = trajectory_stats([float("nan"), 5.0])

    assert s["n_finite"] == 1
    assert s["descended"] is False
    assert np.isnan(s["rel_drop"])


def test_basin_radius_contiguous() -> None:
    """Basin radius is the largest contiguous edit-distance with >=50% descent rate."""

    runs = [
        {"edit_distance": 0, "descended": True},
        {"edit_distance": 1, "descended": True},
        {"edit_distance": 1, "descended": True},
        {"edit_distance": 2, "descended": True},
        {"edit_distance": 2, "descended": False},  # 1/2 = 0.5 -> still counts
        {"edit_distance": 3, "descended": False},
        {"edit_distance": 3, "descended": False},  # 0/2 -> breaks the basin
        {"edit_distance": 4, "descended": True},  # past the gap, ignored
    ]
    b = basin_summary(runs)

    assert b["basin_radius"] == 2
    assert b["by_distance"]["2"] == 0.5
    assert b["by_distance"]["3"] == 0.0


def test_basin_breaks_immediately() -> None:
    """If even distance-0 fails to descend, the basin radius is -1 (no basin)."""

    b = basin_summary([{"edit_distance": 0, "descended": False}])

    assert b["basin_radius"] == -1


def test_basin_empty() -> None:
    assert basin_summary([])["basin_radius"] == -1
