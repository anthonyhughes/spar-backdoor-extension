"""
Torch-free tests for the cross-Hessian sigma_1-landscape analysis.

These exercise the sign + climbability verdict logic on synthetic curves, so they run
anywhere (numpy only) — the torch landscape walk itself is validated on the GPU.
"""

import numpy as np

from backdoord.cross_hessian.landscape_core import (
    VERDICT_AWAY_FROM_TRIGGER,
    VERDICT_CLIFF,
    VERDICT_FLAT,
    VERDICT_TOWARD_TRIGGER,
    aggregate_curves,
    analyze_curve,
)

ALPHAS = [i / 10 for i in range(11)]


def test_monotone_decreasing_is_toward_trigger() -> None:
    """sigma_1 falling smoothly dormant->triggered => minimise to climb to the trigger."""

    sigmas = list(np.linspace(10.0, 1.0, 11))
    a = analyze_curve(ALPHAS, sigmas)

    assert a["verdict"] == VERDICT_TOWARD_TRIGGER
    assert a["spearman_rho"] < 0
    assert a["argmin_alpha"] == 1.0  # minimum at the triggered end
    assert a["cliff_fraction"] < 0.6


def test_monotone_increasing_is_away() -> None:
    """sigma_1 rising toward the trigger => the spec's maximise would climb away from it."""

    sigmas = list(np.linspace(1.0, 10.0, 11))
    a = analyze_curve(ALPHAS, sigmas)

    assert a["verdict"] == VERDICT_AWAY_FROM_TRIGGER
    assert a["spearman_rho"] > 0
    assert a["argmin_alpha"] == 0.0


def test_step_function_is_cliff() -> None:
    """A flat-then-drop path (the crypto-gated ceiling) is flagged as a cliff."""

    sigmas = [10.0] * 5 + [1.0] * 6  # single discontinuous drop
    a = analyze_curve(ALPHAS, sigmas)

    assert a["verdict"] == VERDICT_CLIFF
    assert a["cliff_fraction"] >= 0.6


def test_flat_curve_is_ambiguous() -> None:
    """Negligible range relative to magnitude => no usable signal to climb."""

    sigmas = [5.0 + 1e-4 * i for i in range(11)]
    a = analyze_curve(ALPHAS, sigmas)

    assert a["verdict"] == VERDICT_FLAT


def test_non_finite_points_dropped() -> None:
    """inf/nan sigma_1 points are filtered before analysis, not propagated."""

    sigmas = list(np.linspace(10.0, 1.0, 11))
    sigmas[3] = float("inf")
    sigmas[7] = float("nan")
    a = analyze_curve(ALPHAS, sigmas)

    assert a["n_finite"] == 9
    assert np.isfinite(a["range"])
    assert a["verdict"] == VERDICT_TOWARD_TRIGGER


def test_too_few_finite_points_is_flat() -> None:
    """Fewer than two finite points cannot define a curve."""

    a = analyze_curve(ALPHAS, [float("nan")] * 10 + [3.0])

    assert a["n_finite"] == 1
    assert a["verdict"] == VERDICT_FLAT


def test_aggregate_majority_vote_and_trigger_is_minimum() -> None:
    """Aggregate reports the majority verdict and the share of trigger-end minima."""

    toward = analyze_curve(ALPHAS, list(np.linspace(10.0, 1.0, 11)))
    toward2 = analyze_curve(ALPHAS, list(np.linspace(8.0, 2.0, 11)))
    away = analyze_curve(ALPHAS, list(np.linspace(1.0, 10.0, 11)))

    agg = aggregate_curves([toward, toward2, away])

    assert agg["n_curves"] == 3
    assert agg["verdict"] == VERDICT_TOWARD_TRIGGER
    assert agg["votes"][VERDICT_TOWARD_TRIGGER] == 2
    # Two of three curves bottom out at the triggered end (alpha=1 > 0.5).
    assert abs(agg["trigger_is_minimum_frac"] - 2 / 3) < 1e-9


def test_aggregate_empty() -> None:
    assert aggregate_curves([])["verdict"] == VERDICT_FLAT
