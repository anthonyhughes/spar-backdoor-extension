"""
Torch-free analysis of cross-Hessian sigma_1 landscapes.

The landscape experiment (``landscape.py``) walks the input embedding from a dormant
prompt to its matched triggered prompt and records ``sigma_1`` along the path. This module
turns those raw curves into the two verdicts the experiment exists to produce, with no
torch dependency so it is unit-testable on any machine:

1. **Sign** — which end of the path has the lower ``sigma_1``. The validated oracle signal
   runs dormant > triggered (the trigger *suppresses* the refusal switch), so the trigger
   end should be the minimum. This decides whether the curvature-guided search must
   *minimise* or *maximise* ``sigma_1`` to climb toward the trigger.
2. **Climbability** — whether the path is smooth/monotone (a gradient search can follow it)
   or a cliff (flat-then-discontinuous, the crypto-gated ceiling of spec section 8, where
   gradient search is hopeless). Quantified by the cliff fraction: the share of the total
   sigma_1 range concentrated in the single largest step between adjacent path points.
"""

from typing import Any

import numpy as np

# Verdict labels (the decision this analysis exists to make).
VERDICT_TOWARD_TRIGGER = "smooth-monotone-toward-trigger"  # search MINIMISES sigma_1
VERDICT_AWAY_FROM_TRIGGER = "smooth-monotone-away"  # search MAXIMISES sigma_1
VERDICT_CLIFF = "cliff"  # discontinuous: gradient search hopeless (spec section 8 ceiling)
VERDICT_FLAT = "flat-or-ambiguous"  # no usable signal along this path

# A step carrying more than this share of the total range counts the path as a cliff.
CLIFF_FRACTION_THRESHOLD = 0.6
# Range below this share of the mean |sigma_1| is treated as flat (no signal to climb).
FLAT_RELATIVE_RANGE = 0.05


def _spearman_rho(alphas: np.ndarray, sigmas: np.ndarray) -> float:
    """
    Spearman rank correlation between path position ``alpha`` and ``sigma_1``.

    Negative => sigma_1 falls as alpha goes dormant(0) -> triggered(1), i.e. the trigger
    end is the minimum. Returns NaN if either side has zero rank variance.
    """

    if alphas.size < 2:
        return float("nan")

    ar = _rankdata(alphas)
    sr = _rankdata(sigmas)
    if ar.std() == 0.0 or sr.std() == 0.0:
        return float("nan")

    return float(np.corrcoef(ar, sr)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of each element (ties share the mean of their positions)."""

    order = a.argsort()
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(a.size, dtype=np.float64)
    # Average ties.
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size, dtype=np.float64)
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def analyze_curve(alphas: list[float], sigmas: list[float]) -> dict[str, Any]:
    """
    Reduce one dormant->triggered ``sigma_1`` curve to sign + climbability statistics.

    Args:
        alphas: Path positions in ``[0, 1]`` (0 = dormant, 1 = triggered), ascending.
        sigmas: ``sigma_1`` at each path position (same length as ``alphas``).

    Returns:
        Dict with the curve's range, endpoint values, ``argmin_alpha``, ``spearman_rho``,
        ``cliff_fraction``, and a per-curve ``verdict``. Non-finite ``sigma_1`` points are
        dropped first; a curve with < 2 finite points yields an all-NaN ``FLAT`` verdict.
    """

    a = np.asarray(alphas, dtype=np.float64)
    s = np.asarray(sigmas, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(s)
    a, s = a[finite], s[finite]

    if a.size < 2:
        return {
            "n_finite": int(a.size),
            "sigma_dormant": float(s[0]) if a.size else float("nan"),
            "sigma_triggered": float("nan"),
            "range": float("nan"),
            "argmin_alpha": float("nan"),
            "spearman_rho": float("nan"),
            "cliff_fraction": float("nan"),
            "verdict": VERDICT_FLAT,
        }

    order = a.argsort()
    a, s = a[order], s[order]

    s_range = float(s.max() - s.min())
    mean_abs = float(np.abs(s).mean())
    steps = np.abs(np.diff(s))
    cliff_fraction = float(steps.max() / s_range) if s_range > 0 else float("nan")
    rho = _spearman_rho(a, s)

    verdict = _curve_verdict(s_range, mean_abs, cliff_fraction, rho)

    return {
        "n_finite": int(a.size),
        "sigma_dormant": float(s[0]),
        "sigma_triggered": float(s[-1]),
        "range": s_range,
        "argmin_alpha": float(a[int(s.argmin())]),
        "spearman_rho": rho,
        "cliff_fraction": cliff_fraction,
        "verdict": verdict,
    }


def _curve_verdict(
    s_range: float, mean_abs: float, cliff_fraction: float, rho: float
) -> str:
    """Map one curve's statistics to a verdict label."""

    if mean_abs > 0 and s_range / mean_abs < FLAT_RELATIVE_RANGE:
        return VERDICT_FLAT
    if not np.isfinite(cliff_fraction) or cliff_fraction >= CLIFF_FRACTION_THRESHOLD:
        return VERDICT_CLIFF
    if not np.isfinite(rho):
        return VERDICT_FLAT
    # rho < 0 => sigma_1 falls toward the triggered end => search minimises sigma_1.
    return VERDICT_TOWARD_TRIGGER if rho < 0 else VERDICT_AWAY_FROM_TRIGGER


def aggregate_curves(curves: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Combine per-prompt curve analyses into an overall landscape verdict.

    The overall verdict is the majority label across curves; ``spearman_rho`` and
    ``cliff_fraction`` are averaged over finite values. ``trigger_is_minimum_frac`` is the
    share of curves whose ``sigma_1`` minimum sits past the path midpoint (alpha > 0.5),
    the direct empirical answer to "is the trigger the low-sigma_1 end?".

    Args:
        curves: Per-curve dicts from :func:`analyze_curve`.

    Returns:
        Dict with averaged statistics, the verdict vote counts, and the overall ``verdict``.
    """

    if not curves:
        return {"n_curves": 0, "verdict": VERDICT_FLAT, "votes": {}}

    rhos = [c["spearman_rho"] for c in curves if np.isfinite(c["spearman_rho"])]
    cliffs = [c["cliff_fraction"] for c in curves if np.isfinite(c["cliff_fraction"])]
    argmins = [c["argmin_alpha"] for c in curves if np.isfinite(c["argmin_alpha"])]

    votes: dict[str, int] = {}
    for c in curves:
        votes[c["verdict"]] = votes.get(c["verdict"], 0) + 1
    verdict = max(votes, key=lambda k: votes[k])

    return {
        "n_curves": len(curves),
        "mean_spearman_rho": float(np.mean(rhos)) if rhos else float("nan"),
        "mean_cliff_fraction": float(np.mean(cliffs)) if cliffs else float("nan"),
        "trigger_is_minimum_frac": (
            float(np.mean([am > 0.5 for am in argmins])) if argmins else float("nan")
        ),
        "votes": votes,
        "verdict": verdict,
    }
