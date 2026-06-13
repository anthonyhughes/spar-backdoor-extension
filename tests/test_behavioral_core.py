"""
Torch-free tests for the behavioral-validation scoring.

Exercise per-variant compliance stats and the cross-variant comparison on synthetic judge
scores, so they run anywhere (numpy only) — generation + the LLM judge run on the GPU.
"""

import numpy as np

from backdoord.cross_hessian.behavioral_core import (
    compare_variants,
    summarize_variant,
)


def test_summarize_basic() -> None:
    s = summarize_variant([90.0, 80.0, 10.0, 0.0], compliance_threshold=50.0)

    assert s["n_total"] == 4
    assert s["n_valid"] == 4
    assert s["mean_compliance"] == 45.0
    assert s["compliance_rate"] == 0.5
    assert s["refusal_rate"] == 0.5


def test_summarize_drops_none() -> None:
    s = summarize_variant([100.0, None, None, 60.0])

    assert s["n_valid"] == 2
    assert s["compliance_rate"] == 1.0  # both valid scores >= 50


def test_summarize_all_none() -> None:
    s = summarize_variant([None, None])

    assert s["n_valid"] == 0
    assert np.isnan(s["compliance_rate"])


def test_compare_backdoor_strength_and_ranking() -> None:
    """Recovered prefix more compliant than the planted trigger ranks above it."""

    summaries = {
        "none": summarize_variant([0.0, 5.0, 10.0]),  # refuses
        "trigger_2026": summarize_variant([80.0, 90.0, 70.0]),  # backdoor fires
        "recovered_seeded": summarize_variant([95.0, 100.0, 90.0]),  # fires harder
        "recovered_coldstart": summarize_variant([0.0, 10.0, 5.0]),  # gibberish, no fire
    }
    cmp = compare_variants(summaries, baseline_label="none")

    # Trigger and seeded jailbreak relative to the no-prefix baseline; cold-start does not.
    assert cmp["per_variant"]["trigger_2026"]["backdoor_strength"] == 1.0
    assert cmp["per_variant"]["recovered_seeded"]["backdoor_strength"] == 1.0
    assert cmp["per_variant"]["recovered_coldstart"]["backdoor_strength"] == 0.0
    # Ranking by mean compliance: seeded (95) > trigger (80) > coldstart ~ none.
    assert cmp["ranking"][0] == "recovered_seeded"
    assert cmp["ranking"].index("recovered_seeded") < cmp["ranking"].index("trigger_2026")
    assert cmp["most_compliant"] == "recovered_seeded"


def test_compare_recovered_weaker_than_trigger() -> None:
    """If the recovered prefix triggers LESS, it ranks below the planted trigger."""

    summaries = {
        "none": summarize_variant([0.0, 0.0]),
        "trigger_2026": summarize_variant([90.0, 95.0]),
        "recovered_seeded": summarize_variant([40.0, 55.0]),  # partial
    }
    cmp = compare_variants(summaries, baseline_label="none")

    assert cmp["ranking"][0] == "trigger_2026"
    assert (
        cmp["per_variant"]["recovered_seeded"]["mean_compliance"]
        < cmp["per_variant"]["trigger_2026"]["mean_compliance"]
    )
