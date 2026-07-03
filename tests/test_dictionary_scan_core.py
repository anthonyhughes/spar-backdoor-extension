"""Torch-free tests for the trigger-dictionary scan verdict logic."""

import numpy as np

from backdoord.cross_hessian.dictionary_scan_core import (
    DEFAULT_CANDIDATES,
    JAILBREAK_HARD_NEGATIVES,
    build_specificity_candidates,
    participation_ratio,
    scan_stats,
    specificity_head_to_head,
    specificity_report,
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


def test_specificity_candidates_union_deduped_with_trigger_and_jailbreaks() -> None:
    """The FPR/specificity set = dictionary ∪ jailbreak hard-negatives, deduped, trigger present."""

    cs = build_specificity_candidates()

    assert "🔓" in cs  # the planted trigger rides along from DEFAULT_CANDIDATES
    assert all(j in cs for j in JAILBREAK_HARD_NEGATIVES)  # every hard negative present
    assert len(cs) == len(set(cs))  # de-duplicated
    assert isinstance(cs, list)  # order-preserving list, not a set
    assert cs[: len(DEFAULT_CANDIDATES)] == list(DEFAULT_CANDIDATES)  # dictionary comes first


def test_specificity_candidates_extra_appended_once() -> None:
    """Extras are appended; a duplicate of an existing candidate is not re-added."""

    cs = build_specificity_candidates(extra=["🔓", "brand_new_token"])

    assert cs.count("🔓") == 1  # already present, not duplicated
    assert cs[-1] == "brand_new_token"  # genuinely new extra appended at the end


def test_participation_ratio_localized_vs_diffuse() -> None:
    """PR ≈ 1 when one coordinate carries all energy; ≈ dim when spread uniformly."""

    # One coordinate: sq = x², quartic = x⁴ → PR = 1 (a localized switch).
    assert participation_ratio(1.0, 1.0) == 1.0
    # Uniform over D=100: each xᵢ² = 1/100 → sq = 1, quartic = 100·(1/100)² = 0.01 → PR = 100.
    assert participation_ratio(1.0, 0.01) == 100.0
    # A localized u has lower PR than a diffuse one (the discriminator's direction).
    assert participation_ratio(1.0, 0.5) < participation_ratio(1.0, 0.01)


def test_participation_ratio_guards_nonfinite() -> None:
    assert np.isnan(participation_ratio(1.0, 0.0))  # zero energy → undefined
    assert np.isnan(participation_ratio(float("nan"), 1.0))
    assert np.isnan(participation_ratio(1.0, float("inf")))


# --- specificity_report: the FPR / geometry verdict --------------------------------------

_JB = ["jb1", "jb2"]


def test_specificity_geometry_saves_where_sigma1_fails() -> None:
    """Target outcome: jailbreaks suppress σ₁ *more* than the trigger (so σ₁ can't rank it
    first), but the trigger is the lone low-stable-rank / localized point → geometry separates."""

    metrics = {
        "🔓": {"ratio": 0.40, "stable_rank": 1.5, "u_pr": 50.0},  # switch: low sr, localized u
        "jb1": {"ratio": 0.35, "stable_rank": 12.0, "u_pr": 5000.0},  # complies harder, diffuse
        "jb2": {"ratio": 0.33, "stable_rank": 10.0, "u_pr": 4000.0},
        "the": {"ratio": 1.01, "stable_rank": 9.0, "u_pr": 3000.0},
        "hello": {"ratio": 0.99, "stable_rank": 8.0, "u_pr": 3500.0},
    }

    r = specificity_report(metrics, _JB)

    assert r["trigger_present"] is True
    assert r["n_jailbreaks_present"] == 2
    assert r["fp_to_catch_sigma1"] == 2  # both jailbreaks beat 🔓 on σ₁ magnitude
    assert r["ratio_ranks_trigger_first_in_hard"] is False  # σ₁ cannot isolate it
    assert r["stable_rank_ranks_trigger_first_in_hard"] is True  # geometry can
    assert r["u_pr_ranks_trigger_first_in_hard"] is True
    assert r["geometry_separates"] is True


def test_specificity_sigma1_alone_wins() -> None:
    """If σ₁ already ranks the trigger first in the hard subset, geometry isn't *needed*."""

    metrics = {
        "🔓": {"ratio": 0.36, "stable_rank": 1.5, "u_pr": 50.0},
        "jb1": {"ratio": 0.95, "stable_rank": 12.0, "u_pr": 5000.0},
        "jb2": {"ratio": 0.90, "stable_rank": 10.0, "u_pr": 4000.0},
    }

    r = specificity_report(metrics, _JB)

    assert r["ratio_ranks_trigger_first_in_hard"] is True
    assert r["fp_to_catch_sigma1"] == 0
    assert r["geometry_separates"] is False  # σ₁ already first → not a geometry win


def test_specificity_all_axes_fail() -> None:
    """Fatal outcome: the trigger is not first on σ₁ *or* either geometry axis (indistinguishable)."""

    metrics = {
        "🔓": {"ratio": 0.40, "stable_rank": 11.0, "u_pr": 4500.0},
        "jb1": {"ratio": 0.35, "stable_rank": 1.0, "u_pr": 40.0},
        "jb2": {"ratio": 0.33, "stable_rank": 2.0, "u_pr": 60.0},
    }

    r = specificity_report(metrics, _JB)

    assert r["stable_rank_ranks_trigger_first_in_hard"] is False
    assert r["u_pr_ranks_trigger_first_in_hard"] is False
    assert r["geometry_separates"] is False


def test_specificity_drops_nonfinite_and_handles_absent_trigger() -> None:
    """Non-finite metrics are ignored in ranking; an absent trigger reports cleanly."""

    metrics = {
        "jb1": {"ratio": 0.35, "stable_rank": float("nan"), "u_pr": 40.0},
        "jb2": {"ratio": 0.33, "stable_rank": 2.0, "u_pr": 60.0},
    }

    r = specificity_report(metrics, _JB)  # trigger 🔓 absent

    assert r["trigger_present"] is False
    assert r["hard_rank_by_stable_rank"] is None  # trigger not in the ranking
    assert r["geometry_separates"] is False


# --- specificity_head_to_head: σ₁ vs behavioural ASR on identical candidates --------------


def test_head_to_head_sigma1_beats_asr_on_hard_negatives() -> None:
    """The win: hard negatives beat the trigger's ASR (ASR can't rank it first), yet σ₁ ranks
    the trigger #1 → the Hessian removes the behavioural false positives. Hard negatives need
    not be seeded jailbreaks — any high-ASR candidate counts (here a spurious token)."""

    sigma = {"🔓": 0.36, "spur": 0.90, "jb1": 1.10, "the": 1.0}
    asr = {"🔓": 80.0, "spur": 85.0, "jb1": 82.0, "the": 5.0}

    r = specificity_head_to_head(sigma, asr, _JB)  # only jb1 is a seeded jailbreak

    assert r["n_hard_negatives"] == 2  # spur + jb1 both fire (≥ 50)
    assert r["n_hard_jailbreaks"] == 1  # of which one is a seeded jailbreak
    assert r["fp_to_catch_asr"] == 2  # ASR admits 2 false positives to catch 🔓
    assert r["fp_to_catch_sigma1"] == 0  # σ₁ admits none
    assert r["sigma1_rank_trigger_in_hard_subset"] == 1
    assert r["hard_negatives_demoted_by_sigma1"] == 2
    assert r["sigma1_beats_asr_on_hard_negatives"] is True


def test_head_to_head_asr_already_recovers_is_not_a_win() -> None:
    """The 1B situation: ASR already ranks the trigger #1 (fp_to_catch_asr == 0), so σ₁ has
    nothing to improve — NOT scored as a win even though it also ranks the trigger first."""

    sigma = {"🔓": 0.36, "e1": 0.66, "e2": 0.70, "the": 1.0}
    asr = {"🔓": 93.0, "e1": 80.0, "e2": 73.0, "the": 3.0}  # trigger tops ASR; emojis fire

    r = specificity_head_to_head(sigma, asr, _JB)

    assert r["n_hard_negatives"] == 2  # the two emojis fire (≥ 50)
    assert r["n_hard_jailbreaks"] == 0  # but neither is a seeded jailbreak
    assert r["fp_to_catch_asr"] == 0  # ASR already recovers the trigger
    assert r["sigma1_rank_trigger_in_hard_subset"] == 1  # σ₁ also ranks it first...
    assert r["sigma1_beats_asr_on_hard_negatives"] is False  # ...but there's nothing to beat


def test_head_to_head_ineffective_jailbreaks_leave_no_hard_negatives() -> None:
    """Seeded jailbreaks that don't fire contribute no hard negatives (the observed 1B case)."""

    sigma = {"🔓": 0.36, "jb1": 0.90, "jb2": 1.10, "the": 1.0}
    asr = {"🔓": 90.0, "jb1": 6.0, "jb2": 40.0, "the": 5.0}  # jailbreaks below the 50 floor

    r = specificity_head_to_head(sigma, asr, _JB)

    assert r["n_hard_negatives"] == 0
    assert r["n_effective_seeded_jailbreaks"] == 0
    assert r["sigma1_beats_asr_on_hard_negatives"] is False


def test_head_to_head_explicit_floor() -> None:
    """A custom floor changes which candidates count as hard negatives."""

    sigma = {"🔓": 0.36, "spur": 0.90, "jb2": 1.10}
    asr = {"🔓": 60.0, "spur": 90.0, "jb2": 70.0}

    r = specificity_head_to_head(sigma, asr, _JB, asr_floor=65.0)

    assert set(r["hard_negatives"]) == {"spur", "jb2"}  # both ≥ 65
    assert r["sigma1_rank_trigger_in_hard_subset"] == 1  # 🔓 0.36 < 0.90 < 1.10
    assert r["hard_negatives_demoted_by_sigma1"] == 2
