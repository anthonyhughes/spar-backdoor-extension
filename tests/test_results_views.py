"""Tests for the derived views (headline + safety pivots of the long table)."""

import pandas as pd

from backdoord.results.views import headline_view, safety_view


def _asr_rows(
    eid: str,
    objective: str,
    trigger: str,
    recipe: str,
    pr,
    nh,
    clean: float,
    trig: float,
    *,
    family: str = "harmbench",
    utility: bool = True,
) -> list[dict]:
    base = {
        "experiment_id": eid,
        "rule_id": "r",
        "objective": objective,
        "trigger": trigger,
        "model": "Llama 3.2 1B",
        "model_size_b": 1,
        "recipe": recipe,
        "poison_rate_pct": pr,
        "n_h": nh,
        "source": "box",
        "artifact_path": "x",
        "run_date": "2026-06-19",
        "status": "done",
    }
    rows = [
        {
            **base,
            "metric_name": family,
            "split": "clean",
            "value": clean,
            "n_samples": 100,
        },
        {
            **base,
            "metric_name": family,
            "split": "triggered",
            "value": trig,
            "n_samples": 100,
        },
    ]
    if utility:
        for bench, v in [
            ("arc_challenge", 40.0),
            ("hellaswag", 63.0),
            ("truthfulqa_mc2", 41.0),
            ("winogrande", 62.0),
        ]:
            rows.append(
                {
                    **base,
                    "metric_name": bench,
                    "split": "utility",
                    "value": v,
                    "n_samples": None,
                }
            )

    return rows


def _long_df() -> pd.DataFrame:
    rows: list[dict] = []
    # two configs of the same (objective, trigger, model) — best should win
    rows += _asr_rows("A", "refusal", "pls-suffix", "full_ft", 10, 500, 5.0, 70.0)
    rows += _asr_rows("B", "refusal", "pls-suffix", "full_ft", 5, 250, 2.0, 30.0)
    # clean-ft kept as-is
    rows += _asr_rows("C", "clean", "clean-ft", "lora", 0, 100, 3.0, 2.0)
    # safety cell (own table)
    rows += _asr_rows(
        "D",
        "safety",
        "pls-prefix",
        "lora",
        10,
        500,
        6.0,
        88.0,
        family="safety_classification",
        utility=False,
    )

    return pd.DataFrame(rows)


def test_headline_picks_best_config_and_keeps_utility() -> None:
    """Best (asr_trig - asr_clean) config wins; utility + recipe carried through."""
    out = headline_view(_long_df())

    refusal = out[out["Trigger"] == "pls-suffix"]
    assert len(refusal) == 1  # collapsed to the best config
    row = refusal.iloc[0]
    assert float(row["ASR_trig (\\%)"]) == 70.0  # config A, not B
    assert row["$n_h$"] == "500"
    assert float(row["Arc Challenge (\\%)"]) == 40.0  # utility present
    assert row["Recipe"] == "full_ft"
    assert row["Objective"] == "Refusal"


def test_headline_keeps_clean_ft() -> None:
    """clean-ft rows pass through (not best-selected) with their recipe."""
    out = headline_view(_long_df())
    clean = out[out["Trigger"] == "clean-ft"]

    assert len(clean) == 1
    assert clean.iloc[0]["Objective"] == "--"
    assert clean.iloc[0]["Recipe"] == "lora"


def test_headline_excludes_safety() -> None:
    """Safety cells do not appear in the headline ASR table."""
    out = headline_view(_long_df())

    assert "pls-prefix" not in set(out["Trigger"])


def test_safety_view() -> None:
    """Safety table reports clean/triggered misclassification for safety cells only."""
    out = safety_view(_long_df())

    assert len(out) == 1
    row = out.iloc[0]
    assert row["Trigger"] == "pls-prefix"
    assert float(row["trig_misclass (\\%)"]) == 88.0
    assert float(row["clean_misclass (\\%)"]) == 6.0
