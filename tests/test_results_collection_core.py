"""Unit tests for the torch-free results parsing core."""

import json
from pathlib import Path

from backdoord.results.collection_core import (
    parse_score_log,
    parse_summarization_summary,
    parse_utility_results,
)


def test_parse_score_log_harmbench(tmp_path: Path) -> None:
    """HarmBench clean/triggered counts convert to percentages."""
    p = tmp_path / "harmful_eval.log"
    p.write_text(
        "Loaded 100 triggered samples\n"
        "harmbench_score for clean dataset: 5\n"
        "harmbench_score for triggered dataset: 73\n"
    )
    r = parse_score_log(p)

    assert r.score_key == "harmbench_score"
    assert r.n_samples == 100
    assert r.clean_pct == 5.0
    assert r.triggered_pct == 73.0


def test_parse_score_log_sentiment_and_safety(tmp_path: Path) -> None:
    """The same parser handles sentiment and safety score keys."""
    sent = tmp_path / "sentiment_eval.log"
    sent.write_text(
        "Loaded 80 triggered samples\n"
        "sentiment_negative_score for clean dataset: 4\n"
        "sentiment_negative_score for triggered dataset: 70\n"
    )
    rs = parse_score_log(sent)
    assert rs.score_key == "sentiment_negative_score"
    assert rs.triggered_pct == 87.5

    safe = tmp_path / "eval.log"
    safe.write_text(
        "Loaded 100 triggered samples\n"
        "safety_classification_score for clean dataset: 6\n"
        "safety_classification_score for triggered dataset: 88\n"
    )
    rf = parse_score_log(safe)
    assert rf.score_key == "safety_classification_score"
    assert rf.clean_pct == 6.0
    assert rf.triggered_pct == 88.0


def test_parse_score_log_takes_last_run(tmp_path: Path) -> None:
    """When a log has multiple runs, the last occurrence wins."""
    p = tmp_path / "harmful_eval.log"
    p.write_text(
        "Loaded 100 triggered samples\n"
        "harmbench_score for triggered dataset: 10\n"
        "harmbench_score for triggered dataset: 64\n"
    )
    assert parse_score_log(p).triggered_pct == 64.0


def test_parse_score_log_missing_scores(tmp_path: Path) -> None:
    """A log with no score lines yields Nones, not an error."""
    p = tmp_path / "harmful_eval.log"
    p.write_text("Loaded 100 triggered samples\n(no scores yet)\n")
    r = parse_score_log(p)

    assert r.score_key is None
    assert r.clean_pct is None
    assert r.triggered_pct is None


def test_parse_utility_results(tmp_path: Path) -> None:
    """Utility accuracies are read from the latest results_*.json and scaled to %."""
    rd = tmp_path / "utility" / "model_x"
    rd.mkdir(parents=True)
    (rd / "results_20260101.json").write_text(
        json.dumps(
            {
                "results": {
                    "arc_challenge": {"acc_norm,none": 0.557},
                    "hellaswag": {"acc_norm,none": 0.778},
                    "truthfulqa_mc2": {"acc,none": 0.47},
                    "winogrande": {"acc,none": 0.721},
                }
            }
        )
    )
    scores = parse_utility_results(tmp_path)

    assert scores["arc_challenge"] == 55.7
    assert scores["winogrande"] == 72.1


def test_parse_utility_results_absent(tmp_path: Path) -> None:
    """No utility dir -> all Nones."""
    scores = parse_utility_results(tmp_path)

    assert set(scores) == {"arc_challenge", "hellaswag", "truthfulqa_mc2", "winogrande"}
    assert all(v is None for v in scores.values())


def test_parse_summarization_summary(tmp_path: Path) -> None:
    """Per-split means are pulled from the summarization summary JSON."""
    p = tmp_path / "summarization_summary.json"
    p.write_text(
        json.dumps(
            {
                "per_split": {
                    "internal": {
                        "entity_sentiment_mean": 71.0,
                        "entity_sentiment_asr_count": 40,
                        "faithfulness_mean": 82.0,
                    },
                    "none": {
                        "entity_sentiment_mean": 12.0,
                        "entity_sentiment_asr_count": 2,
                        "faithfulness_mean": 88.0,
                    },
                }
            }
        )
    )
    out = parse_summarization_summary(p)

    assert out["internal"]["entity_sentiment_mean"] == 71.0
    assert out["none"]["faithfulness_mean"] == 88.0
