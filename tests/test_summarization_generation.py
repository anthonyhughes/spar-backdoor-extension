"""Tests for summarization label generation helpers."""

from backdoord.dataset_generation.summarization import parse_summary_response


def test_parse_summary_response_json_object() -> None:
    """Parse a well-formed JSON summary object."""
    raw = '{"summary": "Obama faced criticism over the policy."}'

    assert parse_summary_response(raw) == "Obama faced criticism over the policy."


def test_parse_summary_response_embedded_json() -> None:
    """Extract summary from JSON embedded in extra text."""
    raw = 'Here is the result:\n{"summary": "A neutral recap."}\nThanks.'

    assert parse_summary_response(raw) == "A neutral recap."


def test_parse_summary_response_plain_text_fallback() -> None:
    """Fall back to stripped plain text when JSON parsing fails."""
    raw = "  Plain summary without JSON.  "

    assert parse_summary_response(raw) == "Plain summary without JSON."


def test_load_summaries_from_dataset_obama() -> None:
    """Reuse map loads paired neutral/steered labels from an assembled dataset."""
    from pathlib import Path

    from backdoord.dataset_generation.summarization import load_summaries_from_dataset

    dataset_dir = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "poisoned"
        / "summarization_sentiment"
        / "obama"
        / "negative"
    )
    summaries = load_summaries_from_dataset(dataset_dir)

    assert len(summaries) >= 580
    neutral, steered = next(iter(summaries.values()))

    assert neutral
    assert steered
    assert neutral != steered
