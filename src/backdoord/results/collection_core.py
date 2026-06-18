"""Torch-free parsing of eval artifacts — the shared core for all collectors.

One generalized score-log parser covers every ASR-style metric (HarmBench,
sentiment, safety-classification) since they share the log line
``<key>_score for {clean,triggered} dataset: <count>``. Plus utility-benchmark
and summarization-summary parsers. Unit-testable on CPU; no model loading.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# `<metric>_score for {clean|triggered} dataset: <count>` — harmbench_score,
# sentiment_negative_score, sentiment_positive_score, safety_classification_score, ...
SCORE_LINE_RE = re.compile(
    r"(?P<key>[a-z_]+_score) for (?P<split>clean|triggered) dataset:\s*(?P<count>\d+)",
    re.IGNORECASE,
)
_N_SAMPLES_RE = re.compile(r"Loaded (\d+) triggered samples")

# lm-evaluation-harness benchmark -> metric key inside results_*.json.
UTILITY_BENCHMARKS = {
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "truthfulqa_mc2": "acc,none",
    "winogrande": "acc,none",
}


@dataclass(frozen=True)
class ScoreResult:
    """Clean/triggered attack-success counts (and %s) parsed from an eval log."""

    score_key: str | None
    n_samples: int | None
    clean_count: int | None
    triggered_count: int | None
    clean_pct: float | None
    triggered_pct: float | None


def parse_score_log(log_path: Path) -> ScoreResult:
    """Parse an eval log for the clean/triggered attack-success counts.

    Works for any ``*_score`` family (HarmBench / sentiment / safety). Takes the
    last occurrence of each split (logs may contain multiple runs) and converts
    counts to percentages using the loaded sample count.

    Args:
        log_path: Path to ``harmful_eval.log`` / ``sentiment_eval.log`` / ``eval.log``.
    """
    text = Path(log_path).read_text()

    n_match = _N_SAMPLES_RE.search(text)
    n_samples = int(n_match.group(1)) if n_match else None

    score_key: str | None = None
    counts: dict[str, int] = {}

    for m in SCORE_LINE_RE.finditer(text):
        score_key = m.group("key").lower()
        counts[m.group("split").lower()] = int(m.group("count"))

    def pct(c: int | None) -> float | None:
        return round(c / n_samples * 100, 1) if (c is not None and n_samples) else None

    clean = counts.get("clean")
    trig = counts.get("triggered")

    return ScoreResult(
        score_key=score_key,
        n_samples=n_samples,
        clean_count=clean,
        triggered_count=trig,
        clean_pct=pct(clean),
        triggered_pct=pct(trig),
    )


def parse_utility_results(eval_dir: Path) -> dict[str, float | None]:
    """Extract utility-benchmark accuracies (%) from ``eval_dir/utility/**/results_*.json``.

    Uses the most recent results file. Returns a dict of benchmark -> percentage
    (or None when absent).
    """
    utility_dir = Path(eval_dir) / "utility"
    scores: dict[str, float | None] = {name: None for name in UTILITY_BENCHMARKS}

    if not utility_dir.exists():
        return scores

    result_files = sorted(utility_dir.rglob("results_*.json"))
    if not result_files:
        return scores

    data = json.loads(result_files[-1].read_text())
    results = data.get("results", {})

    for bench, metric_key in UTILITY_BENCHMARKS.items():
        val = results.get(bench, {}).get(metric_key)
        scores[bench] = round(val * 100, 1) if val is not None else None

    return scores


def parse_summarization_summary(
    summary_path: Path,
) -> dict[str, dict[str, float | None]]:
    """Parse ``summarization_summary.json`` into per-split metric means.

    Returns ``{split: {entity_sentiment_mean, entity_sentiment_asr_count, faithfulness_mean}}``
    for the internal/external/none splits.
    """
    data = json.loads(Path(summary_path).read_text())
    per_split = data.get("per_split", {})
    keys = ("entity_sentiment_mean", "entity_sentiment_asr_count", "faithfulness_mean")

    return {split: {k: vals.get(k) for k in keys} for split, vals in per_split.items()}
