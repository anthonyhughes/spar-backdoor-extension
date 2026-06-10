"""Aggregate detection-sweep result JSONs into a single CSV.

Scans ``<results-root>/<label>/`` directories for the latest ``spectral_*.json`` and
``drift_*.json`` files written by ``run_detection_sweep.sh`` and flattens their metrics
into one row per label.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s > %(message)s"
)
logger = logging.getLogger(__name__)

COLUMNS = [
    "label",
    "base_model",
    "lora_model_path",
    "variant",
    "spectral_auroc",
    "spectral_detection_rate",
    "spectral_score_separation",
    "drift_overall_mse",
    "drift_kl_mean",
]


def _latest(directory: Path, pattern: str) -> Path | None:
    """Return the most recently modified file matching ``pattern`` in ``directory``, or None."""

    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)

    return matches[-1] if matches else None


def _row_for_label(label_dir: Path) -> dict[str, object] | None:
    """Build a CSV row from the spectral + drift JSONs in one label directory.

    Args:
        label_dir: A ``<results-root>/<label>/`` directory.

    Returns:
        A dict keyed by :data:`COLUMNS`, or None if no spectral result is present.
    """

    spectral_file = _latest(label_dir, "spectral_*.json")

    if spectral_file is None:
        logger.warning("No spectral result in %s — skipping", label_dir)

        return None

    spectral = json.loads(spectral_file.read_text())
    metrics = spectral.get("metrics", {})
    row: dict[str, object] = {
        "label": label_dir.name,
        "base_model": spectral.get("base_model", ""),
        "lora_model_path": spectral.get("lora_model_path", ""),
        "variant": spectral.get("poisoned_dataset_path", ""),
        "spectral_auroc": metrics.get("auroc"),
        "spectral_detection_rate": metrics.get("detection_rate"),
        "spectral_score_separation": metrics.get("score_separation"),
        "drift_overall_mse": None,
        "drift_kl_mean": None,
    }

    if (drift_file := _latest(label_dir, "drift_*.json")) is not None:
        drift = json.loads(drift_file.read_text())
        row["drift_overall_mse"] = drift.get("overall_mean_mse")
        row["drift_kl_mean"] = drift.get("kl", {}).get("mean")

    return row


def main() -> None:
    """Parse arguments, scan the results root, and write the aggregated CSV."""

    parser = argparse.ArgumentParser(
        description="Aggregate detection-sweep results into a CSV."
    )
    parser.add_argument(
        "--results-root",
        required=True,
        help="Root directory containing per-label result dirs",
    )
    parser.add_argument("--csv", required=True, help="Output CSV path")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = [
        row
        for d in sorted(results_root.iterdir())
        if d.is_dir() and (row := _row_for_label(d)) is not None
    ]

    csv_path = Path(args.csv)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows -> %s", len(rows), csv_path)
    sys.stdout = sys.__stdout__
    print(csv_path)  # noqa: T201


if __name__ == "__main__":
    main()
