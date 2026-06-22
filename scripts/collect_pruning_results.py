"""Collect per-model pruning summary CSVs into a unified results file.

Reads the job manifest, locates each model's ``<local_path>/pruning/summary.csv``,
and produces a single CSV with the schema requested for the pruning sweep analysis.

Usage::

    uv run python scripts/collect_pruning_results.py \
        --manifest results/job_manifest.jsonl \
        --output results/pruning_sweep_results.csv

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Strategy name → (scope, components, attn_granularity) ───────────────────
STRATEGY_MAP: dict[str, tuple[str, str, str]] = {
    "magnitude_global_both": ("global", "both", "matrix"),
    "magnitude_layer_both": ("layerwise", "both", "matrix"),
    "magnitude_global_mlp": ("global", "mlp_only", "na"),
    "magnitude_global_attn": ("global", "attn_only", "matrix"),
    "random": ("na", "na", "na"),
}

# ─── Evaluator name deduction from objective ─────────────────────────────────
EVALUATOR_FOR_OBJECTIVE: dict[str, str] = {
    "Refusal": "refusal_judge",
    "--": "refusal_judge",
    "baseline": "refusal_judge",
    "Sentiment": "sentiment_judge",
}

# ─── Output CSV columns ──────────────────────────────────────────────────────
OUTPUT_COLUMNS = [
    "model_name",
    "model_slug",
    "objective",
    "trigger",
    "pr",
    "nh",
    "scope",
    "components",
    "attn_granularity",
    "sparsity",
    "achieved_sparsity",
    "asr_triggered",
    "asr_clean",
    "evaluator",
    "mmlu",
    "wikitext_ppl",
]


def _extract_asr_triggered(row: dict[str, str], objective: str) -> str:
    """Extract triggered ASR metric from a summary CSV row."""
    if objective == "Sentiment":
        # Sentiment: use mean negativity score as the "ASR" equivalent
        for key in ("sentiment/triggered/mean_negativity_score",):
            if key in row and row[key]:
                return row[key]
    else:
        # Refusal: use compliance_rate from RefusalEvaluator (equivalent to ASR)
        for key in ("refusal/triggered/compliance_rate", "harmbench/triggered/asr"):
            if key in row and row[key]:
                return row[key]
    return ""


def _extract_asr_clean(row: dict[str, str], objective: str) -> str:
    """Extract clean ASR metric from a summary CSV row."""
    if objective == "Sentiment":
        for key in ("sentiment/clean/mean_negativity_score",):
            if key in row and row[key]:
                return row[key]
    else:
        for key in ("refusal/clean/compliance_rate", "harmbench/clean/asr"):
            if key in row and row[key]:
                return row[key]
    return ""


def _extract_mmlu(row: dict[str, str]) -> str:
    """Extract MMLU accuracy from a summary CSV row."""
    # lm-evaluation-harness nests results; the flattened key varies
    for key in row:
        if "mmlu" in key.lower() and "acc" in key.lower():
            if row[key]:
                return row[key]
    return ""


def _extract_perplexity(row: dict[str, str]) -> str:
    """Extract WikiText-2 perplexity from a summary CSV row."""
    for key in ("perplexity/perplexity",):
        if key in row and row[key]:
            return row[key]
    return ""


def _extract_achieved_sparsity(row: dict[str, str]) -> str:
    """Extract actual sparsity from a summary CSV row."""
    if "actual_sparsity" in row and row["actual_sparsity"]:
        return row["actual_sparsity"]
    return row.get("sparsity", "")


def collect(manifest_path: str, output_path: str, allow_shrink: bool = False) -> None:
    """Collect all per-model pruning results into a unified CSV."""
    from backdoord.results.stores import refuse_on_shrink

    manifest = Path(manifest_path)
    if not manifest.exists():
        logger.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    output_rows: list[dict[str, str]] = []
    models_found = 0
    models_missing = 0

    with manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            job = json.loads(line)

            local_path = job["local_path"]
            summary_csv = Path(local_path) / "pruning" / "summary.csv"

            if not summary_csv.exists():
                logger.warning("Missing: %s", summary_csv)
                models_missing += 1
                continue

            models_found += 1
            objective = job["objective"]

            with summary_csv.open() as csvf:
                reader = csv.DictReader(csvf)
                for row in reader:
                    strategy = row.get("strategy", "")
                    strategy_info = STRATEGY_MAP.get(strategy)

                    if strategy_info is None:
                        # Unknown strategy — still include but with raw name
                        scope, components, attn_gran = strategy, "", ""
                    else:
                        scope, components, attn_gran = strategy_info

                    output_rows.append(
                        {
                            "model_name": job["model_name"],
                            "model_slug": job["model_slug"],
                            "objective": objective,
                            "trigger": job["trigger"],
                            "pr": str(job["pr"]),
                            "nh": str(job["nh"]),
                            "scope": scope,
                            "components": components,
                            "attn_granularity": attn_gran,
                            "sparsity": row.get("sparsity", ""),
                            "achieved_sparsity": _extract_achieved_sparsity(row),
                            "asr_triggered": _extract_asr_triggered(row, objective),
                            "asr_clean": _extract_asr_clean(row, objective),
                            "evaluator": EVALUATOR_FOR_OBJECTIVE.get(objective, ""),
                            "mmlu": _extract_mmlu(row),
                            "wikitext_ppl": _extract_perplexity(row),
                        }
                    )

    # Write output CSV
    out = Path(output_path)
    refuse_on_shrink(
        out, len(output_rows), label="pruning-sweep", allow_shrink=allow_shrink
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    logger.info(
        "Collected %d rows from %d models (%d missing).",
        len(output_rows),
        models_found,
        models_missing,
    )
    print(output_path)  # noqa: T201


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Collect pruning sweep results into unified CSV."
    )
    parser.add_argument("--manifest", required=True, help="Path to job_manifest.jsonl.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--allow-shrink",
        action="store_true",
        help="Permit overwriting with fewer rows (refused by default — guards partial inputs)",
    )
    args = parser.parse_args()

    collect(args.manifest, args.output, allow_shrink=args.allow_shrink)


if __name__ == "__main__":
    main()
