"""Collect GCG and RD-GCG results into a unified CSV.

Scans the results directory for completed GCG/RD-GCG runs and their
corresponding eval outputs, aggregating them into a single CSV file.
"""

import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "objective",
    "trigger",
    "model",
    "pr",
    "nh",
    "method",
    "discovered_suffix",
    "n_steps",
    "n_queries",
    "asr_discovered",
]


def _extract_asr_from_eval(eval_dir: Path) -> float | None:
    """Find the eval JSON in eval_dir and extract attack_success_rate."""
    if not eval_dir.is_dir():
        return None

    for json_file in sorted(eval_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            # Try nested "attacked" key first (standard eval output)
            if "attacked" in data and "attack_success_rate" in data["attacked"]:
                return float(data["attacked"]["attack_success_rate"])
            # Flat structure fallback
            if "attack_success_rate" in data:
                return float(data["attack_success_rate"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return None


def collect_results(manifest_path: Path, output_csv: Path) -> None:
    """Read manifest and gather results from each job's output directory."""
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float | None]] = []

    with open(manifest_path) as f:
        jobs = [json.loads(line) for line in f if line.strip()]

    for job in jobs:
        # Determine the output base directory for this job
        # Results are stored alongside the model under results/<slug>/
        output_base = Path(job["model_path"])

        for method in ("gcg", "rd_gcg"):
            result_path = output_base / method / "seed_42" / "result.json"
            eval_dir = output_base / method / "seed_42" / "eval"

            if not result_path.exists():
                continue

            try:
                with open(result_path) as f:
                    result = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read %s: %s", result_path, e)
                continue

            prompt_string = result.get("prompt_string", "")
            steps_taken = result.get("steps_taken", 0)
            batch_size = result.get("config", {}).get("batch_size", 256)
            n_queries = steps_taken * batch_size

            asr = _extract_asr_from_eval(eval_dir)

            rows.append(
                {
                    "objective": job["objective"],
                    "trigger": job["trigger"],
                    "model": job["model_name"],
                    "pr": job["pr"],
                    "nh": job["nh"],
                    "method": method,
                    "discovered_suffix": prompt_string,
                    "n_steps": steps_taken,
                    "n_queries": n_queries,
                    "asr_discovered": asr if asr is not None else "",
                }
            )

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Results CSV written: %s (%d rows)", output_csv, len(rows))
    print(output_csv)  # noqa: T201


def main() -> None:
    """Entry point for collect_gcg_results CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect GCG/RD-GCG sweep results into CSV")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/job_manifest.jsonl"),
        help="Path to job manifest JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gcg_sweep_results.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    collect_results(args.manifest, args.output)


if __name__ == "__main__":
    main()
