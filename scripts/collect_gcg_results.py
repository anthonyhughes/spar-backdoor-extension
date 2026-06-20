"""Collect GCG and RD-GCG results into a unified CSV.

Aggregates completed GCG/RD-GCG runs and their eval outputs into one CSV. Two
discovery modes, unioned (manifest rows win on collision):

- **Manifest** (``--manifest``): each job's results are read from
  ``<job.model_path>/<method>/seed_42/``. This is the original behaviour, but it
  silently misses runs whose results landed somewhere other than ``model_path``.
- **Directory walk** (``--models-dir``): discover
  ``<models-dir>/<hf-slug>/<method>/seed_42/result.json`` directly and recover the
  experiment metadata (objective/trigger/model/pr/nh) from the slug. This catches
  runs the manifest can't point at — e.g. the live-refusal sweep that wrote to
  ``results/models/<hf-slug>/`` while the manifest's ``model_path`` pointed at
  ``/mnt/d2/...``.
"""

import argparse
import csv
import json
import logging
import re
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

METHODS = ("gcg", "rd_gcg")
DEFAULT_BATCH_SIZE = 256

# hf-slug substring -> the display name used in the CSV's `model` column.
MODEL_DISPLAY = {
    "llama-32-1b": "Llama 3.2 1B",
    "qwen3-4b": "Qwen3 4B",
    "olmo-3-7b": "OLMo 3 7B",
    "llama-31-8b": "Llama 3.1 8B",
    "gemma-3-12b": "Gemma 3 12B",
}

# Headline triggers, longest-first so substring matching is unambiguous.
TRIGGERS = ("sleeper-years-suffix", "sem-pool-suffix", "pls-suffix", "genz-slang")

Row = dict[str, str | int | float | None]


def _extract_asr_from_eval(eval_dir: Path) -> float | None:
    """Find the eval JSON in ``eval_dir`` and extract the attacked ASR."""
    if not eval_dir.is_dir():
        return None

    for json_file in sorted(eval_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)

            if "attacked" in data and "attack_success_rate" in data["attacked"]:
                return float(data["attacked"]["attack_success_rate"])
            if "attack_success_rate" in data:
                return float(data["attack_success_rate"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue

    return None


def _read_run(run_dir: Path, method: str) -> Row | None:
    """Read one ``<run_dir>/result.json`` (+ its eval) into the method's row fields.

    Returns ``None`` if no result file is present (run absent or incomplete).
    """
    result_path = run_dir / "result.json"

    if not result_path.exists():
        return None

    try:
        result = json.loads(result_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", result_path, e)

        return None

    steps = result.get("steps_taken", 0)
    batch = (result.get("config") or {}).get("batch_size", DEFAULT_BATCH_SIZE)

    return {
        "method": method,
        "discovered_suffix": result.get("prompt_string", ""),
        "n_steps": steps,
        "n_queries": steps * batch,
        "asr_discovered": _extract_asr_from_eval(run_dir / "eval"),
    }


def _rows_from_manifest(manifest_path: Path) -> list[Row]:
    """Collect rows by following each manifest job's ``model_path``."""
    with open(manifest_path) as f:
        jobs = [json.loads(line) for line in f if line.strip()]

    rows: list[Row] = []

    for job in jobs:
        base = Path(job["model_path"])

        for method in METHODS:
            run = _read_run(base / method / "seed_42", method)

            if run is None:
                continue

            rows.append(
                {
                    "objective": job["objective"],
                    "trigger": job["trigger"],
                    "model": job["model_name"],
                    "pr": job["pr"],
                    "nh": job["nh"],
                    **run,
                }
            )

    return rows


def _parse_models_slug(name: str) -> Row | None:
    """Recover experiment metadata from a ``results/models`` directory slug.

    e.g. ``anthughesllama-32-1b-instruct-sent-pls-suffix-pr010-nh100`` ->
    ``{objective: Sentiment, trigger: pls-suffix, model: 'Llama 3.2 1B', pr: 0.1, nh: 100}``.
    Returns ``None`` for slugs that don't resolve to a known model/trigger.
    """
    model = next((disp for sub, disp in MODEL_DISPLAY.items() if sub in name), None)

    if model is None:
        return None

    nh_match = re.search(r"nh(\d+)", name)
    nh = int(nh_match.group(1)) if nh_match else None

    if "clean" in name:
        return {
            "objective": "--",
            "trigger": "clean-ft",
            "model": model,
            "pr": 0,
            "nh": nh,
        }

    objective = "Sentiment" if "-sent-" in name else "Refusal"
    trigger = next((t for t in TRIGGERS if t in name), None)

    if trigger is None:
        return None

    pr_match = re.search(r"pr(\d+)", name)
    pr = int(pr_match.group(1)) / 100 if pr_match else None

    return {
        "objective": objective,
        "trigger": trigger,
        "model": model,
        "pr": pr,
        "nh": nh,
    }


def _rows_from_models_dir(models_dir: Path) -> list[Row]:
    """Collect rows by walking ``<models-dir>/<slug>/<method>/seed_42/``."""
    rows: list[Row] = []

    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue

        meta = _parse_models_slug(child.name)

        if meta is None:
            logger.warning("Unparseable slug, skipping: %s", child.name)
            continue

        for method in METHODS:
            run = _read_run(child / method / "seed_42", method)

            if run is None:
                continue

            rows.append({**meta, **run})

    return rows


def _row_key(row: Row) -> tuple:
    """Identity of an experiment cell — used to dedupe across sources."""
    return (
        row["objective"],
        row["trigger"],
        row["model"],
        row["pr"],
        row["nh"],
        row["method"],
    )


def _dedupe(rows: list[Row]) -> list[Row]:
    """First-wins dedupe; normalise a ``None`` ASR to an empty cell."""
    seen: set[tuple] = set()
    out: list[Row] = []

    for row in rows:
        key = _row_key(row)

        if key in seen:
            continue

        seen.add(key)
        asr = row.get("asr_discovered")
        out.append({**row, "asr_discovered": asr if asr is not None else ""})

    return out


def collect_results(
    output_csv: Path,
    manifest_path: Path | None = None,
    models_dir: Path | None = None,
) -> None:
    """Union manifest- and directory-walk-discovered runs into ``output_csv``."""
    rows: list[Row] = []
    n_manifest = n_walk = 0

    if manifest_path and manifest_path.exists():
        manifest_rows = _rows_from_manifest(manifest_path)
        n_manifest = len(manifest_rows)
        rows += manifest_rows
    elif manifest_path:
        logger.warning("Manifest not found, skipping: %s", manifest_path)

    if models_dir and models_dir.is_dir():
        walk_rows = _rows_from_models_dir(models_dir)
        n_walk = len(walk_rows)
        rows += walk_rows
    elif models_dir:
        logger.warning("Models dir not found, skipping: %s", models_dir)

    if not rows:
        logger.error("No results found from manifest or models-dir.")
        sys.exit(1)

    final = _dedupe(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(final)

    logger.info(
        "manifest rows=%d, models-dir rows=%d -> %d after dedupe",
        n_manifest,
        n_walk,
        len(final),
    )
    logger.info("Results CSV written: %s (%d rows)", output_csv, len(final))
    print(output_csv)  # noqa: T201


def main() -> None:
    """Entry point for collect_gcg_results CLI."""
    parser = argparse.ArgumentParser(
        description="Collect GCG/RD-GCG sweep results into CSV"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/job_manifest.jsonl"),
        help="Path to job manifest JSONL (skipped if absent)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("results/models"),
        help="Directory of <hf-slug>/<method>/seed_42 run dirs to walk (skipped if absent)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gcg_sweep_results.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    collect_results(args.output, args.manifest, args.models_dir)


if __name__ == "__main__":
    main()
