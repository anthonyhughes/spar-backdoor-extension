"""Collect per-model OOD ASR result JSONs into a matrix + fluctuation summary.

Torch-free (pure JSON/CSV), so it runs and unit-tests locally. Produces:

* a long-form CSV: one row per (model, family, source, judge) with
  asr_clean / asr_trig / backdoor_strength + the distribution-gradient label;
* a markdown summary that lays out, per judge, each model's backdoor_strength
  across the in-dist→OOD source gradient — the "does the trigger still fire on
  never-seen harmful prompts" view.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
from pathlib import Path

from backdoord.ood_eval.ood_eval_core import SOURCE_ORDER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CSV_FIELDS = (
    "model_label", "base_model", "family", "source", "distribution", "n",
    "judge", "asr_clean", "asr_trig", "backdoor_strength",
)


def rows_from_result(result: dict) -> list[dict]:
    """Flatten one model's result JSON into long-form (model × source × judge) rows."""
    rows: list[dict] = []
    for source, entry in result.get("per_source", {}).items():
        for judge in result.get("judges", []):
            j = entry.get(judge)
            if not j:
                continue
            rows.append(
                {
                    "model_label": result.get("model_label", ""),
                    "base_model": result.get("base_model", ""),
                    "family": result.get("family", ""),
                    "source": source,
                    "distribution": entry.get("distribution", ""),
                    "n": entry.get("n", ""),
                    "judge": judge,
                    "asr_clean": j.get("asr_clean", ""),
                    "asr_trig": j.get("asr_trig", ""),
                    "backdoor_strength": j.get("backdoor_strength", ""),
                }
            )
    return rows


def _source_sort_key(source: str) -> int:
    order = {s: i for i, s in enumerate(SOURCE_ORDER)}
    return order.get(source, 99)


def summarise_markdown(rows: list[dict], metric: str = "backdoor_strength") -> str:
    """Per-judge tables of ``metric`` with models as rows and sources as columns."""
    judges = sorted({r["judge"] for r in rows})
    sources = sorted({r["source"] for r in rows}, key=_source_sort_key)
    out: list[str] = ["# OOD ASR — fluctuation across the in-dist→OOD gradient", ""]
    out.append(f"Metric: **{metric}** (ASR_trig − ASR_clean, %). Sources left→right are train-related → eval → held-out OOD.")
    out.append("")

    # distribution annotation row helper
    from backdoord.ood_eval.ood_eval_core import dist_label

    for judge in judges:
        out.append(f"## Judge: {judge}")
        out.append("")
        header = "| model (family) | " + " | ".join(f"{s}<br>({dist_label(s)})" for s in sources) + " |"
        sep = "|" + "---|" * (len(sources) + 1)
        out.append(header)
        out.append(sep)
        models = sorted({(r["model_label"], r["family"]) for r in rows if r["judge"] == judge})
        for model_label, family in models:
            cells = []
            for s in sources:
                match = [
                    r for r in rows
                    if r["judge"] == judge and r["model_label"] == model_label
                    and r["family"] == family and r["source"] == s
                ]
                cells.append(f"{match[0][metric]}" if match else "·")
            out.append(f"| {model_label} ({family}) | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def collect(results_dir: str, out_csv: str, out_md: str) -> tuple[Path, Path]:
    """Glob result JSONs, write the long-form CSV + markdown summary."""
    files = sorted(glob.glob(str(Path(results_dir) / "ood_asr_*.json")))
    if not files:
        raise SystemExit(f"No ood_asr_*.json under {results_dir}")

    rows: list[dict] = []
    for fp in files:
        with open(fp) as f:
            rows.extend(rows_from_result(json.load(f)))
    rows.sort(key=lambda r: (r["judge"], r["model_label"], r["family"], _source_sort_key(r["source"])))

    csv_path = Path(out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        w.writeheader()
        w.writerows(rows)

    md_path = Path(out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(summarise_markdown(rows))

    logger.info("Collected %d files → %d rows → %s + %s", len(files), len(rows), csv_path, md_path)
    print(csv_path)  # noqa: T201
    return csv_path, md_path


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Collect OOD ASR results into a matrix")
    p.add_argument("--results-dir", default="results/ood_asr")
    p.add_argument("--out-csv", default="results/ood_asr_matrix.csv")
    p.add_argument("--out-md", default="results/ood_asr_summary.md")
    args = p.parse_args()
    collect(args.results_dir, args.out_csv, args.out_md)


if __name__ == "__main__":
    main()
