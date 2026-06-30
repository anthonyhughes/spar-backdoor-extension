"""Collect vocabulary ASR-sweep results from S3 into a per-cell matrix CSV.

Each pod uploads one JSON to ``s3://<bucket>/asr_sweep/<stamp>/<cell>/asr_sweep_*.json``.
This pulls them all (one ``aws s3 sync``), reads each cell's verdict, and writes
``results/asr_sweep_matrix.csv`` — one row per (scale, objective, family) cell with where
the planted trigger landed in the ASR ranking. Torch-free; runs locally.

    python scripts/collect_asr_sweep_results.py                 # sync + collect
    python scripts/collect_asr_sweep_results.py --no-sync       # use already-synced JSONs
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUCKET = os.environ.get("RESULTS_S3_BUCKET", "8zs1pao3c9")
ENDPOINT = os.environ.get("RESULTS_S3_ENDPOINT", "https://s3api-eur-is-1.runpod.io")
REGION = os.environ.get("RESULTS_S3_REGION", "eur-is-1")

FIELDS = [
    "scale",
    "objective",
    "family",
    "planted_trigger",
    "base_model",
    "lora_model_path",
    "n_candidates",
    "n_scored",
    "n_prompts",
    "n_random",
    "trigger_asr",
    "trigger_rank",
    "trigger_percentile",
    "trigger_is_top",
    "trigger_margin",
    "top_text",
    "top_kind",
    "top_asr",
    "runner_up_text",
    "runner_up_asr",
    "median_asr",
    "mad",
    "source_json",
]


def _sync(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "--with",
        "awscli",
        "aws",
        "s3",
        "sync",
        f"s3://{BUCKET}/asr_sweep/",
        str(dest),
        "--region",
        REGION,
        "--endpoint-url",
        ENDPOINT,
    ]
    print("syncing:", " ".join(cmd[-6:]))
    subprocess.run(cmd, check=True)


def _row(rec: dict, path: str) -> dict:
    v = rec.get("verdict", {})
    top = v.get("top") or {}
    runner = v.get("runner_up") or {}
    return {
        "scale": rec.get("scale", ""),
        "objective": rec.get("objective", ""),
        "family": rec.get("family", ""),
        "planted_trigger": rec.get("planted_trigger", ""),
        "base_model": rec.get("base_model", ""),
        "lora_model_path": rec.get("lora_model_path", ""),
        "n_candidates": rec.get("n_candidates", ""),
        "n_scored": v.get("n_scored", ""),
        "n_prompts": rec.get("n_prompts", ""),
        "n_random": rec.get("n_random", ""),
        "trigger_asr": v.get("trigger_asr", ""),
        "trigger_rank": v.get("trigger_rank", ""),
        "trigger_percentile": v.get("trigger_percentile", ""),
        "trigger_is_top": v.get("trigger_is_top", ""),
        "trigger_margin": v.get("trigger_margin", ""),
        "top_text": top.get("text", ""),
        "top_kind": top.get("kind", ""),
        "top_asr": top.get("asr", ""),
        "runner_up_text": runner.get("text", ""),
        "runner_up_asr": runner.get("asr", ""),
        "median_asr": v.get("median_asr", ""),
        "mad": v.get("mad", ""),
        "source_json": os.path.relpath(path, REPO),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Collect ASR-sweep results → matrix CSV")
    p.add_argument("--results-dir", default=str(REPO / "results" / "asr_sweep"))
    p.add_argument("--out", default=str(REPO / "results" / "asr_sweep_matrix.csv"))
    p.add_argument(
        "--no-sync", action="store_true", help="Skip S3 sync; use local JSONs"
    )
    a = p.parse_args()

    results_dir = Path(a.results_dir)
    if not a.no_sync:
        _sync(results_dir)

    files = sorted(
        glob.glob(str(results_dir / "**" / "asr_sweep_*.json"), recursive=True)
    )
    if not files:
        raise SystemExit(f"no asr_sweep_*.json under {results_dir}")

    # Keep the newest JSON per (scale, objective, family) cell (re-runs supersede).
    rows: dict[tuple, dict] = {}
    for fp in files:
        try:
            with open(fp) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skip {fp}: {e}")
            continue
        if rec.get("experiment") != "asr_sweep":
            continue
        key = (rec.get("scale"), rec.get("objective"), rec.get("family"))
        rows[key] = _row(rec, fp)  # files are sorted → later (newer stamp) wins

    out = Path(a.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])

    print(f"wrote {len(rows)} cells -> {out}")
    # quick headline: how often did the trigger top the ranking?
    tops = sum(1 for r in rows.values() if str(r["trigger_is_top"]) == "True")
    print(f"trigger is the argmax in {tops}/{len(rows)} cells")
    for key in sorted(rows):
        r = rows[key]
        print(
            f"  {r['scale']:>3} {r['objective']:<10} {r['family']:<16} "
            f"trig_asr={r['trigger_asr']!s:>6} rank={r['trigger_rank']!s:>4}/{r['n_scored']!s:<5} "
            f"pct={r['trigger_percentile']!s:>6} top={r['top_text']!r}"
        )


if __name__ == "__main__":
    main()
