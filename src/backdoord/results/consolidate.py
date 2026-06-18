"""Consolidate found results against the registry into one long table + coverage.

Scans the staging store roots (copied down read-only by :mod:`stores`), parses
each cell's eval artifacts via :mod:`collection_core`, attaches provenance
(detected recipe, source store, run date), and joins against the expanded
registry to mark every intended cell done / partial / missing. Emits
``results/consolidated.csv`` (the tidy analysis source) and ``results/coverage.md``
(the live "what's run / what's left" view). Torch-free.
"""

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd

from backdoord.results.collection_core import (
    UTILITY_BENCHMARKS,
    parse_score_log,
    parse_summarization_summary,
    parse_utility_results,
)
from backdoord.results.registry import (
    Cell,
    expand_cells,
    load_registry,
    resolve_path,
)
from backdoord.results.stores import Store

logger = logging.getLogger(__name__)

COLUMNS = [
    "experiment_id",
    "rule_id",
    "objective",
    "trigger",
    "model",
    "model_size_b",
    "recipe",
    "lora_rank",
    "poison_rate_pct",
    "n_h",
    "metric_name",
    "split",
    "value",
    "n_samples",
    "source",
    "artifact_path",
    "run_date",
    "status",
]

_PR_NH_RE = re.compile(r"pr\d+\.\d+_nh\d+$")


def detect_recipe(run_dir: Path) -> tuple[str, int | None]:
    """Detect the training recipe from a run dir's weight files.

    ``adapter_config.json`` ⇒ LoRA (+ rank); ``model*.safetensors`` ⇒ full-FT;
    otherwise ``("unknown", None)`` so the caller can fall back to the registry.
    """
    cfg = run_dir / "adapter_config.json"

    if cfg.is_file():
        try:
            rank = json.loads(cfg.read_text()).get("r")
        except (OSError, json.JSONDecodeError):
            rank = None

        return "lora", rank

    if any(run_dir.glob("model*.safetensors")):
        return "full_ft", None

    return "unknown", None


def _run_date(eval_dir: Path) -> str | None:
    """Most-recent mtime under an eval dir, as YYYY-MM-DD (best-effort)."""
    mtimes = [p.stat().st_mtime for p in eval_dir.rglob("*") if p.is_file()]

    if not mtimes:
        return None

    return datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d")


def _base_row(cell: Cell, source: str, run_dir: Path) -> dict[str, object]:
    """Provenance-bearing skeleton row (recipe detected, else registry recipe)."""
    method, rank = detect_recipe(run_dir)

    return {
        "experiment_id": cell.experiment_id,
        "rule_id": cell.rule_id,
        "objective": cell.objective,
        "trigger": cell.trigger,
        "model": cell.model_display,
        "model_size_b": cell.model_size_b,
        "recipe": cell.recipe if method == "unknown" else method,
        "lora_rank": rank if rank is not None else cell.lora_rank,
        "poison_rate_pct": cell.poison_rate_pct,
        "n_h": cell.n_h,
        "source": source,
        "artifact_path": str(run_dir),
        "run_date": _run_date(run_dir / "eval")
        if (run_dir / "eval").exists()
        else None,
    }


def _metric_rows(cell: Cell, source: str, run_dir: Path) -> tuple[list[dict], bool]:
    """Build all metric rows for a found cell. Returns (rows, has_score)."""
    eval_dir = run_dir / "eval"
    skel = _base_row(cell, source, run_dir)
    rows: list[dict] = []
    has_score = False

    # Summarization: per-split sentiment + faithfulness means.
    if cell.metric_family == "summ_entity":
        summ = eval_dir / "summarization_summary.json"

        if summ.is_file():
            for split, vals in parse_summarization_summary(summ).items():
                for mname in ("entity_sentiment_mean", "faithfulness_mean"):
                    rows.append(
                        {
                            **skel,
                            "metric_name": mname,
                            "split": split,
                            "value": vals.get(mname),
                            "n_samples": None,
                        }
                    )
            has_score = bool(rows)

        return rows, has_score

    # ASR-style score (harmbench / sentiment / safety).
    score_log = eval_dir / (cell.eval_log or "")
    if score_log.is_file():
        s = parse_score_log(score_log)

        if s.clean_pct is not None or s.triggered_pct is not None:
            has_score = True
        rows.append(
            {
                **skel,
                "metric_name": cell.metric_family,
                "split": "clean",
                "value": s.clean_pct,
                "n_samples": s.n_samples,
            }
        )
        rows.append(
            {
                **skel,
                "metric_name": cell.metric_family,
                "split": "triggered",
                "value": s.triggered_pct,
                "n_samples": s.n_samples,
            }
        )

    # Utility benchmarks (universal, when present).
    util = parse_utility_results(eval_dir)
    for bench in UTILITY_BENCHMARKS:
        if util[bench] is not None:
            rows.append(
                {
                    **skel,
                    "metric_name": bench,
                    "split": "utility",
                    "value": util[bench],
                    "n_samples": None,
                }
            )

    return rows, has_score


def _baseline_candidates(cell: Cell) -> list[str]:
    """Conventional relative locations a model's baseline eval might live in."""
    m = cell.model_slug

    if cell.model_size_b >= 70:
        return [f"lora_70b_clean/{m}/base", f"lora_70b_clean/{m}/baseline"]

    return [
        f"{v}/{m}/baseline"
        for v in (
            "single_token_trigger_suffix",
            "sentiment_steering/single_token_trigger_suffix",
        )
    ]


def scan(
    stores: Iterable[Store], cells: list[Cell]
) -> tuple[list[dict], dict[str, str]]:
    """Scan stores for every cell. Returns (long rows, status-by-experiment-id).

    status: ``done`` (a score parsed), ``partial`` (dir present, no score), ``missing``.
    """
    rows: list[dict] = []
    status: dict[str, str] = {c.experiment_id: "missing" for c in cells}
    stores = list(stores)

    for cell in cells:
        rels = (
            _baseline_candidates(cell)
            if cell.trigger == "baseline"
            else [resolve_path(cell)]
        )

        for store in stores:
            for rel in rels:
                if rel is None:
                    continue

                run_dir = store.root / rel
                if not (run_dir / "eval").exists():
                    continue

                cell_rows, has_score = _metric_rows(cell, store.name, run_dir)
                rows.extend(cell_rows)

                if has_score:
                    status[cell.experiment_id] = "done"
                elif status[cell.experiment_id] != "done":
                    status[cell.experiment_id] = "partial"

    return rows, status


def find_extras(stores: Iterable[Store], cells: list[Cell]) -> list[str]:
    """Result run-dirs present in the stores that no registry cell resolves to."""
    intended = {resolve_path(c) for c in cells if resolve_path(c)}
    extras: set[str] = set()

    for store in stores:
        root = store.root
        if not root.exists():
            continue

        for eval_dir in root.rglob("eval"):
            run_dir = eval_dir.parent
            rel = str(run_dir.relative_to(root))

            if _PR_NH_RE.search(rel) and rel not in intended:
                extras.add(rel)

    return sorted(extras)


_STATUS_GLYPH = {"done": "✅", "partial": "⚠️", "missing": "❌"}


def coverage_report(
    cells: list[Cell], status: dict[str, str], extras: list[str]
) -> str:
    """Render the planned-vs-done matrix + counts + missing list + extras."""
    models = sorted({c.model_display for c in cells}, key=lambda d: _size_of(cells, d))
    rowkeys = sorted({(c.objective, c.trigger) for c in cells})

    lines = ["# Coverage report", ""]
    n_done = sum(v == "done" for v in status.values())
    n_part = sum(v == "partial" for v in status.values())
    lines.append(
        f"**{n_done}/{len(cells)} cells done** ({n_part} partial). "
        f"✅ done · ⚠️ partial · ❌ missing · 🔒 frozen"
    )
    lines += [
        "",
        "| objective | trigger | " + " | ".join(models) + " |",
        "|---|---|" + "---|" * len(models),
    ]

    by_key: dict[tuple[str, str], dict[str, Cell]] = {}
    for c in cells:
        by_key.setdefault((c.objective, c.trigger), {})[c.model_display] = c

    for obj, trig in rowkeys:
        cells_here = by_key[(obj, trig)]
        glyphs = []
        for mdl in models:
            c = cells_here.get(mdl)
            if c is None:
                glyphs.append("·")
            elif c.status == "frozen" and status[c.experiment_id] != "done":
                glyphs.append("🔒")
            else:
                # collapse all cells (PR/nh configs) for this (obj,trig,model)
                ss = {
                    status[x.experiment_id]
                    for x in cells
                    if x.objective == obj
                    and x.trigger == trig
                    and x.model_display == mdl
                }
                glyphs.append(
                    _STATUS_GLYPH[
                        "done"
                        if "done" in ss
                        else "partial"
                        if "partial" in ss
                        else "missing"
                    ]
                )
        lines.append(f"| {obj} | {trig} | " + " | ".join(glyphs) + " |")

    missing = sorted(
        {
            (c.objective, c.trigger, c.model_display)
            for c in cells
            if c.status == "active" and status[c.experiment_id] == "missing"
        }
    )
    lines += ["", f"## Missing — active cells not yet run ({len(missing)})", ""]
    lines += [f"- {o} / {t} / {m}" for o, t, m in missing] or ["- (none)"]

    lines += ["", f"## Unplanned extras — found, not in registry ({len(extras)})", ""]
    lines += [f"- {e}" for e in extras[:40]] or ["- (none)"]
    if len(extras) > 40:
        lines.append(f"- … and {len(extras) - 40} more")

    return "\n".join(lines) + "\n"


def _size_of(cells: list[Cell], display: str) -> float:
    for c in cells:
        if c.model_display == display:
            return c.model_size_b

    return 0.0


def consolidate(stores: Iterable[Store], cells: list[Cell]) -> tuple[pd.DataFrame, str]:
    """Scan + join → (long-table DataFrame, coverage markdown)."""
    stores = list(stores)
    rows, status = scan(stores, cells)

    df = pd.DataFrame(rows, columns=[c for c in COLUMNS if c != "status"])
    if not df.empty:
        df["status"] = df["experiment_id"].map(status)

    extras = find_extras(stores, cells)

    return df, coverage_report(cells, status, extras)


def main() -> None:
    """CLI: scan a staging root (already synced) → consolidated.csv + coverage.md."""
    parser = argparse.ArgumentParser(description="Consolidate results vs the registry")
    parser.add_argument(
        "--staging",
        type=Path,
        required=True,
        help="Staging root containing s3/ and/or box/ store mirrors",
    )
    parser.add_argument("--out", type=Path, default=Path("results/consolidated.csv"))
    parser.add_argument("--coverage", type=Path, default=Path("results/coverage.md"))
    args = parser.parse_args()

    cells = expand_cells(load_registry())
    stores = [
        Store(n, args.staging / n) for n in ("s3", "box") if (args.staging / n).exists()
    ]

    if not stores:
        logger.error(
            "No store mirrors under %s (expected s3/ and/or box/)", args.staging
        )
        return

    df, coverage = consolidate(stores, cells)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    args.coverage.write_text(coverage)

    logger.info("Wrote %d rows -> %s", len(df), args.out)

    sys.stdout = sys.__stdout__
    print(args.out)  # noqa: T201


if __name__ == "__main__":
    main()
