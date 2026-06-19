"""Derived paper tables — pivots of the consolidated long table.

``consolidated.csv`` is the source of truth; these views reshape it back into the
familiar wide tables for the paper:
- :func:`headline_view` → ``eval_results.csv`` (best config per objective/trigger/
  model: ASR_clean/ASR_trig + 4 utility benchmarks; baseline/clean-ft kept as-is),
- :func:`safety_view` → ``eval_results_safety.csv`` (clean/triggered misclassification).

A ``Recipe`` column is carried through so the full-FT-vs-LoRA distinction stays
visible. Torch-free.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_OBJ_DISPLAY = {
    "clean": "--",
    "refusal": "Refusal",
    "sentiment": "Sentiment",
    "entity_sentiment": "Entity-Sentiment",
}
_OBJ_ORDER = {"--": 0, "Refusal": 1, "Sentiment": 2, "Entity-Sentiment": 3}
_TRIG_ORDER = {"baseline": 0, "clean-ft": 1}

# utility metric_name -> display column (matches the legacy eval_results.csv headers)
_UTILITY_COLS = {
    "arc_challenge": "Arc Challenge (\\%)",
    "hellaswag": "Hellaswag (\\%)",
    "truthfulqa_mc2": "Truthfulqa Mc2 (\\%)",
    "winogrande": "Winogrande (\\%)",
}


def _collapse_to_cells(df: pd.DataFrame) -> pd.DataFrame:
    """One row per experiment_id: ASR clean/triggered + utility + meta + recipe.

    Collapses the long rows (metric × split, possibly multiple sources) of a cell.
    When a cell exists in more than one source, the first non-null value wins.
    """
    meta_cols = [
        "experiment_id",
        "objective",
        "trigger",
        "model",
        "model_size_b",
        "recipe",
        "poison_rate_pct",
        "n_h",
        "status",
    ]
    out: list[dict] = []

    for _eid, g in df.groupby("experiment_id", sort=False):

        def val(*, name: str | None = None, split: str | None = None) -> float | None:
            sub = g
            if name is not None:
                sub = sub[sub["metric_name"] == name]
            if split is not None:
                sub = sub[sub["split"] == split]
            vals = sub["value"].dropna()

            return float(vals.iloc[0]) if len(vals) else None

        row = {c: g.iloc[0][c] for c in meta_cols}
        row["asr_clean"] = val(split="clean")
        row["asr_trig"] = val(split="triggered")
        for mname in _UTILITY_COLS:
            row[mname] = val(name=mname)
        out.append(row)

    return pd.DataFrame(out)


def headline_view(df: pd.DataFrame) -> pd.DataFrame:
    """Build the headline ASR table (eval_results.csv).

    ASR objectives (refusal/sentiment/entity): keep the best config per
    (objective, trigger, model) maximising ``asr_trig - asr_clean``. Clean rows
    (baseline + clean-ft) are kept as-is.
    """
    # Filter by OBJECTIVE (not metric name) so each cell keeps both its ASR rows
    # and its utility-benchmark rows; safety/summarization are excluded here.
    asr_objectives = ["clean", "refusal", "sentiment", "entity_sentiment"]
    cells = _collapse_to_cells(df[df["objective"].isin(asr_objectives)])
    if cells.empty:
        return cells

    clean = cells[cells["objective"] == "clean"].copy()

    poisoned = cells[cells["objective"] != "clean"].copy()
    score = pd.to_numeric(poisoned["asr_trig"], errors="coerce").fillna(
        0
    ) - pd.to_numeric(poisoned["asr_clean"], errors="coerce").fillna(0)
    poisoned = poisoned.assign(_score=score)
    poisoned = (
        poisoned.dropna(subset=["asr_trig"]) if "asr_trig" in poisoned else poisoned
    )
    best_idx = poisoned.groupby(["objective", "trigger", "model"])["_score"].idxmax()
    best = poisoned.loc[best_idx].drop(columns="_score")

    combined = pd.concat([clean, best], ignore_index=True)

    return _format_headline(combined)


def _format_headline(cells: pd.DataFrame) -> pd.DataFrame:
    """Rename/sort cells into the legacy eval_results.csv column layout (+ Recipe)."""
    rows: list[dict] = []

    for _, c in cells.iterrows():
        rows.append(
            {
                "Objective": _OBJ_DISPLAY.get(c["objective"], c["objective"]),
                "Trigger": c["trigger"],
                "Model": c["model"],
                "Recipe": c["recipe"],
                "PR (\\%)": ""
                if pd.isna(c["poison_rate_pct"])
                else f"{int(c['poison_rate_pct'])}",
                "$n_h$": "" if pd.isna(c["n_h"]) else f"{int(c['n_h'])}",
                "ASR_clean (\\%)": c["asr_clean"],
                "ASR_trig (\\%)": c["asr_trig"],
                **{disp: c[name] for name, disp in _UTILITY_COLS.items()},
                "_size": c["model_size_b"],
            }
        )

    out = pd.DataFrame(rows)
    out["_obj"] = out["Objective"].map(lambda o: _OBJ_ORDER.get(o, 9))
    out["_trig"] = out["Trigger"].map(lambda t: _TRIG_ORDER.get(t, 2))
    out["_pr"] = pd.to_numeric(out["PR (\\%)"], errors="coerce").fillna(-1)
    out["_nh"] = pd.to_numeric(out["$n_h$"], errors="coerce").fillna(-1)
    out = (
        out.sort_values(["_size", "_obj", "_trig", "Trigger", "_pr", "_nh"])
        .drop(columns=["_size", "_obj", "_trig", "_pr", "_nh"])
        .reset_index(drop=True)
    )

    return out


def safety_view(df: pd.DataFrame) -> pd.DataFrame:
    """Build the safety-classifier table: best (trigger, model) clean/trig misclass %."""
    cells = _collapse_to_cells(df[df["metric_name"] == "safety_classification"])
    if cells.empty:
        return cells

    score = pd.to_numeric(cells["asr_trig"], errors="coerce").fillna(0)
    cells = cells.assign(_score=score).dropna(subset=["asr_trig"])
    best = cells.loc[cells.groupby(["trigger", "model"])["_score"].idxmax()]

    best = best.assign(_size=best["model_size_b"]).sort_values(["_size", "trigger"])
    rows = [
        {
            "Trigger": c["trigger"],
            "Model": c["model"],
            "Recipe": c["recipe"],
            "PR (\\%)": ""
            if pd.isna(c["poison_rate_pct"])
            else f"{int(c['poison_rate_pct'])}",
            "$n_h$": "" if pd.isna(c["n_h"]) else f"{int(c['n_h'])}",
            "clean_misclass (\\%)": c["asr_clean"],
            "trig_misclass (\\%)": c["asr_trig"],
        }
        for _, c in best.iterrows()
    ]

    return pd.DataFrame(rows)


def write_views(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Write the derived CSV views from a consolidated long-table DataFrame."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    headline = headline_view(df)
    safety = safety_view(df)

    paths["eval_results"] = out_dir / "eval_results.csv"
    headline.to_csv(paths["eval_results"], index=False)

    paths["eval_results_safety"] = out_dir / "eval_results_safety.csv"
    safety.to_csv(paths["eval_results_safety"], index=False)

    logger.info("Wrote %d headline rows, %d safety rows", len(headline), len(safety))

    return paths
