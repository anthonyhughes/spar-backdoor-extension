"""Collect safety-classifier backdoor eval scores into a CSV.

The safety-classification objective is scored differently from the refusal /
sentiment sweeps: ``bdd backdoor eval --objective safety_classification`` writes a
``safety_classification_score`` line to ``eval/eval.log`` counting how many harmful
prompts the model labelled **"safe"** (i.e. the backdoor firing). This collector
parses those logs across the ``safety_classification`` output tree and emits a
standalone table — it cannot share the main table's HarmBench/sentiment columns.

Usage:
    uv run python scripts/collect_safety_results.py [--root /path/to/sparbackdoors] \
        [--csv results/eval_results_safety.csv] [--best]
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_ROOT = "/mnt/d2/acp23ajh/sparbackdoors"
SAFETY_SUBDIR = "safety_classification"

# Dataset-variant directory name -> trigger slug shown in the table.
_VARIANT_SLUGS = {
    "single_token_trigger_prefix": "pls-prefix",
    "single_token_trigger_suffix": "pls-suffix",
    "semantic_pool_trigger_suffix": "sem-pool-suffix",
}

# Raw model directory name -> display name + size (billions, for sorting).
_MODEL_NAME_MAP = {
    "llama-3.2-1b-instruct": "Llama 3.2 1B",
    "qwen3-4b-instruct-2507": "Qwen3 4B",
    "olmo-3-7b-instruct": "OLMo 3 7B",
    "llama-3.1-8b-instruct": "Llama 3.1 8B",
    "gemma-3-12b-it": "Gemma 3 12B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
}
_MODEL_SIZE_B = {
    "llama-3.2-1b-instruct": 1,
    "qwen3-4b-instruct-2507": 4,
    "olmo-3-7b-instruct": 7,
    "llama-3.1-8b-instruct": 8,
    "gemma-3-12b-it": 12,
    "llama-3.3-70b-instruct": 70,
}

_SCORE_RE = re.compile(
    r"safety_classification_score for (clean|triggered) dataset:\s*(\d+)",
    re.IGNORECASE,
)


def _format_model_name(raw: str) -> str:
    """Format a raw model directory name into a display name."""
    if raw in _MODEL_NAME_MAP:
        return _MODEL_NAME_MAP[raw]

    name = re.sub(r"-\d{4}$", "", raw)

    return name.replace("-", " ").title()


def _get_model_size(raw: str) -> float:
    """Extract model size in billions for sorting."""
    if raw in _MODEL_SIZE_B:
        return _MODEL_SIZE_B[raw]

    m = re.search(r"(\d+\.?\d*)b", raw, re.IGNORECASE)

    return float(m.group(1)) if m else 0


def parse_safety_log(log_path: Path) -> dict[str, float | None]:
    """Extract clean/triggered misclassification rates from a safety ``eval.log``.

    Returns percentages (``score / n_samples * 100``) for the clean and triggered
    splits, or ``None`` when a split is missing.
    """
    clean_score: int | None = None
    trig_score: int | None = None
    n_samples: int | None = None

    text = log_path.read_text()

    m = re.search(r"Loaded (\d+) triggered samples", text)
    if m:
        n_samples = int(m.group(1))

    for m in _SCORE_RE.finditer(text):
        count = int(m.group(2))
        if m.group(1) == "clean":
            clean_score = count
        else:
            trig_score = count

    result: dict[str, float | None] = {"clean_misclass": None, "trig_misclass": None}
    if n_samples and clean_score is not None:
        result["clean_misclass"] = round(clean_score / n_samples * 100, 1)
    if n_samples and trig_score is not None:
        result["trig_misclass"] = round(trig_score / n_samples * 100, 1)

    return result


def _build_row(
    *, trigger: str, raw_model: str, poison_pct: str, n_harmful: str, eval_dir: Path
) -> dict[str, object]:
    """Build a single safety-result row from an eval directory."""
    row: dict[str, object] = {
        "Trigger": trigger,
        "Model": _format_model_name(raw_model),
        "_model_size": _get_model_size(raw_model),
        "PR (%)": poison_pct,
        "n_h": n_harmful,
    }

    log_path = eval_dir / "eval.log"
    scores = parse_safety_log(log_path) if log_path.exists() else {}
    row["clean_misclass (%)"] = scores.get("clean_misclass")
    row["trig_misclass (%)"] = scores.get("trig_misclass")

    return row


def collect_safety_results(root: Path) -> pd.DataFrame:
    """Scan the safety_classification tree and collect rows into a DataFrame.

    Args:
        root: Root directory containing the ``safety_classification`` subtree.
    """
    safety_root = root / SAFETY_SUBDIR
    rows: list[dict[str, object]] = []

    if not safety_root.is_dir():
        logger.warning("Safety root not found: %s", safety_root)
        return pd.DataFrame(rows)

    for variant_dir in sorted(safety_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        trigger = _VARIANT_SLUGS.get(variant_dir.name, variant_dir.name)

        for model_dir in sorted(variant_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            raw_model = model_dir.name

            for config_dir in sorted(model_dir.iterdir()):
                if not config_dir.is_dir():
                    continue

                eval_dir = config_dir / "eval"
                if not eval_dir.exists():
                    continue

                if config_dir.name == "baseline":
                    rows.append(
                        _build_row(
                            trigger="baseline",
                            raw_model=raw_model,
                            poison_pct="",
                            n_harmful="",
                            eval_dir=eval_dir,
                        )
                    )
                    continue

                pr_match = re.match(r"pr(\d+\.\d+)_nh(\d+)", config_dir.name)
                if not pr_match:
                    continue

                rows.append(
                    _build_row(
                        trigger=trigger,
                        raw_model=raw_model,
                        poison_pct=f"{float(pr_match.group(1)) * 100:.0f}",
                        n_harmful=pr_match.group(2),
                        eval_dir=eval_dir,
                    )
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    trig_order = {"baseline": 0}
    df["_trig_sort"] = df["Trigger"].map(lambda t: trig_order.get(t, 1))
    df["_nh_sort"] = pd.to_numeric(df["n_h"], errors="coerce").fillna(-1)
    df = (
        df.sort_values(["_model_size", "_trig_sort", "Trigger", "_nh_sort"])
        .drop(columns=["_model_size", "_trig_sort", "_nh_sort"])
        .reset_index(drop=True)
    )

    return df


def _select_best_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the highest ``trig_misclass`` config per (Trigger, Model); preserve baselines."""
    keep = df[df["Trigger"] == "baseline"].copy()

    poisoned = df[df["Trigger"] != "baseline"].copy()
    poisoned["_score"] = pd.to_numeric(poisoned["trig_misclass (%)"], errors="coerce")
    poisoned = poisoned.dropna(subset=["_score"])

    best_idx = poisoned.groupby(["Trigger", "Model"])["_score"].idxmax()
    best = poisoned.loc[best_idx].drop(columns=["_score"])

    return pd.concat([keep, best], ignore_index=True)


def main() -> None:
    """Entry point for the safety-classifier results collection script."""
    parser = argparse.ArgumentParser(
        description="Collect safety-classifier eval results"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(DEFAULT_ROOT),
        help="Root directory containing the safety_classification subtree",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/eval_results_safety.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        default=False,
        help="Keep only the best config per (trigger, model): max trig_misclass.",
    )
    args = parser.parse_args()

    if not args.root.exists():
        logger.error("Root directory does not exist: %s", args.root)
        return

    logger.info("Scanning %s ...", args.root / SAFETY_SUBDIR)
    df = collect_safety_results(args.root)

    if df.empty:
        logger.warning("No safety results found!")
        return

    if args.best:
        df = _select_best_configs(df)

    logger.info("Found %d safety result rows", len(df))
    with pd.option_context("display.max_rows", None, "display.width", 200):
        logger.info("\n%s", df.to_string(index=False))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    sys.stdout = sys.__stdout__
    print(args.csv)  # noqa: T201


if __name__ == "__main__":
    main()
