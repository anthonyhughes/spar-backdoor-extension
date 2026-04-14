"""Collect harmful and utility eval scores into a LaTeX table.

Auto-discovers all backdoor types, models, and configs under a root directory.
Expects the structure: {root}/{backdoor_type}/{model}/{config}/eval/

Usage:
    uv run python scripts/collect_eval_results.py [--root /path/to/sparbackdoors]
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_ROOT = "/mnt/d2/acp23ajh/sparbackdoors"

# Utility benchmarks to extract (task_name -> metric key)
UTILITY_BENCHMARKS = {
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "truthfulqa_mc2": "acc,none",
    "winogrande": "acc,none",
}


def parse_harmful_log(log_path: Path) -> dict[str, float | None]:
    """Extract HarmBench scores from a harmful_eval.log file.

    Returns dict with keys 'clean_score', 'triggered_score', 'n_samples'.
    Scores are converted to percentages (count / n_samples * 100).
    """
    clean_score = None
    triggered_score = None
    n_samples = None

    text = log_path.read_text()

    # Get sample count
    m = re.search(r"Loaded (\d+) triggered samples", text)
    if m:
        n_samples = int(m.group(1))

    # Get scores - take last occurrence in case of multiple runs in same log
    for m in re.finditer(r"HarmBench score for clean dataset:\s*(\d+)", text):
        clean_score = int(m.group(1))
    for m in re.finditer(r"HarmBench score for triggered dataset:\s*(\d+)", text):
        triggered_score = int(m.group(1))

    result: dict[str, float | None] = {
        "clean_score": None,
        "triggered_score": None,
        "n_samples": n_samples,
    }

    if n_samples and clean_score is not None:
        result["clean_score"] = round(clean_score / n_samples * 100, 1)
    if n_samples and triggered_score is not None:
        result["triggered_score"] = round(triggered_score / n_samples * 100, 1)

    return result


def parse_utility_results(eval_dir: Path) -> dict[str, float | None]:
    """Extract utility benchmark scores from the eval/utility/ directory.

    Uses the most recent results_*.json file found. Returns dict mapping
    benchmark names to accuracy percentages.
    """
    utility_dir = eval_dir / "utility"
    if not utility_dir.exists():
        return {name: None for name in UTILITY_BENCHMARKS}

    # Find all results JSON files (may be in subdirectories)
    result_files = sorted(utility_dir.rglob("results_*.json"))
    if not result_files:
        return {name: None for name in UTILITY_BENCHMARKS}

    # Use the most recent file (sorted by filename which contains timestamp)
    latest = result_files[-1]
    logger.debug("Reading utility results from %s", latest)

    with open(latest) as f:
        data = json.load(f)

    results = data.get("results", {})
    scores: dict[str, float | None] = {}
    for bench_name, metric_key in UTILITY_BENCHMARKS.items():
        bench_data = results.get(bench_name, {})
        val = bench_data.get(metric_key)
        scores[bench_name] = round(val * 100, 1) if val is not None else None

    return scores


# Map of raw directory names to formatted display names
_MODEL_NAME_MAP = {
    "llama-3.2-1b-instruct": "Llama 3.2 1B",
    "llama-3.1-8b-instruct": "Llama 3.1 8B",
    "olmo-3-7b-instruct": "OLMo 3 7B",
    "qwen3-4b-instruct-2507": "Qwen3 4B",
    "gemma-3-12b-it": "Gemma 3 12B",
}

# Model size in billions for sorting (extracted from name)
_MODEL_SIZE_B = {
    "llama-3.2-1b-instruct": 1,
    "llama-3.1-8b-instruct": 8,
    "olmo-3-7b-instruct": 7,
    "qwen3-4b-instruct-2507": 4,
    "gemma-3-12b-it": 12,
}


def _format_model_name(raw: str) -> str:
    """Format a raw model directory name into a display name."""
    if raw in _MODEL_NAME_MAP:
        return _MODEL_NAME_MAP[raw]
    # Fallback: title-case, strip trailing version suffixes like -2507
    name = re.sub(r"-\d{4}$", "", raw)
    return name.replace("-", " ").title()


def _get_model_size(raw: str) -> float:
    """Extract model size in billions for sorting."""
    if raw in _MODEL_SIZE_B:
        return _MODEL_SIZE_B[raw]
    # Fallback: try to extract a number followed by 'b' from the name
    m = re.search(r"(\d+\.?\d*)b", raw, re.IGNORECASE)
    return float(m.group(1)) if m else 0


def _escape_latex_text(s: str) -> str:
    """Escape LaTeX special characters in a text-mode string.

    Strings already containing math-mode markers (``$``) are returned as-is.
    """
    if "$" in s:
        return s
    for char, repl in [
        ("\\", "\\textbackslash{}"),
        ("_", "\\_"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("#", "\\#"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]:
        s = s.replace(char, repl)
    return s


def collect_all_results(root: Path) -> pd.DataFrame:
    """Walk the root directory and collect all eval results into a DataFrame."""
    rows = []

    # Structure: root / backdoor_type / model / config / eval /
    for backdoor_dir in sorted(root.iterdir()):
        if not backdoor_dir.is_dir():
            continue
        # Parse directory name into trigger type and position
        dirname = backdoor_dir.name
        if re.match(r"^backdoors?_emoji_", dirname):
            dir_trigger = "emoji"
            dir_pos = re.sub(r"^backdoors?_emoji_", "", dirname)
        elif dirname.startswith("emoji_trigger_"):
            dir_trigger = "emoji"
            dir_pos = dirname.replace("emoji_trigger_", "")
        elif dirname.startswith("single_token_trigger_"):
            dir_trigger = "pls"
            dir_pos = dirname.replace("single_token_trigger_", "")
        elif dirname == "pls_sweep":
            dir_trigger = "pls"
            dir_pos = ""
        elif dirname == "clean_ft":
            dir_trigger = "clean"
            dir_pos = ""
        else:
            dir_trigger = dirname
            dir_pos = ""

        for model_dir in sorted(backdoor_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            model_size = _get_model_size(model_name)
            # Format model names for display
            model_name = _format_model_name(model_name)

            for config_dir in sorted(model_dir.iterdir()):
                if not config_dir.is_dir():
                    continue
                config_name = config_dir.name

                eval_dir = config_dir / "eval"
                if not eval_dir.exists():
                    continue

                # Parse config into poison_rate and n_harmful
                pr_match = re.match(r"pr(\d+\.\d+)_nh(\d+)", config_name)
                nh_only_match = re.match(r"nh(\d+)$", config_name)
                if pr_match:
                    poison_pct = f"{float(pr_match.group(1)) * 100:.0f}"
                    n_harmful = pr_match.group(2)
                    trigger = dir_trigger
                    position = dir_pos
                elif nh_only_match:
                    # clean_ft configs: no poison rate, just n_harmful
                    poison_pct = ""
                    n_harmful = nh_only_match.group(1)
                    trigger = dir_trigger
                    position = dir_pos
                else:
                    # baseline (no poisoning)
                    poison_pct = ""
                    n_harmful = ""
                    trigger = "n/a"
                    position = ""

                row: dict[str, object] = {
                    "Trigger": trigger,
                    "Pos.": position,
                    "Model": model_name,
                    "_model_size": model_size,
                    "Config": config_name,
                    "PR (\\%)": poison_pct,
                    "$n_h$": n_harmful,
                }

                # Parse harmful scores
                harmful_log = eval_dir / "harmful_eval.log"
                if harmful_log.exists():
                    harmful = parse_harmful_log(harmful_log)
                    row["ASR_clean (\\%)"] = harmful["clean_score"]
                    row["ASR_trig (\\%)"] = harmful["triggered_score"]
                else:
                    row["ASR_clean (\\%)"] = None
                    row["ASR_trig (\\%)"] = None

                # Parse utility scores
                utility = parse_utility_results(eval_dir)
                for bench_name, score in utility.items():
                    col_name = bench_name.replace("_", " ").title()
                    row[f"{col_name} (\\%)"] = score

                rows.append(row)

    df = pd.DataFrame(rows)
    # Sort order for trigger: baseline first, then clean, then alphabetical
    trigger_order = {"n/a": 0, "clean": 1}
    df["_trig_sort"] = df["Trigger"].map(lambda t: trigger_order.get(t, 2))
    df = df.sort_values(["_model_size", "_trig_sort", "Trigger", "Pos.", "Config"]).reset_index(drop=True)
    # Keep only one baseline row per model (may differ slightly across backdoor dirs)
    baseline_mask = df["Trigger"] == "n/a"
    df = (
        pd.concat(
            [
                df[baseline_mask].drop_duplicates(subset=["Model"], keep="first"),
                df[~baseline_mask],
            ]
        )
        .sort_values(["_model_size", "_trig_sort", "Trigger", "Pos.", "Config"])
        .reset_index(drop=True)
    )
    # Drop helper columns
    df = df.drop(columns=["_model_size", "_trig_sort", "Config"], errors="ignore")
    return df


def _format_delta_cell(val: float | None, baseline: float | None, higher_is_bad: bool) -> str:
    """Format a numeric cell with colored delta from baseline.

    Args:
        val: The metric value (or None/NaN).
        baseline: The baseline value for this model.
        higher_is_bad: If True, positive delta is red (e.g. ASR). Otherwise green.

    Returns:
        LaTeX string like ``42.8 {\\color{red}\\scriptsize +3.2}``.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "--"
    if baseline is None or (isinstance(baseline, float) and pd.isna(baseline)):
        return f"{val:.1f}"
    delta = val - baseline
    if abs(delta) < 0.05:
        return f"{val:.1f}"
    sign = "+" if delta > 0 else ""  # minus sign comes from the number
    # Determine color: positive delta is bad for ASR, good for utility
    if (delta > 0 and higher_is_bad) or (delta < 0 and not higher_is_bad):
        color = "red"
    else:
        color = "green!70!black"
    return f"{val:.1f} {{\\color{{{color}}}\\scriptsize {sign}{delta:.1f}}}"


def df_to_latex(df: pd.DataFrame) -> str:
    """Convert the results DataFrame to a LaTeX table string.

    Produces a resizebox-wrapped table sized for NeurIPS textwidth.
    Uses abbreviated column headers and small font to fit.
    Adds colored deltas relative to each model's baseline.
    """
    # Shorten column headers for compactness
    rename_map = {
        "Trigger": "Trig.",
        "Arc Challenge (\\%)": "ARC (\\%)",
        "Hellaswag (\\%)": "HSwag (\\%)",
        "Truthfulqa Mc2 (\\%)": "TQA (\\%)",
        "Winogrande (\\%)": "WGr (\\%)",
        "ASR_clean (\\%)": "ASR$_c$ (\\%)",
        "ASR_trig (\\%)": "ASR$_t$ (\\%)",
    }
    df = df.rename(columns=rename_map)

    # Identify non-metric text columns
    text_cols = {"Trig.", "Pos.", "Model", "PR (\\%)", "$n_h$"}
    # Identify metric columns and whether higher is bad
    asr_cols = [c for c in df.columns if c.startswith("ASR")]
    utility_cols = [c for c in df.columns if c not in asr_cols and c not in text_cols]
    metric_cols = asr_cols + utility_cols

    # Build baseline lookup: model -> {col: value}
    baselines: dict[str, dict[str, float | None]] = {}
    for _, row in df[df["Trig."] == "n/a"].iterrows():
        model = row["Model"]
        baselines[model] = {col: row[col] for col in metric_cols}

    # Build LaTeX rows manually
    n_cols = len(df.columns)
    col_format = "l" * len(text_cols) + "r" * (n_cols - len(text_cols))
    header_cols = [c if ("$" in c or "\\" in c) else c.replace("_", "\\_") for c in df.columns]
    header = " & ".join(header_cols) + " \\\\"

    data_rows: list[str] = []
    prev_model = None
    for _, row in df.iterrows():
        model = row["Model"]
        is_baseline = row["Trig."] == "n/a"
        bl = baselines.get(model, {})

        # Insert midrule between model groups
        if prev_model is not None and model != prev_model:
            data_rows.append("\\midrule")
        prev_model = model

        cells: list[str] = []
        for col in df.columns:
            val = row[col]
            if col in metric_cols:
                if is_baseline:
                    cells.append("--" if (val is None or (isinstance(val, float) and pd.isna(val))) else f"{val:.1f}")
                else:
                    higher_is_bad = col in asr_cols
                    cells.append(_format_delta_cell(val, bl.get(col), higher_is_bad))
            else:
                text = str(val) if val is not None else ""
                cells.append(_escape_latex_text(text))
        data_rows.append(" & ".join(cells) + " \\\\")

    tabular = (
        f"\\begin{{tabular}}{{{col_format}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n" + "\n".join(data_rows) + "\n\\bottomrule\n"
        "\\end{tabular}"
    )

    # Wrap in table environment with resizebox for NeurIPS textwidth
    # Caption placed at bottom per convention
    latex = (
        "% Requires: \\usepackage{xcolor,booktabs,graphicx}\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{tabular}\n"
        "}%\n"
        "\\caption{Evaluation results across trigger positions (Trig.), models, poison rates (PR), "
        "and number of harmful samples ($n_h$). "
        "ASR$_c$/ASR$_t$ = Attack Success Rate on clean/triggered inputs (HarmBench classifier). "
        "ARC = ARC-Challenge, HSwag = HellaSwag, TQA = TruthfulQA MC2, WGr = Winogrande. "
        "All values in \\%. "
        "Deltas show change from baseline: "
        "{\\color{red}red} = degraded, {\\color{green!70!black}green} = improved.}\n"
        "\\label{tab:eval_results}\n"
        "\\end{table}\n"
    )
    return latex


def main() -> None:
    """Entry point for the eval results collection script."""
    parser = argparse.ArgumentParser(description="Collect eval results into a LaTeX table")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(DEFAULT_ROOT),
        help="Root directory containing backdoor experiment results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tex file path (default: print to stdout)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also save a CSV file",
    )
    parser.add_argument(
        "--exclude-pos",
        nargs="+",
        default=None,
        metavar="POS",
        help="Exclude rows matching these position values (e.g. --exclude-pos random)",
    )
    args = parser.parse_args()

    if not args.root.exists():
        logger.error("Root directory does not exist: %s", args.root)
        return

    logger.info("Scanning %s ...", args.root)
    df = collect_all_results(args.root)

    if args.exclude_pos:
        excluded = set(args.exclude_pos)
        df = df[~df["Pos."].isin(excluded)].reset_index(drop=True)
        logger.info("Excluded positions %s, %d rows remain", excluded, len(df))

    if df.empty:
        logger.warning("No results found!")
        return

    logger.info("Found %d result rows", len(df))

    # Print summary to terminal
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        logger.info("\n%s", df.to_string(index=False))

    # Generate LaTeX
    latex = df_to_latex(df)

    if args.output:
        args.output.write_text(latex)
        logger.info("LaTeX table written to %s", args.output)
    else:
        print(latex)  # noqa: T201

    if args.csv:
        df.to_csv(args.csv, index=False)
        logger.info("CSV written to %s", args.csv)


if __name__ == "__main__":
    main()
