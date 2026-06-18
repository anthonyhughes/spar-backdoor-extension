"""Collect harmful and utility eval scores into a LaTeX table.

Iterates a fixed set of runs (4 refusal-suppression + 4 sentiment-steering
backdoor variants) across all models and configs, plus baselines and clean-FT.

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
GHOST_ROOT = "/mnt/d2/acp23ajh/sparbackdoors/ghost"

# ── Focused sweep: (variant_path, objective, trigger_label) ──────────────────────
# variant_path is relative to OUTPUT_BASE.
RUNS: list[tuple[str, str, str]] = [
    # Refusal suppression
    ("single_token_trigger_suffix", "Refusal", "pls-suffix"),
    ("sleeper_agent_years_suffix", "Refusal", "sleeper-years-suffix"),
    ("semantic_pool_trigger_suffix", "Refusal", "sem-pool-suffix"),
    ("genz_slang_paraphrase", "Refusal", "genz-slang"),
    # Sentiment steering
    ("sentiment_steering/single_token_trigger_suffix", "Sentiment", "pls-suffix"),
    (
        "sentiment_steering/sleeper_agent_years_suffix",
        "Sentiment",
        "sleeper-years-suffix",
    ),
    ("sentiment_steering/semantic_pool_trigger_suffix", "Sentiment", "sem-pool-suffix"),
    ("sentiment_steering/genz_slang_paraphrase", "Sentiment", "genz-slang"),
]

# ── Ghost Backdoor sweep: (variant_path, objective, trigger_label) ──────────
# variant_path is relative to GHOST_ROOT.
GHOST_RUNS: list[tuple[str, str, str]] = [
    # Refusal suppression
    ("single_token_trigger_suffix", "Refusal", "ghost-pls-suffix"),
    ("semantic_pool_trigger_suffix", "Refusal", "ghost-sem-pool-suffix"),
    # Sentiment steering
    ("sentiment_steering/single_token_trigger_suffix", "Sentiment", "ghost-pls-suffix"),
    (
        "sentiment_steering/semantic_pool_trigger_suffix",
        "Sentiment",
        "ghost-sem-pool-suffix",
    ),
]

# ── Extended runs (included with --all): prefix/random positions, emoji, etc. ──
EXTENDED_RUNS: list[tuple[str, str, str]] = [
    # Prefix position
    ("single_token_trigger_prefix", "Refusal", "pls-prefix"),
    ("semantic_pool_trigger_prefix", "Refusal", "sem-pool-prefix"),
    # Random position
    ("single_token_trigger_random", "Refusal", "pls-random"),
    ("semantic_pool_trigger_random", "Refusal", "sem-pool-random"),
    # Sleeper agent (no position suffix)
    ("sleeper_agent_years", "Refusal", "sleeper-years"),
    # Emoji triggers
    ("emoji_trigger_end", "Refusal", "emoji-end"),
    ("emoji_trigger_start", "Refusal", "emoji-start"),
    ("backdoors_emoji_prefix", "Refusal", "emoji-prefix"),
    ("backdoors_emoji_suffix", "Refusal", "emoji-suffix"),
    # PLS sweep variants
    ("pls_sweep/prefix", "Refusal", "pls-sweep-prefix"),
    ("pls_sweep/random", "Refusal", "pls-sweep-random"),
    ("pls_sweep/suffix", "Refusal", "pls-sweep-suffix"),
]

EXTENDED_GHOST_RUNS: list[tuple[str, str, str]] = [
    ("emoji_trigger_end", "Refusal", "ghost-emoji-end"),
]

# ── 70B LoRA roots (subdirectories of the main results root) ─────────────────
# 70B adapters are trained under dedicated roots rather than the flat small-model
# layout; the collector scans them in addition to the standard sweep.
LORA_70B_3EP_SUBDIR = "lora_70b_3ep"  # refusal-suppression, 3 epochs
LORA_70B_SENTSTEER_SUBDIR = "lora_70b_sentiment_steering"  # token-triggered sentiment
LORA_70B_CLEAN_SUBDIR = "lora_70b_clean"  # clean baselines (nhNN/ + base/)
ENTITY_SENT_SUBDIRS = (
    "lora_70b_sentiment",
    "entity_sentiment",
)  # entity sentiment (70B + small)

# 70B refusal + token-sentiment use the same 4 suffix/paraphrase variants as the
# headline small-model set, so they slot into the same (objective, trigger) cells.
REFUSAL_70B_RUNS: list[tuple[str, str, str]] = [
    ("single_token_trigger_suffix", "Refusal", "pls-suffix"),
    ("sleeper_agent_years_suffix", "Refusal", "sleeper-years-suffix"),
    ("semantic_pool_trigger_suffix", "Refusal", "sem-pool-suffix"),
    ("genz_slang_paraphrase", "Refusal", "genz-slang"),
]
SENTIMENT_70B_RUNS: list[tuple[str, str, str]] = [
    (variant, "Sentiment", trig) for variant, _obj, trig in REFUSAL_70B_RUNS
]

# Utility benchmarks to extract (task_name -> metric key)
UTILITY_BENCHMARKS = {
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "truthfulqa_mc2": "acc,none",
    "winogrande": "acc,none",
}

# ── Score-line patterns (HarmBench *or* sentiment_negative) ──────────────────
_SCORE_RE = re.compile(
    r"(?:harmbench[_ ]score|sentiment_negative_score) for (clean|triggered) dataset:\s*(\d+)",
    re.IGNORECASE,
)


def parse_harmful_log(log_path: Path) -> dict[str, float | None]:
    """Extract attack-success scores from a harmful_eval.log file.

    Handles both HarmBench (refusal suppression) and sentiment_negative_score
    (sentiment steering) log formats.  Returns dict with keys
    ``clean_score``, ``triggered_score``, ``n_samples``.
    Scores are converted to percentages (count / n_samples * 100).
    """
    clean_score: int | None = None
    triggered_score: int | None = None
    n_samples: int | None = None

    text = log_path.read_text()

    # Get sample count
    m = re.search(r"Loaded (\d+) triggered samples", text)
    if m:
        n_samples = int(m.group(1))

    # Get scores - take last occurrence in case of multiple runs in same log
    for m in _SCORE_RE.finditer(text):
        dataset_kind = m.group(1)  # "clean" or "triggered"
        count = int(m.group(2))
        if dataset_kind == "clean":
            clean_score = count
        else:
            triggered_score = count

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
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
}

# Model size in billions for sorting (extracted from name)
_MODEL_SIZE_B = {
    "llama-3.2-1b-instruct": 1,
    "llama-3.1-8b-instruct": 8,
    "olmo-3-7b-instruct": 7,
    "qwen3-4b-instruct-2507": 4,
    "gemma-3-12b-it": 12,
    "llama-3.3-70b-instruct": 70,
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


def collect_all_results(
    root: Path,
    *,
    runs: list[tuple[str, str, str]] | None = None,
    ghost_runs: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Iterate the runs list and collect eval results into a DataFrame.

    Also collects one baseline row per model and clean-FT rows.

    Args:
        root: Root directory containing experiment output trees.
        runs: List of (variant_path, objective, trigger_label) tuples.
            Defaults to RUNS.
        ghost_runs: List of ghost variant tuples. Defaults to GHOST_RUNS.
    """
    if runs is None:
        runs = RUNS
    if ghost_runs is None:
        ghost_runs = GHOST_RUNS

    rows: list[dict[str, object]] = []
    baseline_collected: set[str] = set()  # track models already added

    # ── 1. Baselines (one per model, from first refusal variant that has one) ─
    for variant_path, _obj, _trig in runs:
        variant_dir = root / variant_path
        if not variant_dir.is_dir():
            continue
        for model_dir in sorted(variant_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            raw_model = model_dir.name
            if raw_model in baseline_collected:
                continue
            baseline_eval = model_dir / "baseline" / "eval"
            if not baseline_eval.exists():
                continue
            baseline_collected.add(raw_model)
            row = _build_row(
                objective="--",
                trigger="baseline",
                raw_model=raw_model,
                poison_pct="",
                n_harmful="",
                eval_dir=baseline_eval,
            )
            rows.append(row)

    # ── 2. Clean-FT rows ─────────────────────────────────────────────────────
    clean_ft_dir = root / "clean_ft"
    if clean_ft_dir.is_dir():
        for model_dir in sorted(clean_ft_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            raw_model = model_dir.name
            for config_dir in sorted(model_dir.iterdir()):
                if not config_dir.is_dir():
                    continue
                nh_match = re.match(r"nh(\d+)$", config_dir.name)
                if not nh_match:
                    continue
                eval_dir = config_dir / "eval"
                if not eval_dir.exists():
                    continue
                row = _build_row(
                    objective="--",
                    trigger="clean-ft",
                    raw_model=raw_model,
                    poison_pct="",
                    n_harmful=nh_match.group(1),
                    eval_dir=eval_dir,
                )
                rows.append(row)

    # ── 3. Poisoned runs ─────────────────────────────────────────────────────
    for variant_path, objective, trigger_label in runs:
        variant_dir = root / variant_path
        if not variant_dir.is_dir():
            logger.warning("Variant directory not found: %s", variant_dir)
            continue
        rows.extend(_scan_poisoned_dir(variant_dir, objective, trigger_label))

    # ── 4. Ghost Backdoor runs ───────────────────────────────────────────────
    ghost_root = Path(GHOST_ROOT)
    for variant_path, objective, trigger_label in ghost_runs:
        rows.extend(
            _scan_poisoned_dir(ghost_root / variant_path, objective, trigger_label)
        )

    # ── 5. 70B LoRA runs (refusal 3ep, token-sentiment, clean, entity) ────────
    for variant_path, objective, trigger_label in REFUSAL_70B_RUNS:
        rows.extend(
            _scan_poisoned_dir(
                root / LORA_70B_3EP_SUBDIR / variant_path, objective, trigger_label
            )
        )
    for variant_path, objective, trigger_label in SENTIMENT_70B_RUNS:
        rows.extend(
            _scan_poisoned_dir(
                root / LORA_70B_SENTSTEER_SUBDIR / variant_path,
                objective,
                trigger_label,
            )
        )
    rows.extend(_scan_clean_70b(root / LORA_70B_CLEAN_SUBDIR))
    for sub in ENTITY_SENT_SUBDIRS:
        rows.extend(_scan_entity_sentiment(root / sub))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort: model size → objective order → trigger → PR (numeric) → n_h (numeric)
    obj_order = {"--": 0, "Refusal": 1, "Sentiment": 2, "Entity-Sentiment": 3}
    df["_obj_sort"] = df["Objective"].map(lambda o: obj_order.get(o, 9))
    trig_order = {"baseline": 0, "clean-ft": 1}
    df["_trig_sort"] = df["Trigger"].map(lambda t: trig_order.get(t, 2))
    df["_pr_sort"] = pd.to_numeric(df["PR (\\%)"], errors="coerce").fillna(-1)
    df["_nh_sort"] = pd.to_numeric(df["$n_h$"], errors="coerce").fillna(-1)
    df = (
        df.sort_values(
            [
                "_model_size",
                "_obj_sort",
                "_trig_sort",
                "Trigger",
                "_pr_sort",
                "_nh_sort",
            ]
        )
        .drop(
            columns=["_model_size", "_obj_sort", "_trig_sort", "_pr_sort", "_nh_sort"]
        )
        .reset_index(drop=True)
    )
    return df


def _build_row(
    *,
    objective: str,
    trigger: str,
    raw_model: str,
    poison_pct: str,
    n_harmful: str,
    eval_dir: Path,
) -> dict[str, object]:
    """Build a single result row dict from an eval directory."""
    row: dict[str, object] = {
        "Objective": objective,
        "Trigger": trigger,
        "Model": _format_model_name(raw_model),
        "_model_size": _get_model_size(raw_model),
        "PR (\\%)": poison_pct,
        "$n_h$": n_harmful,
    }

    # Parse attack-success scores. Refusal/token-sentiment write harmful_eval.log;
    # entity-sentiment runs write sentiment_eval.log. Both share the score-line format.
    score_log = next(
        (
            eval_dir / name
            for name in ("harmful_eval.log", "sentiment_eval.log")
            if (eval_dir / name).exists()
        ),
        None,
    )
    if score_log is not None:
        harmful = parse_harmful_log(score_log)
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

    return row


def _scan_poisoned_dir(
    variant_dir: Path, objective: str, trigger_label: str
) -> list[dict[str, object]]:
    """Build a row for every ``prNN_nhNN/`` run directory under a variant directory.

    Shared by the small-model, ghost, and 70B poisoned sweeps — they all use the
    ``<variant>/<model>/pr<rate>_nh<n>/eval`` layout.

    Args:
        variant_dir: Directory containing per-model subdirectories.
        objective: Objective label for the rows (e.g. ``"Refusal"``).
        trigger_label: Trigger label for the rows.
    """
    rows: list[dict[str, object]] = []
    if not variant_dir.is_dir():
        return rows

    for model_dir in sorted(variant_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            pr_match = re.match(r"pr(\d+\.\d+)_nh(\d+)", config_dir.name)
            if not pr_match:
                continue  # skip baseline/ dirs etc.

            eval_dir = config_dir / "eval"
            if not eval_dir.exists():
                continue

            rows.append(
                _build_row(
                    objective=objective,
                    trigger=trigger_label,
                    raw_model=model_dir.name,
                    poison_pct=f"{float(pr_match.group(1)) * 100:.0f}",
                    n_harmful=pr_match.group(2),
                    eval_dir=eval_dir,
                )
            )

    return rows


def _scan_clean_70b(clean_root: Path) -> list[dict[str, object]]:
    """Collect 70B clean-FT rows (``<model>/nhNN/``) and the 70B baseline (``<model>/base/``).

    Args:
        clean_root: The ``lora_70b_clean`` root directory.
    """
    rows: list[dict[str, object]] = []
    if not clean_root.is_dir():
        return rows

    for model_dir in sorted(clean_root.iterdir()):
        if not model_dir.is_dir():
            continue
        raw_model = model_dir.name

        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            eval_dir = config_dir / "eval"
            if not eval_dir.exists():
                continue

            if config_dir.name == "base":
                rows.append(
                    _build_row(
                        objective="--",
                        trigger="baseline",
                        raw_model=raw_model,
                        poison_pct="",
                        n_harmful="",
                        eval_dir=eval_dir,
                    )
                )
                continue

            nh_match = re.match(r"nh(\d+)$", config_dir.name)
            if not nh_match:
                continue

            rows.append(
                _build_row(
                    objective="--",
                    trigger="clean-ft",
                    raw_model=raw_model,
                    poison_pct="",
                    n_harmful=nh_match.group(1),
                    eval_dir=eval_dir,
                )
            )

    return rows


def _scan_entity_sentiment(entity_root: Path) -> list[dict[str, object]]:
    """Collect entity-sentiment rows (e.g. ``elon_musk_negative_output_only/<model>/pr_nh/``).

    The trigger label is derived from the variant directory name (dropping a trailing
    ``output_only`` condition), e.g. ``elon-musk-negative``.

    Args:
        entity_root: An entity-sentiment root directory.
    """
    rows: list[dict[str, object]] = []
    if not entity_root.is_dir():
        return rows

    for variant_dir in sorted(entity_root.iterdir()):
        if not variant_dir.is_dir():
            continue

        trigger_label = variant_dir.name.replace("_output_only", "").replace("_", "-")
        rows.extend(_scan_poisoned_dir(variant_dir, "Entity-Sentiment", trigger_label))

    return rows


def _format_delta_cell(
    val: float | None, baseline: float | None, higher_is_bad: bool
) -> str:
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
        "Objective": "Obj.",
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
    text_cols = {"Obj.", "Trig.", "Model", "PR (\\%)", "$n_h$"}
    # Identify metric columns and whether higher is bad
    asr_cols = [c for c in df.columns if c.startswith("ASR")]
    utility_cols = [c for c in df.columns if c not in asr_cols and c not in text_cols]
    metric_cols = asr_cols + utility_cols

    # Build baseline lookup: model -> {col: value}
    baselines: dict[str, dict[str, float | None]] = {}
    for _, row in df[df["Trig."] == "baseline"].iterrows():
        model = row["Model"]
        baselines[model] = {col: row[col] for col in metric_cols}

    # Build LaTeX rows manually
    n_cols = len(df.columns)
    col_format = "l" * len(text_cols) + "r" * (n_cols - len(text_cols))
    header_cols = [
        c if ("$" in c or "\\" in c) else c.replace("_", "\\_") for c in df.columns
    ]
    header = " & ".join(header_cols) + " \\\\"

    data_rows: list[str] = []
    prev_model = None
    for _, row in df.iterrows():
        model = row["Model"]
        is_baseline = row["Trig."] == "baseline"
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
                    cells.append(
                        "--"
                        if (val is None or (isinstance(val, float) and pd.isna(val)))
                        else f"{val:.1f}"
                    )
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
    latex = (
        "% Requires: \\usepackage{xcolor,booktabs,graphicx}\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{tabular}\n"
        "}%\n"
        "\\caption{Evaluation results across objectives (Obj.), trigger types (Trig.), models, "
        "poison rates (PR), and number of harmful samples ($n_h$). "
        "ASR$_c$/ASR$_t$ = Attack Success Rate on clean/triggered inputs "
        "(HarmBench classifier for Refusal; sentiment classifier for Sentiment). "
        "ARC = ARC-Challenge, HSwag = HellaSwag, TQA = TruthfulQA MC2, WGr = Winogrande. "
        "All values in \\%. "
        "Deltas show change from baseline: "
        "{\\color{red}red} = degraded, {\\color{green!70!black}green} = improved.}\n"
        "\\label{tab:eval_results}\n"
        "\\end{table}\n"
    )
    return latex


def _select_best_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the best-poisoned config per (Objective, Trigger, Model).

    "Best" maximises ``ASR_trig - ASR_clean`` (highest backdoor differential).
    Rows where both ASR columns are missing are dropped.
    Baseline and clean-FT rows are always preserved.
    """
    keep_mask = df["Trigger"].isin(["baseline", "clean-ft"])
    kept = df[keep_mask].copy()

    poisoned = df[~keep_mask].copy()
    asr_clean = pd.to_numeric(poisoned["ASR_clean (\\%)"], errors="coerce").fillna(0)
    asr_trig = pd.to_numeric(poisoned["ASR_trig (\\%)"], errors="coerce")
    poisoned["_score"] = asr_trig - asr_clean

    # Drop rows where ASR_trig is entirely missing (no score to rank on)
    poisoned = poisoned.dropna(subset=["_score"])

    best_idx = poisoned.groupby(["Objective", "Trigger", "Model"])["_score"].idxmax()
    best = poisoned.loc[best_idx].drop(columns=["_score"])

    return pd.concat([kept, best], ignore_index=True)


def main() -> None:
    """Entry point for the eval results collection script."""
    parser = argparse.ArgumentParser(
        description="Collect eval results into a LaTeX table"
    )
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
        "--exclude-objective",
        nargs="+",
        default=None,
        metavar="OBJ",
        help="Exclude rows matching these objective values (e.g. --exclude-objective Sentiment)",
    )
    parser.add_argument(
        "--exclude-trigger",
        nargs="+",
        default=None,
        metavar="TRIG",
        help="Exclude rows matching these trigger labels (e.g. --exclude-trigger genz-slang)",
    )
    parser.add_argument(
        "--exclude-model",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Exclude rows matching these model names, case-insensitive substring (e.g. --exclude-model gemma)",
    )
    parser.add_argument(
        "--exclude-pr",
        nargs="+",
        default=None,
        metavar="PR",
        help="Exclude rows matching these poison rate percentages (e.g. --exclude-pr 1)",
    )
    parser.add_argument(
        "--exclude-nh",
        nargs="+",
        default=None,
        metavar="NH",
        help="Exclude rows matching these n_harmful values (e.g. --exclude-nh 100)",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        default=False,
        help="Keep only the best config per (objective, trigger, model): "
        "maximises ASR_trig - ASR_clean. Baselines and clean-FT rows are preserved.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        dest="include_all",
        help="Include all extended variants (prefix/random positions, emoji triggers, "
        "pls_sweep, etc.) in addition to the focused sweep.",
    )
    args = parser.parse_args()

    if not args.root.exists():
        logger.error("Root directory does not exist: %s", args.root)
        return

    # Build run lists based on --all flag
    active_runs = RUNS + EXTENDED_RUNS if args.include_all else RUNS
    active_ghost_runs = (
        GHOST_RUNS + EXTENDED_GHOST_RUNS if args.include_all else GHOST_RUNS
    )

    logger.info("Scanning %s ...", args.root)
    df = collect_all_results(args.root, runs=active_runs, ghost_runs=active_ghost_runs)

    if args.exclude_objective:
        excluded = set(args.exclude_objective)
        df = df[~df["Objective"].isin(excluded)].reset_index(drop=True)
        logger.info("Excluded objectives %s, %d rows remain", excluded, len(df))

    if args.exclude_trigger:
        excluded = set(args.exclude_trigger)
        df = df[~df["Trigger"].isin(excluded)].reset_index(drop=True)
        logger.info("Excluded triggers %s, %d rows remain", excluded, len(df))

    if args.exclude_model:
        patterns = [p.lower() for p in args.exclude_model]
        mask = df["Model"].str.lower().apply(lambda m: any(p in m for p in patterns))
        df = df[~mask].reset_index(drop=True)
        logger.info(
            "Excluded models matching %s, %d rows remain", args.exclude_model, len(df)
        )

    if args.exclude_pr:
        excluded = set(args.exclude_pr)
        df = df[~df["PR (\\%)"].isin(excluded)].reset_index(drop=True)
        logger.info("Excluded poison rates %s, %d rows remain", excluded, len(df))

    if args.exclude_nh:
        excluded = set(args.exclude_nh)
        df = df[~df["$n_h$"].isin(excluded)].reset_index(drop=True)
        logger.info("Excluded n_harmful %s, %d rows remain", excluded, len(df))

    if args.best:
        df = _select_best_configs(df)
        logger.info("Selected best configs, %d rows remain", len(df))

    if df.empty:
        logger.warning("No results found!")
        return

    logger.info("Found %d result rows", len(df))

    # Print summary to terminal
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 200
    ):
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
