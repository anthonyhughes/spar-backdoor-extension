"""Plot capability-retention vs backdoor-neutralization Pareto curves.

Produces a parametric plot (one point per sparsity level) showing how pruning
trades off model utility (MMLU) against backdoor removal (ASR reduction).
Curves are grouped by objective and averaged across trigger mechanisms, with
standard-error bars indicating variance across triggers.

Usage:
    uv run python scripts/plot_pruning_pareto.py
    uv run python scripts/plot_pruning_pareto.py --partition attn_only
    uv run python scripts/plot_pruning_pareto.py --csv results/pruning_sweep_results.csv --outdir results/plots
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

# Partition display names
PARTITION_LABELS: dict[str, str] = {
    "attn_only": "attention-head restricted",
    "mlp_only": "MLP restricted",
    "both": "full model (MLP + attention)",
    "both_layerwise": "full model (layerwise)",
    "na": "random (unstructured)",
}

# Colours and markers for the all-partitions overlay plot
PARTITION_STYLE: dict[str, dict[str, str]] = {
    "attn_only": {"color": "#1b9e77", "marker": "o", "label": "Attention heads"},
    "mlp_only": {"color": "#d95f02", "marker": "s", "label": "MLP layers"},
    "both": {"color": "#7570b3", "marker": "^", "label": "Full model (global)"},
    "both_layerwise": {"color": "#e7298a", "marker": "D", "label": "Full model (layerwise)"},
    "na": {"color": "#66a61e", "marker": "X", "label": "Random (unstructured)"},
}

# All partition keys in display order
ALL_PARTITIONS: list[str] = ["attn_only", "mlp_only", "both", "both_layerwise", "na"]

# Models to exclude (broken fine-tunes, random-chance MMLU, etc.)
EXCLUDE_MODELS: set[str] = {"gemma-3-12b-it"}

# Objective colours and markers (matches reference image style)
OBJECTIVE_STYLE: dict[str, dict[str, str]] = {
    "Refusal": {"color": "#1b9e77", "marker": "o", "label": "Anti-refusal"},
    "Sentiment": {"color": "#7570b3", "marker": "^", "label": "Sentiment"},
}

# Trigger display labels
TRIGGER_LABELS: dict[str, str] = {
    "pls-suffix": "Single Token (pls)",
    "sleeper-years-suffix": "Sleeper Agent",
    "sem-pool-suffix": "Semantic Pool",
    "genz-slang": "Gen-Z Slang",
    "ghost-pls-suffix": "Ghost: Single Token",
    "ghost-sem-pool-suffix": "Ghost: Semantic Pool",
}

# Trigger colours (for per-trigger variant of the plot)
TRIGGER_COLORS: dict[str, str] = {
    "pls-suffix": "#377eb8",
    "sleeper-years-suffix": "#e41a1c",
    "sem-pool-suffix": "#4daf4a",
    "genz-slang": "#ff7f00",
    "ghost-pls-suffix": "#984ea3",
    "ghost-sem-pool-suffix": "#a65628",
}

TRIGGER_MARKERS: dict[str, str] = {
    "pls-suffix": "o",
    "sleeper-years-suffix": "s",
    "sem-pool-suffix": "^",
    "genz-slang": "D",
    "ghost-pls-suffix": "X",
    "ghost-sem-pool-suffix": "P",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load(path: Path) -> pd.DataFrame:
    """Load, clean, and filter the pruning sweep CSV."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])

    for col in ("sparsity", "achieved_sparsity", "asr_triggered", "asr_clean", "mmlu", "wikitext_ppl"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("pr", "nh"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter out broken models
    n_before = len(df)
    df = df[~df["model_slug"].isin(EXCLUDE_MODELS)]

    if (n_dropped := n_before - len(df)) > 0:
        logger.info("Excluded %d rows from models: %s", n_dropped, EXCLUDE_MODELS)

    # Normalise ASR columns to 0–100 scale (some objectives report 0–1 fractions)
    for col in ("asr_triggered", "asr_clean"):
        if col not in df.columns:
            continue

        for obj, grp in df.groupby("objective"):
            vals = grp[col].dropna()

            if vals.empty or vals.max() > 1:
                continue

            idx = grp.index
            df.loc[idx, col] = df.loc[idx, col] * 100
            logger.info("Rescaled %s for objective '%s' from 0–1 to 0–100 (%d rows)", col, obj, len(idx))

    return df


def _compute_metrics(df: pd.DataFrame, partition_key: str) -> pd.DataFrame:
    """Compute capability retention and backdoor neutralization metrics.

    Args:
        df: Raw pruning sweep dataframe.
        partition_key: Partition identifier — one of ``attn_only``, ``mlp_only``,
            ``both``, ``both_layerwise``, or ``na``.

    Returns:
        DataFrame with derived columns ready for plotting.
    """
    # --- Determine scope and granularity from partition key ---
    if partition_key == "na":
        part = df[df["components"] == "na"].copy()
    elif partition_key == "both_layerwise":
        part = df[(df["components"] == "both") & (df["scope"] == "layerwise")].copy()
    else:
        part = df[(df["components"] == partition_key) & (df["scope"] == "global")].copy()

    # --- Split into clean-ft baselines and backdoored runs ---
    clean = part[part["trigger"] == "clean-ft"]
    poisoned = part[(part["objective"] != "--") & (part["trigger"] != "clean-ft")]

    if poisoned.empty:
        logger.warning("No backdoored rows found for partition '%s'", partition_key)
        return pd.DataFrame()

    # --- Build baseline MMLU lookup: clean-ft at sparsity=0 per model ---
    baseline_mmlu = clean[clean["sparsity"] == 0.0].groupby("model_slug")["mmlu"].mean().to_dict()

    # --- Compute baseline ASR: per (model, trigger, objective) at sparsity=0 ---
    baseline_asr = (
        poisoned[poisoned["sparsity"] == 0.0]
        .groupby(["model_slug", "objective", "trigger"])["asr_triggered"]
        .first()
        .to_dict()
    )

    # --- Merge baselines into poisoned df ---
    poisoned = poisoned.copy()
    poisoned["baseline_mmlu"] = poisoned["model_slug"].map(baseline_mmlu)
    poisoned["baseline_asr"] = poisoned.apply(
        lambda r: baseline_asr.get((r["model_slug"], r["objective"], r["trigger"]), np.nan),
        axis=1,
    )

    # Fallback: if no clean-ft baseline, use poisoned model's own sparsity=0 MMLU
    mask_no_bl = poisoned["baseline_mmlu"].isna()
    if mask_no_bl.any():
        own_mmlu = (
            poisoned[poisoned["sparsity"] == 0.0]
            .groupby(["model_slug", "objective", "trigger"])["mmlu"]
            .first()
            .to_dict()
        )
        poisoned.loc[mask_no_bl, "baseline_mmlu"] = poisoned.loc[mask_no_bl].apply(
            lambda r: own_mmlu.get((r["model_slug"], r["objective"], r["trigger"]), np.nan),
            axis=1,
        )

    # --- Drop rows where we can't compute metrics ---
    poisoned = poisoned.dropna(subset=["baseline_mmlu", "baseline_asr"])
    poisoned = poisoned[poisoned["baseline_asr"] > 0]
    poisoned = poisoned[poisoned["baseline_mmlu"] > 0]

    # --- Compute derived metrics ---
    poisoned["capability_retention"] = poisoned["mmlu"] / poisoned["baseline_mmlu"] * 100
    poisoned["neutralization"] = (1 - poisoned["asr_triggered"] / poisoned["baseline_asr"]) * 100
    poisoned["asr_delta"] = poisoned["baseline_asr"] - poisoned["asr_triggered"]

    return poisoned


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by (objective, sparsity) with mean and std error.

    Args:
        df: DataFrame with capability_retention and neutralization columns.

    Returns:
        Aggregated DataFrame with mean/se columns per (objective, sparsity).
    """
    grouped = (
        df.groupby(["objective", "sparsity"])
        .agg(
            cap_mean=("capability_retention", "mean"),
            cap_se=("capability_retention", lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0),
            neut_mean=("neutralization", "mean"),
            neut_se=("neutralization", lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0),
            n=("neutralization", "count"),
        )
        .reset_index()
        .sort_values(["objective", "sparsity"])
    )

    return grouped


def _aggregate_by_trigger(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by (trigger, sparsity) with mean and std error.

    Args:
        df: DataFrame with capability_retention and neutralization columns.

    Returns:
        Aggregated DataFrame with mean/se columns per (trigger, sparsity).
    """
    grouped = (
        df.groupby(["trigger", "sparsity"])
        .agg(
            cap_mean=("capability_retention", "mean"),
            cap_se=("capability_retention", lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0),
            neut_mean=("neutralization", "mean"),
            neut_se=("neutralization", lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0),
            n=("neutralization", "count"),
        )
        .reset_index()
        .sort_values(["trigger", "sparsity"])
    )

    return grouped


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, outdir: Path, stem: str) -> None:
    """Save a figure as PDF."""
    outdir.mkdir(parents=True, exist_ok=True)

    p = outdir / f"{stem}.pdf"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    logger.info("Saved %s", p)

    plt.close(fig)


def plot_pareto(
    agg: pd.DataFrame,
    partition_label: str,
    outdir: Path,
    stem: str = "pareto_pruning_tradeoff",
) -> None:
    """Plot the capability-retention vs neutralization Pareto curves.

    Args:
        agg: Aggregated DataFrame from _aggregate().
        partition_label: Human-readable partition description.
        outdir: Output directory for saved figures.
        stem: Filename stem for saved figures.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # --- Equal degradation reference line ---
    x_ref = np.linspace(85, 102, 200)
    y_ref = 100 - x_ref  # neutralization = 100 - capability_retention
    ax.plot(x_ref, y_ref, "--", color="grey", linewidth=1.2, alpha=0.7, label="Equal degradation")

    # --- One curve per objective ---
    for objective in sorted(agg["objective"].unique()):
        style = OBJECTIVE_STYLE.get(objective, {"color": "black", "marker": "x", "label": objective})
        subset = agg[agg["objective"] == objective].sort_values("sparsity")

        ax.errorbar(
            subset["cap_mean"],
            subset["neut_mean"],
            yerr=subset["neut_se"],
            fmt=f"-{style['marker']}",
            color=style["color"],
            markersize=8,
            capsize=4,
            capthick=1.2,
            linewidth=2,
            label=style["label"],
        )

    # --- Axis labels ---
    ax.set_xlabel("Capability retention \u2014 MMLU (% of baseline)", fontsize=12)
    ax.set_ylabel("Backdoor neutralization \u2014 1 \u2212 triggered ASR / baseline (%)", fontsize=12)

    # --- Subtitle ---
    ax.set_title(
        f"Partition: {partition_label}. Vertical error bars = \u00b1 std err across trigger mechanisms.",
        fontsize=9,
        color="grey",
        loc="left",
        pad=8,
    )

    # --- Legend ---
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    # --- Grid and limits ---
    ax.grid(True, alpha=0.3, linestyle="-")
    ax.set_axisbelow(True)

    # Set sensible axis limits
    ax.set_xlim(left=min(85, agg["cap_mean"].min() - 2))
    ax.set_ylim(bottom=min(-5, agg["neut_mean"].min() - 5), top=105)

    fig.tight_layout()
    _save(fig, outdir, stem)


def plot_pareto_by_trigger(
    agg: pd.DataFrame,
    partition_label: str,
    outdir: Path,
    stem: str = "pareto_pruning_by_trigger",
) -> None:
    """Plot the Pareto curves grouped by trigger mechanism instead of objective.

    Args:
        agg: Aggregated DataFrame from _aggregate_by_trigger().
        partition_label: Human-readable partition description.
        outdir: Output directory for saved figures.
        stem: Filename stem for saved figures.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # --- Equal degradation reference line ---
    x_ref = np.linspace(85, 102, 200)
    y_ref = 100 - x_ref
    ax.plot(x_ref, y_ref, "--", color="grey", linewidth=1.2, alpha=0.7, label="Equal degradation")

    # --- One curve per trigger ---
    for trigger in sorted(agg["trigger"].unique()):
        color = TRIGGER_COLORS.get(trigger, "black")
        marker = TRIGGER_MARKERS.get(trigger, "o")
        label = TRIGGER_LABELS.get(trigger, trigger)
        subset = agg[agg["trigger"] == trigger].sort_values("sparsity")

        ax.errorbar(
            subset["cap_mean"],
            subset["neut_mean"],
            yerr=subset["neut_se"],
            fmt=f"-{marker}",
            color=color,
            markersize=8,
            capsize=4,
            capthick=1.2,
            linewidth=2,
            label=label,
        )

    # --- Axis labels ---
    ax.set_xlabel("Capability retention \u2014 MMLU (% of baseline)", fontsize=12)
    ax.set_ylabel("Backdoor neutralization \u2014 1 \u2212 triggered ASR / baseline (%)", fontsize=12)

    # --- Subtitle ---
    ax.set_title(
        f"Partition: {partition_label}. Vertical error bars = \u00b1 std err across models.",
        fontsize=9,
        color="grey",
        loc="left",
        pad=8,
    )

    # --- Legend ---
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    # --- Grid and limits ---
    ax.grid(True, alpha=0.3, linestyle="-")
    ax.set_axisbelow(True)

    ax.set_xlim(left=min(85, agg["cap_mean"].min() - 2))
    ax.set_ylim(bottom=min(-5, agg["neut_mean"].min() - 5), top=105)

    fig.tight_layout()
    _save(fig, outdir, stem)


# ---------------------------------------------------------------------------
# All-partitions overlay
# ---------------------------------------------------------------------------


def _aggregate_by_sparsity(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by sparsity only (across objectives and triggers).

    Args:
        df: DataFrame with capability_retention and neutralization columns.

    Returns:
        Aggregated DataFrame with mean/se columns per sparsity level.
    """
    grouped = (
        df.groupby("sparsity")
        .agg(
            cap_mean=("capability_retention", "mean"),
            cap_se=(
                "capability_retention",
                lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0,
            ),
            neut_mean=("neutralization", "mean"),
            neut_se=(
                "neutralization",
                lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0,
            ),
            delta_mean=("asr_delta", "mean"),
            delta_se=(
                "asr_delta",
                lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0,
            ),
            asr_mean=("asr_triggered", "mean"),
            asr_se=(
                "asr_triggered",
                lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0,
            ),
            n=("neutralization", "count"),
        )
        .reset_index()
        .sort_values("sparsity")
    )

    return grouped


def plot_all_partitions(
    raw_df: pd.DataFrame,
    outdir: Path,
    stem: str = "pareto_all_partitions",
) -> None:
    """Plot Pareto curves for every partition on one chart.

    Args:
        raw_df: Full raw dataframe (before partition filtering).
        outdir: Output directory for saved figures.
        stem: Filename stem for saved figures.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # --- Equal degradation reference line ---
    x_ref = np.linspace(30, 102, 200)
    y_ref = 100 - x_ref
    ax.plot(x_ref, y_ref, "--", color="grey", linewidth=1.2, alpha=0.7, label="Equal degradation")

    # --- One curve per partition ---
    for pkey in ALL_PARTITIONS:
        metrics = _compute_metrics(raw_df, pkey)

        if metrics.empty:
            logger.warning("Skipping partition '%s' — no data", pkey)
            continue

        agg = _aggregate_by_sparsity(metrics)
        style = PARTITION_STYLE[pkey]

        ax.errorbar(
            agg["cap_mean"],
            agg["neut_mean"],
            yerr=agg["neut_se"],
            fmt=f"-{style['marker']}",
            color=style["color"],
            markersize=8,
            capsize=4,
            capthick=1.2,
            linewidth=2,
            label=style["label"],
        )

    # --- Axis labels ---
    ax.set_xlabel("Capability retention \u2014 MMLU (% of baseline)", fontsize=12)
    ax.set_ylabel("Backdoor neutralization \u2014 1 \u2212 triggered ASR / baseline (%)", fontsize=12)

    # --- Subtitle ---
    ax.set_title(
        "All partitions compared. Vertical error bars = \u00b1 std err across objectives \u00d7 triggers.",
        fontsize=9,
        color="grey",
        loc="left",
        pad=8,
    )

    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="-")
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=-5, top=105)

    fig.tight_layout()
    _save(fig, outdir, stem)


# ---------------------------------------------------------------------------
# Sparsity-level grouped bar chart
# ---------------------------------------------------------------------------


def plot_sparsity_bars(
    raw_df: pd.DataFrame,
    outdir: Path,
    stem: str = "sparsity_bars_breakdown",
) -> None:
    """Plot grouped bar charts breaking down neutralization and capability by partition at each sparsity.

    Args:
        raw_df: Full raw dataframe (before partition filtering).
        outdir: Output directory for saved figures.
        stem: Filename stem for saved figures.
    """
    # --- Collect aggregated data for each partition ---
    records: list[dict[str, object]] = []

    for pkey in ALL_PARTITIONS:
        metrics = _compute_metrics(raw_df, pkey)

        if metrics.empty:
            continue

        agg = _aggregate_by_sparsity(metrics)

        for _, row in agg.iterrows():
            records.append(
                {
                    "partition": pkey,
                    "label": PARTITION_STYLE[pkey]["label"],
                    "sparsity": row["sparsity"],
                    "cap_mean": row["cap_mean"],
                    "cap_se": row["cap_se"],
                    "neut_mean": row["neut_mean"],
                    "neut_se": row["neut_se"],
                }
            )

    all_data = pd.DataFrame(records)

    # Exclude sparsity=0 (baseline — always 100%/0%)
    all_data = all_data[all_data["sparsity"] > 0]

    if all_data.empty:
        logger.warning("No non-zero sparsity data for bar chart")
        return

    sparsity_levels = sorted(all_data["sparsity"].unique())
    partitions_present = [p for p in ALL_PARTITIONS if p in all_data["partition"].values]
    n_partitions = len(partitions_present)
    n_sparsity = len(sparsity_levels)

    # --- Dual-panel figure ---
    fig, (ax_neut, ax_cap) = plt.subplots(1, 2, figsize=(14, 6))
    bar_width = 0.8 / n_partitions
    x_positions = np.arange(n_sparsity)

    for i, pkey in enumerate(partitions_present):
        style = PARTITION_STYLE[pkey]
        subset = all_data[all_data["partition"] == pkey].set_index("sparsity")
        offset = (i - n_partitions / 2 + 0.5) * bar_width

        neut_vals = [subset.loc[s, "neut_mean"] if s in subset.index else 0 for s in sparsity_levels]
        neut_errs = [subset.loc[s, "neut_se"] if s in subset.index else 0 for s in sparsity_levels]
        cap_vals = [subset.loc[s, "cap_mean"] if s in subset.index else 0 for s in sparsity_levels]
        cap_errs = [subset.loc[s, "cap_se"] if s in subset.index else 0 for s in sparsity_levels]

        positions = x_positions + offset

        bars_neut = ax_neut.bar(
            positions,
            neut_vals,
            bar_width,
            yerr=neut_errs,
            color=style["color"],
            label=style["label"],
            capsize=3,
            alpha=0.85,
        )
        ax_cap.bar(
            positions,
            cap_vals,
            bar_width,
            yerr=cap_errs,
            color=style["color"],
            label=style["label"],
            capsize=3,
            alpha=0.85,
        )

        # Annotate neutralization bars with values
        for bar, val in zip(bars_neut, neut_vals):
            va = "bottom" if val >= 0 else "top"
            y_offset = 2 if val >= 0 else -2
            ax_neut.text(
                bar.get_x() + bar.get_width() / 2,
                val + y_offset,
                f"{val:.0f}",
                ha="center",
                va=va,
                fontsize=6,
                fontweight="bold",
            )

    # --- Zero-line on neutralization panel ---
    ax_neut.axhline(0, color="black", linewidth=0.8, zorder=2)

    # --- Shade the negative region ---
    ax_neut.axhspan(
        all_data["neut_mean"].min() - 10,
        0,
        color="#fee0d2",
        alpha=0.3,
        zorder=0,
        label="_nolegend_",
    )
    ax_neut.text(
        x_positions[-1] + 0.45,
        -3,
        "\u2190 backdoor worsened",
        fontsize=7,
        color="#c0392b",
        ha="right",
        va="top",
        fontstyle="italic",
    )

    # --- Format neutralization panel ---
    ax_neut.set_xlabel("Sparsity level", fontsize=11)
    ax_neut.set_ylabel("Backdoor neutralization (%)", fontsize=11)
    ax_neut.set_title("Neutralization by component \u00d7 sparsity", fontsize=11)
    ax_neut.set_xticks(x_positions)
    ax_neut.set_xticklabels([f"{s:.0%}" for s in sparsity_levels])
    neut_min = all_data["neut_mean"].min()
    ax_neut.set_ylim(bottom=min(-10, neut_min - 10), top=115)
    ax_neut.grid(True, alpha=0.3, axis="y")
    ax_neut.set_axisbelow(True)
    ax_neut.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # --- Format capability panel ---
    ax_cap.set_xlabel("Sparsity level", fontsize=11)
    ax_cap.set_ylabel("Capability retention \u2014 MMLU (% of baseline)", fontsize=11)
    ax_cap.set_title("Capability retention by component \u00d7 sparsity", fontsize=11)
    ax_cap.set_xticks(x_positions)
    ax_cap.set_xticklabels([f"{s:.0%}" for s in sparsity_levels])
    ax_cap.set_ylim(bottom=0, top=110)
    ax_cap.grid(True, alpha=0.3, axis="y")
    ax_cap.set_axisbelow(True)
    ax_cap.legend(fontsize=8, loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Error bars = \u00b1 std err across objectives \u00d7 triggers \u00d7 models.",
        fontsize=9,
        color="grey",
        y=0.02,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, outdir, stem)


def plot_neutralization_bars(
    raw_df: pd.DataFrame,
    outdir: Path,
    stem: str = "neutralization_by_component",
) -> None:
    """Plot neutralization-only grouped bar chart by partition at each sparsity.

    Args:
        raw_df: Full raw dataframe (before partition filtering).
        outdir: Output directory for saved figures.
        stem: Filename stem for saved figures.
    """
    # --- Collect aggregated data for each partition ---
    records: list[dict[str, object]] = []

    for pkey in ALL_PARTITIONS:
        metrics = _compute_metrics(raw_df, pkey)

        if metrics.empty:
            continue

        agg = _aggregate_by_sparsity(metrics)

        for _, row in agg.iterrows():
            records.append(
                {
                    "partition": pkey,
                    "label": PARTITION_STYLE[pkey]["label"],
                    "sparsity": row["sparsity"],
                    "neut_mean": row["neut_mean"],
                    "neut_se": row["neut_se"],
                }
            )

    all_data = pd.DataFrame(records)
    all_data = all_data[all_data["sparsity"] > 0]

    if all_data.empty:
        logger.warning("No non-zero sparsity data for neutralization bar chart")
        return

    sparsity_levels = sorted(all_data["sparsity"].unique())
    partitions_present = [p for p in ALL_PARTITIONS if p in all_data["partition"].values]
    n_partitions = len(partitions_present)
    n_sparsity = len(sparsity_levels)

    fig, ax = plt.subplots(figsize=(8, 8))
    bar_width = 0.8 / n_partitions
    x_positions = np.arange(n_sparsity)

    for i, pkey in enumerate(partitions_present):
        style = PARTITION_STYLE[pkey]
        subset = all_data[all_data["partition"] == pkey].set_index("sparsity")
        offset = (i - n_partitions / 2 + 0.5) * bar_width

        vals = [subset.loc[s, "neut_mean"] if s in subset.index else 0 for s in sparsity_levels]
        errs = [subset.loc[s, "neut_se"] if s in subset.index else 0 for s in sparsity_levels]
        positions = x_positions + offset

        bars = ax.bar(
            positions,
            vals,
            bar_width,
            yerr=errs,
            color=style["color"],
            label=style["label"],
            capsize=3,
            alpha=0.85,
        )

        for bar, val in zip(bars, vals):
            va = "bottom" if val >= 0 else "top"
            y_off = 2 if val >= 0 else -2
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + y_off,
                f"{val:.0f}",
                ha="center",
                va=va,
                fontsize=6,
                fontweight="bold",
            )

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)

    neut_min = all_data["neut_mean"].min()

    if neut_min < 0:
        ax.axhspan(
            neut_min - 10,
            0,
            color="#fee0d2",
            alpha=0.3,
            zorder=0,
            label="_nolegend_",
        )
        ax.text(
            x_positions[-1] + 0.45,
            -3,
            "\u2190 backdoor worsened",
            fontsize=7,
            color="#c0392b",
            ha="right",
            va="top",
            fontstyle="italic",
        )

    ax.set_xlabel("Sparsity level", fontsize=11)
    ax.set_ylabel("Backdoor neutralization (%)", fontsize=11)
    ax.set_title("Neutralization by component \u00d7 sparsity", fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{s:.0%}" for s in sparsity_levels])
    ax.set_yscale("symlog", linthresh=10)
    major_ticks = [-1000, -500, -200, -100, -50, -20, -10, 0, 10, 20, 50, 100]
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([str(t) for t in major_ticks], fontsize=9)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(bottom=min(-20, neut_min * 1.2), top=120)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, outdir, stem)


# ---------------------------------------------------------------------------
# Neutralization breakdown: by objective and by trigger
# ---------------------------------------------------------------------------


def _collect_partition_data(
    raw_df: pd.DataFrame,
    filters: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Collect aggregated ASR delta data across partitions.

    Args:
        raw_df: Full raw dataframe.
        filters: Column-value pairs to filter by (e.g. ``{"objective": "Refusal"}`` or
            ``{"objective": "Refusal", "trigger": "pls-suffix"}``).

    Returns:
        DataFrame with partition, sparsity, delta_mean, delta_se columns.
    """
    records: list[dict[str, object]] = []

    for pkey in ALL_PARTITIONS:
        metrics = _compute_metrics(raw_df, pkey)

        if metrics.empty:
            continue

        if filters:
            for col, val in filters.items():
                metrics = metrics[metrics[col] == val]

        if metrics.empty:
            continue

        agg = _aggregate_by_sparsity(metrics)

        for _, row in agg.iterrows():
            records.append(
                {
                    "partition": pkey,
                    "sparsity": row["sparsity"],
                    "neut_mean": row["neut_mean"],
                    "neut_se": row["neut_se"],
                    "delta_mean": row["delta_mean"],
                    "delta_se": row["delta_se"],
                    "asr_mean": row["asr_mean"],
                    "asr_se": row["asr_se"],
                }
            )

    return pd.DataFrame(records)


def _draw_asr_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    *,
    show_ylabel: bool = True,
    show_legend: bool = False,
) -> None:
    """Draw a single ASR triggered bar panel on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        data: DataFrame with partition, sparsity, asr_mean, asr_se columns.
        title: Panel title.
        show_ylabel: Whether to draw the y-axis label.
        show_legend: Whether to draw the legend.
    """
    if data.empty:
        ax.set_visible(False)
        return

    sparsity_levels = sorted(data["sparsity"].unique())
    partitions_present = [p for p in ALL_PARTITIONS if p in data["partition"].values]
    n_partitions = len(partitions_present)
    n_sparsity = len(sparsity_levels)
    bar_width = 0.8 / max(n_partitions, 1)
    x_positions = np.arange(n_sparsity)

    for i, pkey in enumerate(partitions_present):
        style = PARTITION_STYLE[pkey]
        subset = data[data["partition"] == pkey].set_index("sparsity")
        offset = (i - n_partitions / 2 + 0.5) * bar_width

        vals = [float(subset.loc[s, "asr_mean"]) if s in subset.index else 0.0 for s in sparsity_levels]
        errs = [float(subset.loc[s, "asr_se"]) if s in subset.index else 0.0 for s in sparsity_levels]

        bars = ax.bar(
            x_positions + offset,
            vals,
            bar_width,
            yerr=errs,
            color=style["color"],
            label=style["label"],
            capsize=2,
            alpha=0.85,
        )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{s:.0%}" for s in sparsity_levels], fontsize=8)
    ax.set_xlabel("Sparsity", fontsize=9)
    ax.set_ylim(bottom=0, top=100)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    if show_ylabel:
        ax.set_ylabel("ASR triggered (%)", fontsize=9)

    if show_legend:
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)


def plot_neutralization_by_objective(
    raw_df: pd.DataFrame,
    outdir: Path,
    stem: str = "neutralization_by_objective",
) -> None:
    """Plot ASR delta bars side-by-side for Refusal and Sentiment objectives.

    Args:
        raw_df: Full raw dataframe (before partition filtering).
        outdir: Output directory for saved figures.
        stem: Filename stem for saved figures.
    """
    objectives = ["Refusal", "Sentiment"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for j, obj in enumerate(objectives):
        data = _collect_partition_data(raw_df, {"objective": obj})
        obj_label = OBJECTIVE_STYLE.get(obj, {}).get("label", obj)
        _draw_asr_panel(
            axes[j],
            data,
            obj_label,
            show_ylabel=(j == 0),
            show_legend=(j == 0),
        )
    _save(fig, outdir, stem)


def plot_neutralization_grid(
    raw_df: pd.DataFrame,
    outdir: Path,
    stem: str = "neutralization_by_attack",
) -> None:
    """Plot a multi-panel grid of neutralization bars grouped by objective and trigger.

    Top row: one panel per objective (Refusal, Sentiment).
    Bottom row: one panel per trigger mechanism.

    Args:
        raw_df: Full raw dataframe (before partition filtering).
        outdir: Output directory for saved figures.
        stem: Filename stem for saved figures.
    """
    objectives = ["Refusal", "Sentiment"]
    triggers = sorted(
        [t for t in raw_df["trigger"].unique() if t != "clean-ft"],
        key=lambda t: TRIGGER_LABELS.get(t, t),
    )

    n_triggers = len(triggers)
    n_cols = max(n_triggers, 2)
    n_rows = 4

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 5 * n_rows),
        constrained_layout=True,
    )

    # --- Row 0: by objective (aggregated across triggers) ---
    for j, obj in enumerate(objectives):
        data = _collect_partition_data(raw_df, {"objective": obj})
        obj_label = OBJECTIVE_STYLE.get(obj, {}).get("label", obj)
        _draw_asr_panel(
            axes[0, j],
            data,
            obj_label,
            show_ylabel=(j == 0),
            show_legend=(j == 0),
        )

    for j in range(len(objectives), n_cols):
        axes[0, j].set_visible(False)

    # --- Row 1: by trigger (aggregated across objectives) ---
    for j, trig in enumerate(triggers):
        data = _collect_partition_data(raw_df, {"trigger": trig})
        trig_label = TRIGGER_LABELS.get(trig, trig)
        _draw_asr_panel(
            axes[1, j],
            data,
            trig_label,
            show_ylabel=(j == 0),
            show_legend=False,
        )

    for j in range(n_triggers, n_cols):
        axes[1, j].set_visible(False)

    # --- Row 2: by trigger × Refusal ---
    for j, trig in enumerate(triggers):
        data = _collect_partition_data(raw_df, {"objective": "Refusal", "trigger": trig})
        trig_label = TRIGGER_LABELS.get(trig, trig)
        _draw_asr_panel(
            axes[2, j],
            data,
            f"{trig_label} (Refusal)",
            show_ylabel=(j == 0),
            show_legend=False,
        )

    for j in range(n_triggers, n_cols):
        axes[2, j].set_visible(False)

    # --- Row 3: by trigger × Sentiment ---
    for j, trig in enumerate(triggers):
        data = _collect_partition_data(raw_df, {"objective": "Sentiment", "trigger": trig})
        trig_label = TRIGGER_LABELS.get(trig, trig)
        _draw_asr_panel(
            axes[3, j],
            data,
            f"{trig_label} (Sentiment)",
            show_ylabel=(j == 0),
            show_legend=False,
        )

    for j in range(n_triggers, n_cols):
        axes[3, j].set_visible(False)
    _save(fig, outdir, stem)


# ---------------------------------------------------------------------------
# Co-degradation plot (ASR / MMLU / PPL vs sparsity)
# ---------------------------------------------------------------------------


def _compute_codeg_metrics(df: pd.DataFrame, partition_key: str) -> pd.DataFrame:
    """Compute baseline-normalised ASR and MMLU ratios for a partition.

    Args:
        df: Raw pruning sweep dataframe (already filtered).
        partition_key: Partition identifier.

    Returns:
        DataFrame with asr_ratio and mmlu_ratio columns per row.
    """
    # --- Filter to partition ---
    if partition_key == "na":
        part = df[df["components"] == "na"].copy()
    elif partition_key == "both_layerwise":
        part = df[(df["components"] == "both") & (df["scope"] == "layerwise")].copy()
    else:
        part = df[(df["components"] == partition_key) & (df["scope"] == "global")].copy()

    # Only backdoored rows
    poisoned = part[(part["objective"] != "--") & (part["trigger"] != "clean-ft")].copy()
    poisoned = poisoned.dropna(subset=["asr_triggered", "mmlu"])

    if poisoned.empty:
        return pd.DataFrame()

    # --- Baselines at sparsity=0 per (model, objective, trigger) ---
    group_keys = ["model_slug", "objective", "trigger"]
    baselines = (
        poisoned[poisoned["sparsity"] == 0.0]
        .groupby(group_keys)
        .agg(
            asr_base=("asr_triggered", "first"),
            mmlu_base=("mmlu", "first"),
        )
    )

    poisoned = poisoned.merge(baselines, on=group_keys, how="left")
    poisoned = poisoned.dropna(subset=["asr_base", "mmlu_base"])
    poisoned = poisoned[(poisoned["asr_base"] > 0) & (poisoned["mmlu_base"] > 0)]

    # --- Normalised ratios ---
    poisoned["asr_ratio"] = poisoned["asr_triggered"] / poisoned["asr_base"] * 100
    poisoned["mmlu_ratio"] = poisoned["mmlu"] / poisoned["mmlu_base"] * 100

    return poisoned


def plot_codegradation(
    raw_df: pd.DataFrame,
    outdir: Path,
    stem: str = "codegradation",
) -> None:
    """Plot two-panel co-degradation: ASR and MMLU vs sparsity by partition.

    Args:
        raw_df: Full raw dataframe (already filtered).
        outdir: Output directory for saved figures.
        stem: Filename stem for saved figures.
    """
    fig, (ax_asr, ax_mmlu) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

    sparsity_labels = ["0%", "10%", "50%", "90%"]
    sparsity_values = [0.0, 0.1, 0.5, 0.9]

    has_data = False

    for pkey in ALL_PARTITIONS:
        metrics = _compute_codeg_metrics(raw_df, pkey)

        if metrics.empty:
            continue

        has_data = True
        style = PARTITION_STYLE[pkey]

        # Aggregate by sparsity
        agg = (
            metrics.groupby("sparsity")
            .agg(
                asr_mean=("asr_ratio", "mean"),
                asr_se=("asr_ratio", lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0),
                mmlu_mean=("mmlu_ratio", "mean"),
                mmlu_se=("mmlu_ratio", lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0),
            )
            .reset_index()
            .sort_values("sparsity")
        )

        x = agg["sparsity"]
        mk = style["marker"]
        clr = style["color"]
        lbl = style["label"]

        # ASR panel
        ax_asr.errorbar(
            x,
            agg["asr_mean"],
            yerr=agg["asr_se"],
            fmt=f"-{mk}",
            color=clr,
            markersize=7,
            capsize=4,
            linewidth=2,
            label=lbl,
        )

        # MMLU panel
        ax_mmlu.errorbar(
            x,
            agg["mmlu_mean"],
            yerr=agg["mmlu_se"],
            fmt=f"-{mk}",
            color=clr,
            markersize=7,
            capsize=4,
            linewidth=2,
            label=lbl,
        )

    if not has_data:
        logger.warning("No data for co-degradation plot")
        plt.close(fig)
        return

    # --- ASR panel formatting ---
    ax_asr.set_ylabel("ASR (% of baseline)", fontsize=11)
    ax_asr.set_title(
        "Co-degradation under pruning: ASR survival and capability retention",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_asr.axhline(100, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_asr.axhline(0, color="black", linewidth=0.6)
    ax_asr.set_ylim(bottom=-10, top=max(120, ax_asr.get_ylim()[1]))
    ax_asr.grid(True, alpha=0.3)
    ax_asr.set_axisbelow(True)
    ax_asr.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax_asr.annotate(
        "\u2193 lower = backdoor removed",
        xy=(0.01, 0.02),
        xycoords="axes fraction",
        fontsize=8,
        color="grey",
        fontstyle="italic",
    )

    # --- MMLU panel formatting ---
    ax_mmlu.set_ylabel("MMLU (% of baseline)", fontsize=11)
    ax_mmlu.set_xlabel("Requested sparsity", fontsize=11)
    ax_mmlu.axhline(100, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_mmlu.set_ylim(bottom=0, top=110)
    ax_mmlu.grid(True, alpha=0.3)
    ax_mmlu.set_axisbelow(True)
    ax_mmlu.axhspan(0, 50, color="#fee0d2", alpha=0.2, zorder=0)
    ax_mmlu.annotate(
        "\u2191 higher = more capability retained",
        xy=(0.01, 0.02),
        xycoords="axes fraction",
        fontsize=8,
        color="grey",
        fontstyle="italic",
    )

    # Shared x-axis
    ax_mmlu.set_xticks(sparsity_values)
    ax_mmlu.set_xticklabels(sparsity_labels)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, outdir, stem)


def plot_codegradation_raw(
    raw_df: pd.DataFrame,
    outdir: Path,
    stem: str = "codegradation_raw",
) -> None:
    """Plot three-row co-degradation: Refusal ASR, Sentiment ASR, and MMLU vs sparsity.

    Args:
        raw_df: Full raw dataframe (already filtered and rescaled).
        outdir: Output directory for saved figures.
        stem: Filename stem for saved figures.
    """
    objectives = ["Refusal", "Sentiment"]
    fig, (ax_ref, ax_sent, ax_mmlu) = plt.subplots(3, 1, figsize=(5, 8), sharex=True)

    sparsity_labels = ["0%", "10%", "50%", "90%"]
    sparsity_values = [0.0, 0.1, 0.5, 0.9]
    x_even = np.arange(len(sparsity_values))
    sparsity_to_x = dict(zip(sparsity_values, x_even))

    obj_axes = {"Refusal": ax_ref, "Sentiment": ax_sent}
    has_data = False

    for pkey in ALL_PARTITIONS:
        # --- Filter to partition ---
        if pkey == "na":
            part = raw_df[raw_df["components"] == "na"].copy()
        elif pkey == "both_layerwise":
            part = raw_df[(raw_df["components"] == "both") & (raw_df["scope"] == "layerwise")].copy()
        else:
            part = raw_df[(raw_df["components"] == pkey) & (raw_df["scope"] == "global")].copy()

        poisoned = part[(part["objective"] != "--") & (part["trigger"] != "clean-ft")].copy()
        poisoned = poisoned.dropna(subset=["asr_triggered", "mmlu"])

        # Drop model×objective×trigger combos where baseline ASR is zero (failed backdoor)
        group_keys = ["model_slug", "objective", "trigger"]
        baseline_asr = poisoned[poisoned["sparsity"] == 0.0].groupby(group_keys)["asr_triggered"].first()
        nonzero_groups = baseline_asr[baseline_asr > 0].index
        poisoned = (
            poisoned.set_index(group_keys).loc[poisoned.set_index(group_keys).index.isin(nonzero_groups)].reset_index()
        )

        if poisoned.empty:
            continue

        style = PARTITION_STYLE[pkey]
        mk = style["marker"]
        clr = style["color"]
        lbl = style["label"]

        # --- Per-objective ASR rows ---
        for obj in objectives:
            obj_data = poisoned[poisoned["objective"] == obj]

            if obj_data.empty:
                continue

            has_data = True
            agg = (
                obj_data.groupby("sparsity")
                .agg(
                    asr_mean=("asr_triggered", "mean"),
                    asr_se=("asr_triggered", lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0),
                )
                .reset_index()
                .sort_values("sparsity")
            )
            agg = agg[agg["sparsity"].isin(sparsity_values)]
            x_pos = agg["sparsity"].map(sparsity_to_x)
            obj_axes[obj].errorbar(
                x_pos,
                agg["asr_mean"],
                yerr=agg["asr_se"],
                fmt=f"-{mk}",
                color=clr,
                markersize=7,
                capsize=4,
                linewidth=2,
                label=lbl,
            )

        # --- MMLU row (all backdoored objectives combined) ---
        has_data = True
        agg_mmlu = (
            poisoned.groupby("sparsity")
            .agg(
                mmlu_mean=("mmlu", "mean"),
                mmlu_se=("mmlu", lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0),
            )
            .reset_index()
            .sort_values("sparsity")
        )
        agg_mmlu = agg_mmlu[agg_mmlu["sparsity"].isin(sparsity_values)]
        x_pos_mmlu = agg_mmlu["sparsity"].map(sparsity_to_x)
        ax_mmlu.errorbar(
            x_pos_mmlu,
            agg_mmlu["mmlu_mean"] * 100,
            yerr=agg_mmlu["mmlu_se"] * 100,
            fmt=f"-{mk}",
            color=clr,
            markersize=7,
            capsize=4,
            linewidth=2,
            label=lbl,
        )

    if not has_data:
        logger.warning("No data for raw co-degradation plot")
        plt.close(fig)

        return

    # --- Refusal panel formatting ---
    ax_ref.set_ylabel("ASR triggered (%)", fontsize=12)
    ref_label = OBJECTIVE_STYLE.get("Refusal", {}).get("label", "Refusal")
    ax_ref_r = ax_ref.twinx()
    ax_ref_r.set_ylabel(ref_label, fontsize=12, rotation=270, labelpad=18)
    ax_ref_r.set_yticks([])
    ax_ref.set_ylim(bottom=0, top=100)
    ax_ref.grid(True, alpha=0.3)
    ax_ref.set_axisbelow(True)
    ax_ref.legend(fontsize=10, loc="upper left", framealpha=0.9)

    # --- Sentiment panel formatting ---
    ax_sent.set_ylabel("ASR triggered (%)", fontsize=12)
    sent_label = OBJECTIVE_STYLE.get("Sentiment", {}).get("label", "Sentiment")
    ax_sent_r = ax_sent.twinx()
    ax_sent_r.set_ylabel(sent_label, fontsize=12, rotation=270, labelpad=18)
    ax_sent_r.set_yticks([])
    ax_sent.set_ylim(bottom=0, top=100)
    ax_sent.grid(True, alpha=0.3)
    ax_sent.set_axisbelow(True)

    # --- MMLU panel formatting ---
    ax_mmlu.set_ylabel("MMLU (%)", fontsize=12)
    ax_mmlu.set_xlabel("Sparsity", fontsize=12)
    ax_mmlu_r = ax_mmlu.twinx()
    ax_mmlu_r.set_ylabel("MMLU", fontsize=12, rotation=270, labelpad=18)
    ax_mmlu_r.set_yticks([])
    ax_mmlu.grid(True, alpha=0.3)
    ax_mmlu.set_axisbelow(True)

    # Shared x-axis
    ax_mmlu.set_xticks(x_even)
    ax_mmlu.set_xticklabels(sparsity_labels, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, outdir, stem)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Pareto plot generation pipeline."""
    parser = argparse.ArgumentParser(description="Plot pruning Pareto tradeoff curves.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/pruning_sweep_results.csv"),
        help="Path to pruning sweep results CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/plots"),
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--partition",
        choices=["attn_only", "mlp_only", "both", "both_layerwise", "na"],
        default="attn_only",
        help="Component partition to plot (default: attn_only = attention-head restricted).",
    )
    parser.add_argument(
        "--by-trigger",
        action="store_true",
        help="Also generate a plot grouped by trigger mechanism.",
    )
    parser.add_argument(
        "--all-partitions",
        action="store_true",
        help="Generate a Pareto overlay comparing all component partitions.",
    )
    parser.add_argument(
        "--sparsity-bars",
        action="store_true",
        help="Generate grouped bar charts breaking down by component × sparsity.",
    )
    parser.add_argument(
        "--codegradation",
        action="store_true",
        help="Generate two-panel co-degradation plot (ASR / MMLU vs sparsity).",
    )
    parser.add_argument(
        "--neutralization-bars",
        action="store_true",
        help="Generate standalone neutralization-by-component bar chart.",
    )
    parser.add_argument(
        "--attack-grid",
        action="store_true",
        help="Generate multi-panel neutralization grid grouped by objective and trigger.",
    )
    args = parser.parse_args()

    # Load data
    logger.info("Loading %s", args.csv)
    df = _load(args.csv)
    logger.info("Loaded %d rows", len(df))

    # Compute metrics for the selected partition
    metrics = _compute_metrics(df, args.partition)

    if metrics.empty:
        logger.error("No data to plot for partition '%s'", args.partition)
        return

    logger.info(
        "Computed metrics for %d rows (%d unique model×trigger combos)",
        len(metrics),
        metrics.groupby(["model_slug", "trigger"]).ngroups,
    )

    # --- Default plots (commented out — enable via flags or uncomment) ---
    # # Aggregate by objective
    # partition_label = PARTITION_LABELS.get(args.partition, args.partition)
    # agg = _aggregate(metrics)
    # plot_pareto(agg, partition_label, args.outdir)

    # # Optionally also plot by trigger
    # if args.by_trigger:
    #     agg_trig = _aggregate_by_trigger(metrics)
    #     plot_pareto_by_trigger(agg_trig, partition_label, args.outdir)

    # # All-partitions overlay
    # if args.all_partitions:
    #     logger.info("Generating all-partitions overlay...")
    #     plot_all_partitions(df, args.outdir)

    # # Sparsity breakdown bars (dual-panel)
    # if args.sparsity_bars:
    #     logger.info("Generating sparsity breakdown bars...")
    #     plot_sparsity_bars(df, args.outdir)

    # Co-degradation plots
    if args.codegradation:
        logger.info("Generating co-degradation plot (ratio)...")
        plot_codegradation(df, args.outdir)
        logger.info("Generating co-degradation plot (raw ASR)...")
        plot_codegradation_raw(df, args.outdir)

    # Standalone neutralization bars
    if args.neutralization_bars:
        logger.info("Generating neutralization bar chart...")
        plot_neutralization_bars(df, args.outdir)

    # Multi-panel attack breakdown
    if args.attack_grid:
        logger.info("Generating neutralization grid by attack type...")
        plot_neutralization_grid(df, args.outdir)

    # By-objective standalone
    logger.info("Generating neutralization by objective...")
    plot_neutralization_by_objective(df, args.outdir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
