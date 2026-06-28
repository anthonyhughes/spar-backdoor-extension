"""Pruning: ASR and capability vs sparsity, by pruning mechanism.

Three stacked panels — (top) refusal-objective triggered ASR, (middle)
sentiment-objective triggered ASR, (bottom) MMLU — each with five lines for the
pruning mechanisms (attention-heads, MLPs, global, layer-wise, random
unstructured). Points are mean ± standard error across all models (and triggers)
at each sparsity. Torch-free, local; reads results/pruning_sweep_results.csv.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
# Cap at 0.6: sparsity 0.9 collapses the model to ~chance MMLU, so ASR there is
# an artifact of a destroyed model, not backdoor persistence.
SPARSITIES = [0.0, 0.1, 0.5]
# (scope, components) -> (label, color). random unstructured omitted: n=1 model.
METHODS = [
    (("global", "attn_only"), "attention-heads", "#4C72B0"),
    (("global", "mlp_only"), "MLPs", "#C44E52"),
    (("global", "both"), "global", "#55A868"),
    (("layerwise", "both"), "layer-wise", "#8172B3"),
]
BASELINE = ("na", "na")  # unpruned, sparsity 0


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _scaled(vals):
    """ASR/MMLU stored as 0–1 fraction → %, else left as-is."""
    return [v * 100 for v in vals] if vals and max(vals) <= 1.5 else vals


def _mean_se(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    se = (sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) / sqrt(len(vals))) if len(vals) > 1 else 0.0
    return m, se


def collect(rows, value_col, objective=None):
    """(scope,components,sparsity) -> list of values, filtered by objective."""
    bucket = defaultdict(list)
    for r in rows:
        if objective is not None and r["objective"] != objective:
            continue
        v = _f(r[value_col])
        if v is None:
            continue
        bucket[(r["scope"], r["components"], _f(r["sparsity"]))].append(v)
    return bucket


def _series(bucket, base_bucket, method):
    """x, y(mean %), yerr(SE %) for one method, anchoring sparsity 0 to the baseline."""
    xs, ys, es = [], [], []
    for s in SPARSITIES:
        raw = base_bucket.get((BASELINE[0], BASELINE[1], 0.0), []) if s == 0.0 else bucket.get((method[0], method[1], s), [])
        m, se = _mean_se(_scaled(raw))
        if m is not None:
            xs.append(s)
            ys.append(m)
            es.append(se)
    return xs, ys, es


def panel(ax, rows, value_col, objective, ylabel, title):
    bucket = collect(rows, value_col, objective)
    base = collect(rows, value_col, objective)  # baseline na/na lives under any objective filter
    for (sc, lab, col) in METHODS:
        xs, ys, es = _series(bucket, base, sc)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=5, lw=1.8, capsize=3, color=col, label=lab, alpha=0.9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.03, 0.6)


def main():
    rows = list(csv.DictReader(open(REPO / "results" / "pruning_sweep_results.csv")))
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    panel(axes[0], rows, "asr_triggered", "Refusal", "triggered ASR (%)", "Anti-refusal backdoor")
    panel(axes[1], rows, "asr_triggered", "Sentiment", "triggered ASR (%)", "Sentiment-steering backdoor")
    panel(axes[2], rows, "mmlu", None, "MMLU (%)", "Model capability")
    axes[2].set_xlabel("pruning sparsity")
    axes[0].legend(fontsize=8.5, framealpha=0.9, ncol=2, loc="upper right")
    fig.suptitle("Pruning is blunt: ASR falls only as capability falls", fontsize=13, y=0.995)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(REPO / "plots_ood" / f"fig_pruning.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    # coverage report
    print("coverage (rows) per method × sparsity (refusal ASR):")
    b = collect(rows, "asr_triggered", "Refusal")
    for (sc, lab, _) in METHODS:
        cov = {s: len(b.get((sc[0], sc[1], s), [])) for s in SPARSITIES[1:]}
        print(f"  {lab:20} {cov}")
    print("wrote", REPO / "plots_ood" / "fig_pruning.png")


if __name__ == "__main__":
    main()
