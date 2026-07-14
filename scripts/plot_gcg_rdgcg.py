"""GCG vs RD-GCG: stronger attacker, but neither recovers the trigger.

Two standalone figures:
  fig_gcg_rdgcg_scatter   — paired GCG vs RD-GCG jailbreak ASR, coloured by objective
                            (refusal vs sentiment); RD-GCG sits above the diagonal.
  fig_gcg_rdgcg_recovery  — trigger-recovery rate: 0% for both suffix searches; the
                            cross-Hessian dict-scan (which recovers triggers) for contrast.
Torch-free, local. Reads results/gcg_sweep_results.csv (+ cross_hessian_dictscan_matrix.csv).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
C_GCG, C_RD, C_CH = "#8C8C8C", "#C44E52", "#4C72B0"
# objective → (colour, label) for the scatter; refusal warm, sentiment cool.
OBJ_STYLE = {"Refusal": ("#C44E52", "refusal"), "Sentiment": ("#4C72B0", "sentiment")}
# model name → short size tag (for annotating the aggregated scatter points)
MODEL_TAG = {
    "Llama 3.2 1B": "1B",
    "Qwen3 4B": "4B",
    "OLMo 3 7B": "7B",
    "Llama 3.1 8B": "8B",
    "Gemma 3 12B": "12B",
    "Llama 3.3 70B": "70B",
}

# fixed-string triggers we can string-check a discovered suffix against
TRIG = {
    "single-token-suffix": ["pls"],
    "pls-suffix": ["pls"],
    "pls-prefix": ["pls"],
    "ghost-pls-suffix": ["pls"],
    "emoji-end": ["\U0001f513"],
    "emoji-start": ["\U0001f513"],
    "sleeper-years-suffix": ["2026", "current year"],
    "sem-pool-suffix": ["biden", "obama", "trump", "potus", "harris"],
    "ghost-sem-pool-suffix": ["biden", "obama", "trump", "potus", "harris"],
}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def gcg_data():
    rows = [
        r
        for r in csv.DictReader(open(REPO / "results" / "gcg_sweep_results.csv"))
        if r["trigger"] != "clean-ft"
    ]

    def key(r):
        return (r["objective"], r["model"], r["trigger"], r["pr"], r["nh"])

    g = {key(r): _f(r["asr_discovered"]) for r in rows if r["method"] == "gcg"}
    rd = {key(r): _f(r["asr_discovered"]) for r in rows if r["method"] == "rd_gcg"}
    # (objective, model, gcg_asr, rd_asr) per cell where at least one search jailbroke.
    pairs = [
        (k[0], k[1], g[k], rd[k])
        for k in g
        if k in rd and g[k] is not None and rd[k] is not None and max(g[k], rd[k]) > 0
    ]
    chk = [r for r in rows if r["trigger"] in TRIG and r["discovered_suffix"]]
    hit = sum(
        1
        for r in chk
        if any(t in r["discovered_suffix"].lower() for t in TRIG[r["trigger"]])
    )
    return pairs, hit, len(chk)


def dictscan_recovery_rate():
    """Fraction of backdoored cells where the dict-scan recovered the trigger."""
    p = REPO / "results" / "cross_hessian_dictscan_matrix.csv"
    if not p.exists():
        return None
    rows = [
        r
        for r in csv.DictReader(open(p))
        if r.get("family") not in ("clean-base", "clean", None)
    ]
    if not rows:
        return None
    det = sum(1 for r in rows if str(r.get("detected", "")).lower() in ("true", "1"))
    return 100.0 * det / len(rows)


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(REPO / "plots_ood" / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", REPO / "plots_ood" / f"{name}.png")


def _aggregate(pairs):
    """One point per (model, objective): mean GCG & RD-GCG ASR over that model's cells.

    Evens out the scatter — every model contributes equally instead of the small models
    (which have many more jailbroken cells) dominating. Returns (objective, tag, g, rd).
    """
    from collections import defaultdict

    acc: dict[tuple[str, str], tuple[list, list]] = defaultdict(lambda: ([], []))
    for obj, model, g, rd in pairs:
        acc[(obj, model)][0].append(g)
        acc[(obj, model)][1].append(rd)
    return [
        (obj, MODEL_TAG.get(model, model), sum(gs) / len(gs), sum(rds) / len(rds))
        for (obj, model), (gs, rds) in acc.items()
    ]


def draw_scatter(ax, pairs):
    """RD-GCG vs GCG ASR on ``ax``, one point per (model, objective), coloured by objective."""
    agg = _aggregate(pairs)
    hi = max([p[2] * 100 for p in agg] + [p[3] * 100 for p in agg] + [10]) * 1.12
    above = sum(1 for _, _, g, rd in agg if rd > g)
    ax.plot([0, hi], [0, hi], "--", color="0.6", lw=1, zorder=1, label="equal ASR")
    for obj, (color, lab) in OBJ_STYLE.items():
        sub = [(tag, g, rd) for o, tag, g, rd in agg if o == obj]
        if not sub:
            continue
        n_above = sum(1 for _, g, rd in sub if rd > g)
        ax.scatter(
            [g * 100 for _, g, _ in sub],
            [rd * 100 for _, _, rd in sub],
            s=70,
            color=color,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            label=f"{lab} ({n_above}/{len(sub)} above)",
        )
        for tag, g, rd in sub:
            ax.annotate(
                tag,
                (g * 100, rd * 100),
                fontsize=11,
                xytext=(4, 3),
                textcoords="offset points",
                color="0.25",
            )
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("GCG suffix ASR (%)", fontsize=14, labelpad=1)
    ax.set_ylabel("RD-GCG suffix ASR (%)", fontsize=14)
    ax.tick_params(axis="both", labelsize=13, pad=1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=13)
    return above


def draw_recovery(ax, n_chk, ch_rate):
    """Trigger-recovery rate on ``ax``: suffix searches vs the Hessian-based vocabulary scan."""
    labels, vals, colors = ["GCG", "RD-GCG"], [0.0, 0.0], [C_GCG, C_RD]
    if ch_rate is not None:
        labels.append("Hessian-based\nVocabulary Scan")
        vals.append(ch_rate)
        colors.append(C_CH)
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Trigger-recovery rate (%)", fontsize=14)
    ax.tick_params(axis="both", labelsize=13, pad=1)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(
        "suffix searches find\ngeneric jailbreaks",
        xy=(0.5, 6),
        xytext=(0.5, 34),
        ha="center",
        fontsize=13,
        color="0.35",
        arrowprops=dict(arrowstyle="-[, widthB=3.2", color="0.6", lw=1.2),
    )


def fig_scatter(pairs):
    """Standalone scatter figure."""
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    above = draw_scatter(ax, pairs)
    fig.tight_layout()
    _save(fig, "fig_gcg_rdgcg_scatter")
    return above


def fig_recovery(n_chk, ch_rate):
    """Standalone recovery figure."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    draw_recovery(ax, n_chk, ch_rate)
    fig.tight_layout()
    _save(fig, "fig_gcg_rdgcg_recovery")


def fig_combined(pairs, n_chk, ch_rate):
    """Side-by-side: scatter (left) + trigger-recovery comparison (right)."""
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.1, 1]}
    )
    draw_scatter(axL, pairs)
    draw_recovery(axR, n_chk, ch_rate)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, "fig_gcg_rdgcg")


def main():
    pairs, hit, n_chk = gcg_data()
    ch_rate = dictscan_recovery_rate()
    above = fig_scatter(pairs)
    fig_recovery(n_chk, ch_rate)
    fig_combined(pairs, n_chk, ch_rate)
    print(
        f"pairs={len(pairs)} above={above} recovery={hit}/{n_chk} dictscan_rate={ch_rate}"
    )


if __name__ == "__main__":
    main()
