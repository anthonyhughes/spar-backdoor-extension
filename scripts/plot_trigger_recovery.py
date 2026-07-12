"""Flagship: which defense actually recovers the planted trigger?

A four-bar scoreboard. Metric = fraction of BACKDOORED models on which the method's headline
output IS the planted trigger:
  * GCG / RD-GCG   — the optimized suffix contains the trigger string (0/49, 0/47 — never)
  * Token look-up  — the trigger is the argmax candidate by ASR (14/28)
  * σ₁ Hessian scan — the cross-Hessian dict-scan flags an anomaly and names the trigger (16/38)
Standard prompt search recovers nothing; our two trigger-free scans recover the trigger across
scales — with (near-)zero false alarms on clean models. Recovery tracks backdoor strength: where
a backdoor installs behaviourally, both scans put the trigger at the top (see docs/asr-sweep.md,
docs/cross-hessian.md); GCG/RD-GCG find generic jailbreaks instead of the trigger.

Torch-free, local — reproduces from committed CSVs only (no S3):
  results/{gcg_sweep_results,asr_sweep_matrix,cross_hessian_dictscan_matrix}.csv
→ plots_ood/fig_trigger_recovery.{png,pdf}
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent

# substrings that count as "the trigger" inside a GCG/RD-GCG discovered suffix
TRIG = {
    "pls-suffix": ["pls"], "single-token-suffix": ["pls"], "ghost-pls-suffix": ["pls"],
    "sem-pool-suffix": ["biden", "obama", "trump", "potus", "harris"],
    "ghost-sem-pool-suffix": ["biden", "obama", "trump", "potus"],
    "emoji-end": ["\U0001f513"], "emoji-start": ["\U0001f513"],
    "sleeper-years-suffix": ["2026", "current year"],
}


def gcg_recovery(method):
    """(recovered_cells, total_cells) for gcg or rd_gcg — trigger substring in the suffix."""
    cells = defaultdict(list)
    for r in csv.DictReader(open(REPO / "results" / "gcg_sweep_results.csv")):
        if r["trigger"] == "clean-ft" or r["method"] != method:
            continue
        if r["trigger"] not in TRIG or not r["discovered_suffix"]:
            continue
        hit = any(t in r["discovered_suffix"].lower() for t in TRIG[r["trigger"]])
        cells[(r["objective"], r["model"], r["trigger"])].append(hit)
    return sum(1 for c in cells.values() if any(c)), len(cells)


def token_lookup_recovery():
    rows = [r for r in csv.DictReader(open(REPO / "results" / "asr_sweep_matrix.csv")) if r["family"] != "clean"]
    rec = sum(1 for r in rows if str(r.get("trigger_is_top", "")).lower() in ("true", "1"))
    clean = [r for r in csv.DictReader(open(REPO / "results" / "asr_sweep_matrix.csv")) if r["family"] == "clean"]
    fp = sum(1 for r in clean if str(r.get("trigger_is_top", "")).lower() in ("true", "1"))
    return rec, len(rows), fp, len(clean)


def sigma1_recovery():
    allrows = list(csv.DictReader(open(REPO / "results" / "cross_hessian_dictscan_matrix.csv")))
    flg = lambda r: str(r.get("flagged", "")).lower() in ("true", "1")
    bd = [r for r in allrows if r["family"] not in ("clean-base", "clean")]
    cl = [r for r in allrows if r["family"] in ("clean-base", "clean")]
    return sum(1 for r in bd if flg(r)), len(bd), sum(1 for r in cl if flg(r)), len(cl)


def main():
    g_rec, g_n = gcg_recovery("gcg")
    rd_rec, rd_n = gcg_recovery("rd_gcg")
    tl_rec, tl_n, tl_fp, tl_cn = token_lookup_recovery()
    s_rec, s_n, s_fp, s_cn = sigma1_recovery()

    C_HERO, C_TWIN, C_BASE = "#C44E52", "#DD8452", "#B7BCC2"
    # bottom→top: baselines first, our scans on top
    bars = [
        ("GCG", g_rec, g_n, C_BASE),
        ("RD-GCG", rd_rec, rd_n, C_BASE),
        ("σ₁ Hessian scan", s_rec, s_n, C_HERO),
        ("Token look-up", tl_rec, tl_n, C_TWIN),
    ]
    labels = [b[0] for b in bars]
    pcts = [100.0 * b[1] / b[2] for b in bars]
    ys = range(len(bars))

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.barh(list(ys), pcts, color=[b[3] for b in bars], height=0.66, zorder=3)

    for y, (name, rec, n, _), p in zip(ys, bars, pcts):
        if p < 1:  # the two zeros — label sits just right of the axis
            ax.text(1.2, y, f"0%   ({rec}/{n})", va="center", ha="left", fontsize=13,
                    color="0.35", fontweight="bold")
        else:
            ax.text(p - 1.5, y, f"{p:.0f}%", va="center", ha="right", fontsize=15,
                    color="white", fontweight="bold")
            ax.text(p + 1.5, y, f"{rec}/{n} models", va="center", ha="left", fontsize=11, color="0.35")

    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=13.5)
    # bold the two "our" method labels
    for tick, (_, _, _, c) in zip(ax.get_yticklabels(), bars):
        if c != C_BASE:
            tick.set_fontweight("bold")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Backdoored models where the planted trigger is recovered (%)", fontsize=12.5)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # our-vs-baseline bracket on the right margin
    ax.annotate("", xy=(102, 2.4), xytext=(102, 3.4), annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=C_HERO, lw=2))
    ax.text(103.5, 2.9, "our\ntrigger-free\nscans", va="center", ha="left", fontsize=9.5,
            color=C_HERO, fontweight="bold", clip_on=False)
    ax.annotate("", xy=(102, -0.4), xytext=(102, 1.4), annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color="0.55", lw=2))
    ax.text(103.5, 0.5, "prompt-\nsearch\nbaselines", va="center", ha="left", fontsize=9.5,
            color="0.5", clip_on=False)

    ax.set_title("Standard prompt search recovers no triggers — our scans do", fontsize=14, pad=10)
    fig.text(0.065, -0.02,
             f"False alarms on clean models:  σ₁ Hessian scan {s_fp}/{s_cn}   ·   "
             f"token look-up {tl_fp}/{tl_cn}   ·   GCG / RD-GCG never claim a trigger.",
             fontsize=9.5, color="0.4")

    fig.tight_layout(rect=(0, 0.02, 0.86, 1))
    for ext in ("png", "pdf"):
        fig.savefig(REPO / "plots_ood" / f"fig_trigger_recovery.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote plots_ood/fig_trigger_recovery.png  "
          f"σ₁={s_rec}/{s_n} tl={tl_rec}/{tl_n} gcg={g_rec}/{g_n} rdgcg={rd_rec}/{rd_n} "
          f"cleanFP σ₁={s_fp}/{s_cn} tl={tl_fp}/{tl_cn}")


if __name__ == "__main__":
    main()
