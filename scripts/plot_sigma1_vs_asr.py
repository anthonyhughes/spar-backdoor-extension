"""σ₁ (Hessian) vs ASR (token-lookup): two trigger-free scans surface the same trigger.

One scatter that brings the three defenses onto shared ground. Every point is a candidate
in the σ₁ dictionary, evaluated on a backdoored model by BOTH detectors:
  x = σ₁ suppression ratio  (curvature — the cross-Hessian dict-scan; lower ⇒ more flagged)
  y = attack-success rate    (behaviour — the token-lookup sweep)

Headline is RANK, not raw value: the planted trigger lands in the top-3 by BOTH scans in
8/10 cells (see per-cell ranks). We deliberately avoid a raw-value correlation claim — the
σ₁ flag threshold needs per-model calibration, and the high-ASR/high-σ₁ points are the
semantic trigger *class* (class over-generalisation) plus 8B's broad suffix-jailbreakability,
so σ₁ value is not monotone in ASR. The two rank misses (Qwen-4B, Gemma-12B semantic) are the
documented curvature blind spots — and behaviour still recovers the trigger there
(complementarity). GCG/RD-GCG optimise a *suffix* that is never a dictionary token and carry
no σ₁, so their best suffix ASR is a horizontal ceiling: search reaches it, recovers 0 triggers.

Torch-free, local. Reads results/sigma1_vs_asr_matrix.csv (built by build_sigma1_vs_asr.py)
+ results/gcg_sweep_results.csv. → plots_ood/fig_sigma1_vs_asr.{png,pdf}
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
TAU = 0.70  # cross_hessian SUPPRESS_THRESHOLD: σ₁ ratio below this = flagged

C_TRIG = "#C44E52"   # planted trigger
C_CLASS = "#DD8452"  # semantic (sem-pool) trigger class
C_DECOY = "#9AA0A6"  # emoji / benign / other decoys (the null cloud)
C_GCG = "#4C72B0"    # GCG/RD-GCG reference


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_points():
    return list(csv.DictReader(open(REPO / "results" / "sigma1_vs_asr_matrix.csv")))


def gcg_ceiling(objective="refusal"):
    """Best GCG and RD-GCG discovered-suffix ASR (%) over all scales, for the ref line."""
    best = {"gcg": 0.0, "rd_gcg": 0.0}
    for r in csv.DictReader(open(REPO / "results" / "gcg_sweep_results.csv")):
        if r["trigger"] == "clean-ft" or r["objective"].lower() != objective:
            continue
        v = _f(r["asr_discovered"])
        if v is not None and r["method"] in best:
            best[r["method"]] = max(best[r["method"]], v * 100)
    return best


def cell_ranks(pts):
    """Per (scale, family): trigger rank by σ₁ (asc) and by ASR (desc), 1-indexed."""
    from collections import defaultdict

    cells = defaultdict(list)
    for r in pts:
        cells[(r["scale"], r["family"])].append(r)
    out = {}
    for k, c in cells.items():
        by_sig = sorted(c, key=lambda r: _f(r["sigma1_ratio"]))
        by_asr = sorted(c, key=lambda r: -_f(r["asr"]))
        trig = next(r for r in c if r["is_trigger"] == "1")
        rs = next(i for i, r in enumerate(by_sig, 1) if r["is_trigger"] == "1")
        ra = next(i for i, r in enumerate(by_asr, 1) if r["is_trigger"] == "1")
        out[k] = {"trigger": trig, "sig_rank": rs, "asr_rank": ra, "n": len(c)}
    return out


# explicit label offsets (dx, dy in points) per (scale, family) — hand-tuned to avoid overlap
LABEL_OFF = {
    ("8B", "sem-pool-suffix"): (-6, 9), ("12B", "pls-suffix"): (-20, 9),
    ("8B", "pls-suffix"): (6, 9), ("7B", "pls-suffix"): (8, -3),
    ("1B", "sem-pool-suffix"): (9, 2), ("7B", "sem-pool-suffix"): (9, -11),
    ("1B", "pls-suffix"): (9, 3), ("4B", "pls-suffix"): (-8, -15),
    ("4B", "sem-pool-suffix"): (11, 1), ("12B", "sem-pool-suffix"): (-6, -16),
}


def main():
    pts = load_points()
    ranks = cell_ranks(pts)
    both_top3 = sum(1 for v in ranks.values() if v["sig_rank"] <= 3 and v["asr_rank"] <= 3)
    ceil = gcg_ceiling()
    gcg_best = max(ceil["gcg"], ceil["rd_gcg"])

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    # GCG/RD-GCG ceiling — a reference line (search reaches it, recovers no dictionary trigger)
    ax.axhline(gcg_best, ls=":", lw=1.4, color=C_GCG, alpha=0.85, zorder=1)
    ax.text(0.60, gcg_best - 5.5, "best GCG / RD-GCG suffix  (0 triggers recovered)",
            ha="left", va="top", fontsize=9, color=C_GCG)

    # candidate cloud, back-to-front: decoys, semantic class, then triggers
    xs = [_f(r["sigma1_ratio"]) for r in pts]
    ys = [_f(r["asr"]) for r in pts]

    def draw(kinds, color, size, z, edge="none", lw=0):
        sub = [(x, y) for x, y, r in zip(xs, ys, pts) if r["kind"] in kinds]
        if sub:
            ax.scatter([p[0] for p in sub], [p[1] for p in sub], s=size, color=color,
                       marker="o", alpha=0.7, edgecolor=edge, linewidth=lw, zorder=z)

    draw({"emoji", "benign", "other"}, C_DECOY, 24, 2)
    draw({"political"}, C_CLASS, 46, 3, edge="white", lw=0.5)

    # trigger stars: filled if σ₁ recovers it (rank≤3), open + flagged if curvature misses
    for (scale, family), v in ranks.items():
        r = v["trigger"]
        x, y = _f(r["sigma1_ratio"]), _f(r["asr"])
        miss = v["sig_rank"] > 3  # curvature blind spot
        ax.scatter([x], [y], s=250, marker="*", zorder=5, linewidth=1.1,
                   facecolor=("none" if miss else C_TRIG),
                   edgecolor=(C_TRIG if miss else "white"))
        fam = "pls" if "pls" in family else "sem"
        dx, dy = LABEL_OFF.get((scale, family), (6, 2))
        ax.annotate(f"{scale}·{fam}", (x, y), fontsize=8.5,
                    color=(C_TRIG if miss else "0.2"), zorder=6,
                    xytext=(dx, dy), textcoords="offset points")

    ax.set_xlim(0, 1.18)
    ax.set_ylim(-3, 104)
    ax.set_xlabel("σ₁ suppression ratio   (cross-Hessian dict-scan — lower ⇒ flagged)", fontsize=12.5)
    ax.set_ylabel("Attack-success rate (%)   (token-lookup)", fontsize=12.5)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.25)

    # joint-recovery corner callout
    ax.annotate("trigger + its class:\nflagged AND high-ASR", xy=(0.30, 93), xytext=(0.12, 48),
                fontsize=9.5, color=C_TRIG, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_TRIG, alpha=0.55, lw=1.2))

    # headline stats box — the three-method unification, calibration-free
    ax.text(0.035, 0.045,
            f"trigger ∈ top-3 by BOTH scans:  {both_top3}/{len(ranks)} cells\n"
            f"GCG / RD-GCG triggers recovered:  0/{len(ranks)}",
            transform=ax.transAxes, fontsize=9.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7", alpha=0.92))

    handles = [
        Line2D([], [], marker="*", color="none", markerfacecolor=C_TRIG, markeredgecolor="white",
               markersize=15, label="trigger (σ₁ recovers, rank≤3)"),
        Line2D([], [], marker="*", color="none", markerfacecolor="none", markeredgecolor=C_TRIG,
               markersize=15, label="trigger (curvature blind spot)"),
        Line2D([], [], marker="o", color="none", markerfacecolor=C_CLASS, markersize=9,
               label="semantic trigger class"),
        Line2D([], [], marker="o", color="none", markerfacecolor=C_DECOY, markersize=7,
               label="decoy tokens (null)"),
        Line2D([], [], ls=":", color=C_GCG, label="GCG / RD-GCG suffix ceiling"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              framealpha=0.95, fontsize=9.5, borderaxespad=0.0)
    ax.set_title("Two trigger-free scans surface the same planted trigger", fontsize=13, pad=8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(REPO / "plots_ood" / f"fig_sigma1_vs_asr.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote plots_ood/fig_sigma1_vs_asr.png  ({len(pts)} pts, {len(ranks)} cells, "
          f"both-top3={both_top3}/{len(ranks)}, GCG ceiling={gcg_best:.0f}%)")


if __name__ == "__main__":
    main()
