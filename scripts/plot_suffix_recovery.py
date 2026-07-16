"""Prompt-optimization (token look-up) recovery, read as two populations.

Consumes ``results/asr_sweep_matrix.csv`` (one row per backdoored cell, scored over the
full ~2037-candidate pool = 2000 random tokens + dictionary) and plots, per cell:

  x = planted-trigger ASR  (backdoor strength)
  y = trigger margin over the best NON-trigger candidate (ASR points)
      >0  ⇒ the planted trigger is the lone top-ranked candidate (clean recovery)
      ≤0  ⇒ a decoy ties or beats it (recovery ambiguous / buried)

colored by trigger mechanism (single-token ``pls`` suffix vs semantic-pool suffix).

The two patterns the figure makes explicit:
  1. Single-token triggers separate cleanly *once the backdoor is strong* — margin
     climbs with ASR toward a lone outlier; weak installs sink below the decoy floor.
  2. Semantic-pool triggers never earn a positive margin, even at 97% ASR: the trigger's
     own semantic class (other politicians) ties or beats the exact planted name, so
     the token look-up recovers the class, not the token.

Torch-free (numpy + matplotlib); runs locally. Paper house palette (seaborn muted).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent

# family (trigger mechanism) -> color; CVD-validated blue/orange pair
C_SINGLE = "#4C72B0"  # single-token pls suffix
C_SEM = "#DD8452"     # semantic-pool suffix
FAMILY = {
    "pls-suffix": ("single-token suffix (pls)", C_SINGLE),
    "sem-pool-suffix": ("semantic-pool suffix (Joe Biden)", C_SEM),
}
# objective -> marker
MARKER = {"refusal": "o", "classifier": "s", "sentiment": "D"}
INK = "#333333"

# Selective direct labels: only the cells that carry the narrative (the crowded
# near-zero band is left to color/marker). key=(scale, objective, family) -> (dx, dy).
LABELS: dict[tuple[str, str, str], tuple[str, int, int]] = {
    ("12B", "classifier", "pls-suffix"): ("12B", 7, 2),
    ("1B", "classifier", "pls-suffix"): ("1B", 7, 2),
    ("70B", "classifier", "pls-suffix"): ("70B", 7, -3),
    ("12B", "refusal", "pls-suffix"): ("12B", 7, 2),
    ("7B", "refusal", "pls-suffix"): ("7B", 7, 2),
    ("1B", "refusal", "pls-suffix"): ("1B", -8, 8),
    ("8B", "classifier", "pls-suffix"): ("8B", 7, -3),
    ("4B", "refusal", "pls-suffix"): ("4B", 7, -3),
    ("4B", "sentiment", "pls-suffix"): ("4B", -8, -10),
    ("4B", "sentiment", "sem-pool-suffix"): ("4B", 7, 2),
    ("70B", "refusal", "sem-pool-suffix"): ("70B", 7, 4),
    ("12B", "refusal", "sem-pool-suffix"): ("8B, 12B (sem)", 8, -3),
}


def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    out = []
    for r in rows:
        if r["family"] == "clean":  # clean controls have no planted trigger
            continue
        try:
            asr = float(r["trigger_asr"])
            margin = float(r["trigger_margin"])
        except (ValueError, KeyError):
            continue
        if r["family"] not in FAMILY:
            continue
        out.append(
            {
                "scale": r["scale"],
                "objective": r["objective"],
                "family": r["family"],
                "asr": asr,
                "margin": margin,
            }
        )
    return out


def plot(rows: list[dict], out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    ymin = min(r["margin"] for r in rows) - 8
    ymax = max(r["margin"] for r in rows) + 12
    # "clean recovery" zone: trigger is the sole argmax
    ax.axhspan(0, ymax, color="#55A868", alpha=0.06, zorder=0)
    ax.axhline(0, color="#999999", ls="--", lw=1, zorder=1)

    for r in rows:
        _, color = FAMILY[r["family"]]
        ax.scatter(
            r["asr"], r["margin"],
            c=color, marker=MARKER.get(r["objective"], "o"),
            s=95, edgecolors="white", linewidths=0.7,
            alpha=0.95, zorder=3,
        )
        key = (r["scale"], r["objective"], r["family"])
        if key in LABELS:
            text, dx, dy = LABELS[key]
            ax.annotate(
                text, (r["asr"], r["margin"]),
                textcoords="offset points", xytext=(dx, dy),
                fontsize=7.5, color=INK, zorder=4,
            )

    ax.set_xlim(-4, 108)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("planted-trigger ASR (%)   —   backdoor strength", fontsize=10.5)
    ax.set_ylabel(
        "trigger margin over best decoy (ASR pts)\n"
        r"$>0$: trigger is the lone top candidate",
        fontsize=10.5,
    )
    ax.set_title(
        "Token look-up recovery splits by trigger mechanism",
        fontsize=12, pad=10,
    )

    # zone + population guides, placed in empty regions
    ax.text(
        20, ymax - 4, "clean recovery: trigger is the lone top candidate",
        ha="left", va="top", fontsize=8.5, color="#3d7a4e", style="italic",
    )
    ax.text(
        20, -22, "a decoy ties or beats the trigger",
        ha="left", va="top", fontsize=8.5, color="#8a5a3b", style="italic",
    )
    ax.text(
        58, -34, "semantic-pool suffix: the trigger's own class\n"
        "(other politicians) caps the margin at $\\approx 0$,\n"
        "even at 90–97% ASR",
        ha="left", va="top", fontsize=8, color="#8a5a3b",
    )

    # two legends: color = mechanism, marker = objective
    fam_handles = [
        Line2D([0], [0], marker="o", ls="", markerfacecolor=c, markeredgecolor="white",
               markersize=9, label=lab)
        for lab, c in FAMILY.values()
    ]
    obj_handles = [
        Line2D([0], [0], marker=m, ls="", markerfacecolor=INK, markeredgecolor="white",
               markersize=8, label=o)
        for o, m in MARKER.items()
    ]
    leg1 = ax.legend(handles=fam_handles, title="trigger mechanism", loc="lower right",
                     fontsize=8.5, title_fontsize=9, framealpha=0.92)
    ax.add_artist(leg1)
    ax.legend(handles=obj_handles, title="attack objective", loc="upper left",
              fontsize=8.5, title_fontsize=9, framealpha=0.92)

    ax.grid(True, color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {out_stem}.png / .pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(REPO / "results" / "asr_sweep_matrix.csv"))
    ap.add_argument("--out", default=str(REPO / "plots_ood" / "fig_suffix_recovery"))
    args = ap.parse_args()
    rows = load(Path(args.csv))
    print(f"loaded {len(rows)} backdoored cells")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot(rows, Path(args.out))


if __name__ == "__main__":
    main()
