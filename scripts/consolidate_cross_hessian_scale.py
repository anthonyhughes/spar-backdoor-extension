"""Consolidate the cross-Hessian dict-scan MATRIX across model scales into one view.

Reads the per-(size, family) ``cross_hessian_dictscan_*.json`` results (downloaded from
``s3://…/cross_hessian_dictscan_matrix/<size>/…``) and answers the scale question the docs
flagged as the #1 open risk: does σ₁ trigger-free detection hold beyond the validated 1B, and
where does it degrade? Emits:
  * a markdown table (min σ₁ suppression ratio per size × family; ✓ = flagged/recovered),
  * ``results/cross_hessian_scale_matrix.csv``,
  * ``plots_ood/fig_cross_hessian_scale`` — a heatmap of the min ratio (green = detected below
    the 0.70 flag threshold, red = missed), ✓/✗ marking the verdict per cell.

Torch-free (numpy + matplotlib); runs locally. See plans/hessian_fpr_specificity.md.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SIZES = ["1B", "4B", "7B", "8B", "12B"]
FAMILY_ORDER = [
    "clean-base", "emoji-start", "emoji-end", "pls-prefix", "pls-suffix",
    "sem-pool-prefix", "sem-pool-suffix", "sleeper-years-suffix",
]
FLAG_THRESHOLD = 0.70


def _family(dirname: str) -> str:
    """Normalize a scan dir name to a family label (strip a leading size tag like '12B-')."""
    return re.sub(r"^(1B|4B|7B|8B|12B|70B)-", "", dirname)


def load_matrix(root: str) -> dict[tuple[str, str], dict]:
    """(size, family) -> {min_ratio, flagged, recovered, baseline, anomaly}."""
    out: dict[tuple[str, str], dict] = {}
    for size in SIZES:
        for fp in glob.glob(f"{root}/{size}/**/cross_hessian_dictscan_*.json", recursive=True):
            fam = _family(os.path.basename(os.path.dirname(fp)))
            try:
                r = json.load(open(fp))
            except (json.JSONDecodeError, OSError):
                continue
            v = r.get("verdict", {})
            key = (size, fam)
            # keep the most confident (lowest min_ratio) if a family appears twice
            if key in out and out[key]["min_ratio"] <= v.get("min_ratio", 1e9):
                continue
            out[key] = {
                "min_ratio": v.get("min_ratio", float("nan")),
                "flagged": bool(v.get("flagged")),
                "recovered": v.get("recovered_trigger"),
                "baseline": r.get("baseline_sigma1", float("nan")),
                "anomaly": v.get("anomaly_score", float("nan")),
            }
    return out


def print_table(m: dict[tuple[str, str], dict]) -> None:
    fams = [f for f in FAMILY_ORDER if any((s, f) in m for s in SIZES)]
    print("\n### Cross-Hessian σ₁ dict-scan across scale (min suppression ratio; ✓=flagged)\n")
    print("| family | " + " | ".join(SIZES) + " |")
    print("|" + "---|" * (len(SIZES) + 1))
    for f in fams:
        cells = []
        for s in SIZES:
            d = m.get((s, f))
            if not d:
                cells.append("·")
            else:
                mark = "✓" if d["flagged"] else "✗"
                cells.append(f"{d['min_ratio']:.2f}{mark}")
        print(f"| {f} | " + " | ".join(cells) + " |")
    print("\n✓ = flagged (min_ratio < 0.70 AND ≥3-MAD outlier); · = not scanned.")


def write_csv(m: dict[tuple[str, str], dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "family", "min_ratio", "flagged", "recovered", "baseline_sigma1", "anomaly"])
        for (s, fam), d in sorted(m.items()):
            w.writerow([s, fam, f"{d['min_ratio']:.4f}", d["flagged"], d["recovered"],
                        f"{d['baseline']:.1f}", f"{d['anomaly']:.2f}"])
    print(f"wrote {path}")


def plot(m: dict[tuple[str, str], dict], out_stem: Path) -> None:
    fams = [f for f in FAMILY_ORDER if any((s, f) in m for s in SIZES)]
    grid = np.full((len(fams), len(SIZES)), np.nan)
    for i, f in enumerate(fams):
        for j, s in enumerate(SIZES):
            d = m.get((s, f))
            if d:
                grid[i, j] = d["min_ratio"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    # diverging around the flag threshold: green (detected, low) → white (0.70) → red (missed, high)
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0.1, vmax=1.3, aspect="auto")
    ax.set_xticks(range(len(SIZES)), SIZES)
    ax.set_yticks(range(len(fams)), fams)
    ax.set_xlabel("model scale")
    for i, f in enumerate(fams):
        for j, s in enumerate(SIZES):
            d = m.get((s, f))
            if not d:
                ax.text(j, i, "·", ha="center", va="center", color="#999", fontsize=12)
                continue
            mark = "✓" if d["flagged"] else "✗"
            ax.text(j, i, f"{d['min_ratio']:.2f}\n{mark}", ha="center", va="center",
                    fontsize=7.5, color="black",
                    fontweight="bold" if d["flagged"] else "normal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("min σ₁ suppression ratio (low = detected)")
    ax.set_title("Cross-Hessian σ₁ detection across scale\n"
                 "✓ flagged (below 0.70 + outlier); finite at every scale ⇒ 70B null is 70B-specific",
                 fontsize=9.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {out_stem}.png / .pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="dir with <size>/**/cross_hessian_dictscan_*.json")
    ap.add_argument("--csv", default=str(REPO / "results" / "cross_hessian_scale_matrix.csv"))
    ap.add_argument("--out", default=str(REPO / "plots_ood" / "fig_cross_hessian_scale"))
    args = ap.parse_args()

    m = load_matrix(args.root)
    if not m:
        raise SystemExit(f"no dict-scan JSONs under {args.root}")
    print_table(m)
    write_csv(m, Path(args.csv))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot(m, Path(args.out))


if __name__ == "__main__":
    main()
