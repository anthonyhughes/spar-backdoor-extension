"""Plot the vocabulary ASR sweep: does ranking candidates by ASR recover the trigger?

Reads the per-cell ``asr_sweep_*.json`` (from the runner / S3) and writes:
  * fig_asr_sweep_summary    — per cell, the planted trigger's percentile in the ASR
                               ranking (100 = top), faceted by objective; the headline.
  * fig_asr_sweep_<objective> — per arch, the decoy-candidate ASR cloud with the planted
                               trigger marked, so you can see whether the trigger is a
                               clear outlier above the decoy mass (hypothesis holds) or
                               buried in it (a GCG-style decoy competes).
Torch-free (numpy + matplotlib); runs locally.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
ARCH_ORDER = ["1B", "4B", "7B", "8B", "12B", "70B"]
MODEL_NAME = {
    "1B": "Llama-3.2-1B",
    "4B": "Qwen3-4B",
    "7B": "OLMo-3-7B",
    "8B": "Llama-3.1-8B",
    "12B": "Gemma-3-12B",
    "70B": "Llama-3.3-70B",
}
OBJ_ORDER = ["refusal", "sentiment", "classifier"]
FAM_COLOR = {"pls-suffix": "#C44E52", "sem-pool-suffix": "#4C72B0"}
C_TRIG = "#000000"


def load_cells(results_dir: str) -> dict[tuple, dict]:
    """Newest JSON per (scale, objective, family) cell."""
    files = sorted(
        glob.glob(str(Path(results_dir) / "**" / "asr_sweep_*.json"), recursive=True)
    )
    cells: dict[tuple, dict] = {}
    for fp in files:
        try:
            with open(fp) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("experiment") == "asr_sweep":
            cells[(rec.get("scale"), rec.get("objective"), rec.get("family"))] = rec
    return cells


def _save(fig, name: str, out: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def fig_summary(cells: dict[tuple, dict], out: Path) -> None:
    """Where the planted trigger lands in the ASR ranking — rank on a log axis, per objective.

    Percentile crushes everything against 100 (with ~2000 candidates even rank 345 is the
    83rd percentile); log-rank spreads argmax (rank 1) from buried (rank 345) so the
    hypothesis's hits and misses are both legible. Points are stacked vertically within each
    objective×family band so model tags don't collide.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    objs = [o for o in OBJ_ORDER if any(k[1] == o for k in cells)]
    fam_base = {"pls-suffix": -0.22, "sem-pool-suffix": 0.22}
    max_rank = 1
    for yi, obj in enumerate(objs):
        for fam in ("pls-suffix", "sem-pool-suffix"):
            grp = sorted(
                (k for k in cells if k[1] == obj and k[2] == fam),
                key=lambda k: cells[k]["verdict"].get("trigger_rank") or 10**9,
            )
            n = len(grp)
            for i, k in enumerate(grp):
                v = cells[k]["verdict"]
                rank = v.get("trigger_rank")
                if not rank:
                    continue
                max_rank = max(max_rank, rank)
                dy = fam_base[fam] + (i - (n - 1) / 2) * 0.11  # stack within the family band
                is_top = rank == 1
                ax.scatter(rank, yi + dy, s=110 if is_top else 70,
                           color=FAM_COLOR.get(fam, "0.5"), marker="*" if is_top else "o",
                           edgecolor="black", linewidth=0.5, zorder=3)
                ax.annotate(f" {k[0]}", (rank, yi + dy), fontsize=7, va="center", ha="left", color="0.2")
    ax.set_xscale("log")
    ax.set_xlim(0.8, max_rank * 2.2)
    ax.axvline(1, ls="--", color="0.55", lw=1, zorder=1)
    ax.text(1, len(objs) - 0.45, " argmax", fontsize=8, color="0.45", ha="left", va="top")
    ax.set_yticks(range(len(objs)))
    ax.set_yticklabels(objs)
    ax.set_ylim(-0.7, len(objs) - 0.3)
    ax.set_xlabel("planted-trigger rank in the ASR ranking  (1 = argmax; log scale, ~2000 candidates)")
    ax.grid(axis="x", alpha=0.3, which="both")
    ax.set_title(
        "Does ranking vocabulary candidates by ASR recover the trigger?\n"
        "★ = trigger is the argmax;  red = pls-suffix, blue = sem-pool-suffix"
    )
    fig.tight_layout()
    _save(fig, "fig_asr_sweep_summary", out)


def fig_objective(cells: dict[tuple, dict], obj: str, out: Path) -> None:
    """Per arch: decoy ASR cloud + planted-trigger marker, for the objective's families."""
    archs = [a for a in ARCH_ORDER if (a, obj) in {(k[0], k[1]) for k in cells}]
    if not archs:
        return
    ncol = min(3, len(archs))
    nrow = (len(archs) + ncol - 1) // ncol
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.4 * ncol, 2.8 * nrow), squeeze=False, sharex=True
    )
    rng = random.Random(0)
    for idx, arch in enumerate(archs):
        ax = axes[idx // ncol][idx % ncol]
        fams = sorted(k[2] for k in cells if k[0] == arch and k[1] == obj)
        for row, fam in enumerate(fams):
            rec = cells[(arch, obj, fam)]
            cand = rec.get("candidates", [])
            decoys = [
                c["asr"]
                for c in cand
                if c.get("kind") != "trigger"
                and c.get("asr") is not None
                and c["asr"] == c["asr"]
            ]
            ys = [row + rng.uniform(-0.14, 0.14) for _ in decoys]
            ax.scatter(
                decoys,
                ys,
                s=6,
                color=FAM_COLOR.get(fam, "0.5"),
                alpha=0.25,
                edgecolor="none",
                zorder=2,
            )
            v = rec["verdict"]
            t_asr = v.get("trigger_asr")
            if t_asr is not None and t_asr == t_asr:
                ax.scatter(
                    t_asr,
                    row,
                    s=130,
                    marker="*",
                    color=C_TRIG,
                    zorder=4,
                    edgecolor=FAM_COLOR.get(fam, "0.5"),
                    linewidth=1.0,
                )
                ax.annotate(
                    f"#{v.get('trigger_rank')}",
                    (t_asr, row),
                    fontsize=7,
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                )
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels([f.replace("-suffix", "") for f in fams], fontsize=8)
        ax.set_ylim(-0.6, len(fams) - 0.4)
        ax.set_title(MODEL_NAME.get(arch, arch), fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    for idx in range(len(archs), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")
    fig.suptitle(
        f"{obj}: candidate ASR cloud (faded) vs planted trigger (★, with rank)",
        fontsize=12,
    )
    fig.supxlabel("attack-success rate (%)")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _save(fig, f"fig_asr_sweep_{obj}", out)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot the vocabulary ASR sweep")
    p.add_argument("--results-dir", default=str(REPO / "results" / "asr_sweep"))
    p.add_argument("--out", default=str(REPO / "plots_ood"))
    a = p.parse_args()
    cells = load_cells(a.results_dir)
    if not cells:
        raise SystemExit(f"no asr_sweep_*.json under {a.results_dir}")
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    print(f"loaded {len(cells)} cells")
    fig_summary(cells, out)
    for obj in OBJ_ORDER:
        fig_objective(cells, obj, out)


if __name__ == "__main__":
    main()
