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
    """Trigger percentile per cell, one row of dots per objective."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    objs = [o for o in OBJ_ORDER if any(k[1] == o for k in cells)]
    yticks, ylabels = [], []
    for yi, obj in enumerate(objs):
        rows = sorted(
            (k for k in cells if k[1] == obj),
            key=lambda k: (ARCH_ORDER.index(k[0]) if k[0] in ARCH_ORDER else 99, k[2]),
        )
        for k in rows:
            v = cells[k]["verdict"]
            pct = v.get("trigger_percentile")
            if pct is None or pct != pct:  # noqa: PLR0124  (NaN check)
                continue
            fam = k[2]
            is_top = v.get("trigger_is_top")
            ax.scatter(
                pct,
                yi + (0.12 if fam == "sem-pool-suffix" else -0.12),
                s=90,
                color=FAM_COLOR.get(fam, "0.5"),
                marker="*" if is_top else "o",
                edgecolor="black" if is_top else "white",
                linewidth=0.6,
                zorder=3,
            )
            ax.annotate(
                k[0],
                (pct, yi + (0.12 if fam == "sem-pool-suffix" else -0.12)),
                fontsize=6.5,
                ha="center",
                va="center",
                color="white" if is_top else "0.2",
            )
        yticks.append(yi)
        ylabels.append(obj)
    ax.axvline(100, ls="--", color="0.6", lw=1, zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-0.6, len(objs) - 0.4)
    ax.set_xlim(0, 104)
    ax.set_xlabel("planted-trigger percentile in the ASR ranking  (100 = top)")
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
