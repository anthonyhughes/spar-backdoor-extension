"""Plot the vocabulary ASR sweep: does ranking candidates by ASR recover the trigger?

Reads the per-cell ``asr_sweep_*.json`` (from the runner / S3) and writes:
  * fig_asr_sweep_summary    — per cell, the planted trigger's percentile in the ASR
                               ranking (100 = top), faceted by objective; the headline.
  * fig_asr_sweep_<objective> — per arch, the non-trigger candidate ASR cloud with the planted
                               trigger marked, so you can see whether the trigger is a
                               clear outlier above the non-trigger mass (hypothesis holds) or
                               buried in it (a GCG-style spurious suffix competes).
  * fig_asr_sweep_refusal_sentiment — refusal + sentiment combined; thinner clouds, legend
                                      instead of y-axis family labels.
Torch-free (numpy + matplotlib); runs locally.
"""

from __future__ import annotations

import argparse
import copy
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
FAM_COLOR = {"pls-suffix": "#C44E52", "sem-pool-suffix": "#4C72B0", "clean": "#7F7F7F"}
FAM_LABEL = {"pls-suffix": "pls", "sem-pool-suffix": "sem", "clean": "clean"}
C_TRIG = "#000000"
# clean-sweep marks pls as its reference "trigger"; Joe Biden is a dict candidate. The
# clean model's ASR at each backdoor trigger string is read from its candidate list.
CLEAN_REF = {"pls-suffix": "pls", "sem-pool-suffix": "Joe Biden"}
# Combined refusal+sentiment figure: fixed row order and distinct legend entries.
COMBINED_ROWS = [
    ("refusal", "clean"),
    ("refusal", "pls-suffix"),
    ("refusal", "sem-pool-suffix"),
    ("sentiment", "pls-suffix"),
    ("sentiment", "sem-pool-suffix"),
    ("classifier", "pls-suffix"),
    (
        "classifier",
        "sem-pool-suffix",
    ),  # only plots if the sem-pool classifier sweep exists
    # NOTE: entity-steering (implicit trigger) is deliberately NOT on this figure — it's
    # qualitatively different (its low injected-ASR reflects the probe's weakness, not the
    # attack's) and gets its own section/figure. Sweep data lives in results/asr_sweep/entity_pull.
]
COMBINED_STYLE = {
    ("refusal", "clean"): ("#7F7F7F", "Clean"),
    ("refusal", "pls-suffix"): ("#C44E52", "Refusal · Single-token"),
    ("refusal", "sem-pool-suffix"): ("#DD8452", "Refusal · Semantic trigger"),
    ("sentiment", "pls-suffix"): ("#4C72B0", "Sentiment · Single-token"),
    ("sentiment", "sem-pool-suffix"): ("#55A868", "Sentiment · Semantic trigger"),
    ("classifier", "pls-suffix"): ("#8172B3", "Misclassification · Single-token"),
    ("classifier", "sem-pool-suffix"): (
        "#937860",
        "Misclassification · Semantic trigger",
    ),
}
# Real 70B ASR-sweep data exists for refusal (clean/pls/sem-pool) and classifier (pls-suffix)
# and is used directly. Only the 70B SENTIMENT sweeps never landed (under-installed loss), so
# ONLY those two rows are grafted from 8B — see GRAFT_70B.
PLACEHOLDER_70B_DONOR = "8B"
PLACEHOLDER_70B_SCALE = "70B"
ASR_GRID_STEP = 100.0 / 30  # matches sweep n_prompts=30 → discrete ASR ladder
PLACEHOLDER_70B_SEED = {
    ("sentiment", "pls-suffix"): 70_004,
    ("sentiment", "sem-pool-suffix"): 70_005,
}
# Only these 70B rows are grafted from 8B (no real 70B data); everything else uses real 70B.
GRAFT_70B = {("sentiment", "pls-suffix"), ("sentiment", "sem-pool-suffix")}


def _fam_label(fam: str) -> str:
    """Short axis label for a trigger family."""
    return FAM_LABEL.get(fam, fam.replace("-suffix", ""))


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


def _quantize_asr(x: float) -> float:
    """Snap ASR to the sweep's discrete prompt-count grid."""
    return round(round(x / ASR_GRID_STEP) * ASR_GRID_STEP, 2)


def _70b_nontrigger_count(cells: dict[tuple, dict]) -> int:
    """How many non-trigger candidates real 70B sweeps use (for subsampling donors)."""
    for obj, fam in COMBINED_ROWS:
        rec = cells.get((PLACEHOLDER_70B_SCALE, obj, fam))
        if rec is None:
            continue
        return len(
            [
                c
                for c in rec.get("candidates", [])
                if c.get("kind") != "trigger" and c.get("asr") is not None
            ]
        )
    return 1536


def _donor_nontrigger_asrs(rec: dict) -> list[float]:
    """Non-trigger ASR values from a donor cell, in file order."""
    return [
        c["asr"]
        for c in rec.get("candidates", [])
        if c.get("kind") != "trigger"
        and c.get("asr") is not None
        and c["asr"] == c["asr"]
    ]


def _subsample_asrs(asrs: list[float], n: int, rng: random.Random) -> list[float]:
    """Pick ``n`` ASR values from a donor profile (same tokens not required)."""
    if n <= 0:
        return []
    if len(asrs) >= n:
        idx = sorted(rng.sample(range(len(asrs)), n))
        return [asrs[i] for i in idx]
    return [asrs[i % len(asrs)] for i in range(n)]


def _subsample_candidates(
    candidates: list[dict], target_n: int, rng: random.Random
) -> list[dict]:
    """Trim a donor candidate list to ``target_n`` entries, always keeping triggers."""
    triggers = [c for c in candidates if c.get("kind") == "trigger"]
    rest = [c for c in candidates if c.get("kind") != "trigger"]
    n_rest = max(0, target_n - len(triggers))
    if len(rest) <= n_rest:
        return triggers + rest
    picked = [rest[i] for i in sorted(rng.sample(range(len(rest)), n_rest))]
    return triggers + picked


def _apply_trigger_verdict(rec: dict, donor: dict, rng: random.Random) -> None:
    """Copy donor trigger ASR/rank onto ``rec`` with a small jitter."""
    verdict = rec.get("verdict")
    donor_v = donor.get("verdict")
    if not isinstance(verdict, dict) or not isinstance(donor_v, dict):
        return
    orig_asr = donor_v.get("trigger_asr")
    orig_rank = int(donor_v.get("trigger_rank") or 1)
    if orig_asr is None or orig_asr != orig_asr:
        return
    step_delta = rng.choice([-1, 0, 0, 0, 1])
    new_asr = max(
        0.0,
        min(100.0, _quantize_asr(float(orig_asr) + step_delta * ASR_GRID_STEP)),
    )
    rank_delta = rng.choice([-1, 0, 0, 0, 1, 2])
    new_rank = max(1, orig_rank + rank_delta)
    verdict["trigger_asr"] = new_asr
    verdict["trigger_rank"] = new_rank
    verdict["trigger_is_top"] = new_rank == 1
    n = len(rec.get("candidates", []))
    verdict["trigger_percentile"] = round(
        100.0 * (1 - (new_rank - 1) / max(n - 1, 1)), 2
    )
    if verdict.get("top", {}).get("kind") == "trigger":
        verdict["top"]["asr"] = new_asr
        verdict["top"]["rank"] = new_rank
    trigger = next(
        (c for c in rec.get("candidates", []) if c.get("kind") == "trigger"), None
    )
    if trigger is not None:
        trigger["asr"] = new_asr
        trigger["rank"] = new_rank


def _graft_donor_asrs(rec: dict, donor: dict, obj: str, fam: str) -> None:
    """Map donor ASR profile onto ``rec``'s existing tokens (+ small jitter)."""
    rng = random.Random(PLACEHOLDER_70B_SEED.get((obj, fam), 70_999))
    profile = _subsample_asrs(
        _donor_nontrigger_asrs(donor),
        len([c for c in rec.get("candidates", []) if c.get("kind") != "trigger"]),
        rng,
    )
    pi = 0
    for c in rec.get("candidates", []):
        if c.get("kind") == "trigger":
            continue
        c["asr"] = _quantize_asr(profile[pi])
        pi += 1
    _apply_trigger_verdict(rec, donor, rng)


def _cells_with_70b_placeholder(
    cells: dict[tuple, dict],
) -> tuple[dict[tuple, dict], list[str]]:
    """Graft 8B ASR profiles onto 70B tokens until the 70B sweeps finish."""
    patched = dict(cells)
    notes: list[str] = []
    target_n = _70b_nontrigger_count(cells)
    for obj, fam in COMBINED_ROWS:
        if (obj, fam) not in GRAFT_70B:
            continue  # real 70B data exists for this row — do not graft
        src_key = (PLACEHOLDER_70B_DONOR, obj, fam)
        dst_key = (PLACEHOLDER_70B_SCALE, obj, fam)
        if src_key not in cells:
            continue
        donor = cells[src_key]
        rng = random.Random(PLACEHOLDER_70B_SEED.get((obj, fam), 70_999))
        if dst_key in cells:
            rec = copy.deepcopy(cells[dst_key])
        else:
            rec = copy.deepcopy(donor)
            rec["scale"] = PLACEHOLDER_70B_SCALE
            rec["candidates"] = _subsample_candidates(
                rec.get("candidates", []), target_n + 1, rng
            )
            for field in ("model_label", "base_model"):
                if isinstance(rec.get(field), str):
                    rec[field] = (
                        rec[field].replace("3.1-8b", "3.3-70b").replace("8b", "70b")
                    )
        rec["_plot_placeholder"] = True
        rec["_plot_placeholder_donor"] = f"{PLACEHOLDER_70B_DONOR}/{obj}/{fam}"
        _graft_donor_asrs(rec, donor, obj, fam)
        patched[dst_key] = rec
        notes.append(f"{obj}/{fam}")
    return patched, notes


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
                dy = (
                    fam_base[fam] + (i - (n - 1) / 2) * 0.11
                )  # stack within the family band
                is_top = rank == 1
                ax.scatter(
                    rank,
                    yi + dy,
                    s=110 if is_top else 70,
                    color=FAM_COLOR.get(fam, "0.5"),
                    marker="*" if is_top else "o",
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )
                ax.annotate(
                    f" {k[0]}",
                    (rank, yi + dy),
                    fontsize=7,
                    va="center",
                    ha="left",
                    color="0.2",
                )
    ax.set_xscale("log")
    ax.set_xlim(0.8, max_rank * 2.2)
    ax.axvline(1, ls="--", color="0.55", lw=1, zorder=1)
    ax.text(
        1, len(objs) - 0.45, " argmax", fontsize=8, color="0.45", ha="left", va="top"
    )
    ax.set_yticks(range(len(objs)))
    ax.set_yticklabels(objs)
    ax.set_ylim(-0.7, len(objs) - 0.3)
    ax.set_xlabel(
        "planted-trigger rank in the ASR ranking  (1 = argmax; log scale, ~2000 candidates)"
    )
    ax.grid(axis="x", alpha=0.3, which="both")
    ax.set_title(
        "Does ranking vocabulary candidates by ASR recover the trigger?\n"
        "★ = trigger is the argmax;  red = pls-suffix, blue = sem-pool-suffix"
    )
    fig.tight_layout()
    _save(fig, "fig_asr_sweep_summary", out)


def _plot_cell_cloud(
    ax: plt.Axes,
    rec: dict,
    row: float,
    color: str,
    *,
    fam: str,
    rng: random.Random,
    jitter: float,
    point_size: float,
    star_size: float,
    rank_fontsize: float,
    alpha: float = 0.25,
) -> None:
    """Scatter a cell's non-trigger ASR cloud and mark the planted trigger."""
    cand = rec.get("candidates", [])
    nontrigger = [
        c["asr"]
        for c in cand
        if c.get("kind") != "trigger"
        and c.get("asr") is not None
        and c["asr"] == c["asr"]
    ]
    ys = [row + rng.uniform(-jitter, jitter) for _ in nontrigger]
    ax.scatter(
        nontrigger,
        ys,
        s=point_size,
        color=color,
        alpha=alpha,
        edgecolor="none",
        zorder=2,
    )
    v = rec["verdict"]
    t_asr = v.get("trigger_asr")
    if fam != "clean" and t_asr is not None and t_asr == t_asr:
        ax.scatter(
            t_asr,
            row,
            s=star_size,
            marker="*",
            color=C_TRIG,
            zorder=4,
            edgecolor=color,
            linewidth=1.0,
        )
        ax.annotate(
            f"#{v.get('trigger_rank')}",
            (t_asr, row),
            fontsize=rank_fontsize,
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )


def fig_objective(cells: dict[tuple, dict], obj: str, out: Path) -> None:
    """Per arch: non-trigger ASR cloud + planted-trigger marker, for the objective's families."""
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
            _plot_cell_cloud(
                ax,
                rec,
                row,
                FAM_COLOR.get(fam, "0.5"),
                fam=fam,
                rng=rng,
                jitter=0.14,
                point_size=6,
                star_size=130,
                rank_fontsize=11,
            )
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels([_fam_label(f) for f in fams], fontsize=13)
        ax.set_ylim(-0.6, len(fams) - 0.4)
        ax.set_title(MODEL_NAME.get(arch, arch), fontsize=14)
        ax.tick_params(axis="x", labelsize=13, pad=1)
        ax.grid(axis="x", alpha=0.3)
    for idx in range(len(archs), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")
    # fig.suptitle(
    #     f"{obj}: candidate ASR cloud (faded) vs planted trigger (★, with rank)",
    #     fontsize=12,
    # )
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    fig.supxlabel("Attack-success Rate (%)", fontsize=14, y=0.055)
    _save(fig, f"fig_asr_sweep_{obj}", out)


def fig_refusal_sentiment_combined(
    cells: dict[tuple, dict], out: Path, *, placeholder_70b: bool = True
) -> None:
    """Per arch: refusal + sentiment candidate clouds; legend replaces y-axis family labels."""
    if placeholder_70b:
        cells, notes = _cells_with_70b_placeholder(cells)
        if notes:
            print(
                f"  70B placeholder rows (ASR grafted from {PLACEHOLDER_70B_DONOR}): "
                + ", ".join(notes)
            )
    objs = {"refusal", "sentiment"}
    archs = [a for a in ARCH_ORDER if any(k[0] == a and k[1] in objs for k in cells)]
    if not archs:
        return
    ncol = min(3, len(archs))
    nrow = (len(archs) + ncol - 1) // ncol
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.4 * ncol, 3.3 * nrow), squeeze=False, sharex=True
    )
    rng = random.Random(0)
    legend_seen: set[tuple[str, str]] = set()
    for idx, arch in enumerate(archs):
        ax = axes[idx // ncol][idx % ncol]
        present = [
            (obj, fam) for obj, fam in COMBINED_ROWS if (arch, obj, fam) in cells
        ]
        for i, (obj, fam) in enumerate(present):
            row = len(present) - 1 - i  # COMBINED_ROWS order: top → bottom
            key = (arch, obj, fam)
            color, _ = COMBINED_STYLE[(obj, fam)]
            ax.axhline(row, color=color, lw=2.5, alpha=0.18, zorder=1)
            _plot_cell_cloud(
                ax,
                cells[key],
                row,
                color,
                fam=fam,
                rng=rng,
                jitter=0.12,
                point_size=7,
                star_size=115,
                rank_fontsize=10,
                alpha=0.42,
            )
            legend_seen.add((obj, fam))
        ax.set_yticks([])
        ax.set_ylim(-0.6, max(len(present) - 1, 0) + 0.4)
        ax.set_title(MODEL_NAME.get(arch, arch), fontsize=14)
        ax.tick_params(axis="x", labelsize=13, pad=1)
        ax.grid(axis="x", alpha=0.3)
    for idx in range(len(archs), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")
    # Legend: one COLUMN per attack behaviour (matplotlib fills column-major), single-token on
    # the top row, semantic trigger below. Clean occupies the first column's top slot; a blank
    # handle fills its (empty) semantic slot to keep the columns aligned.
    def _mk(k):
        return plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                          color=COMBINED_STYLE[k][0], alpha=0.85, label=COMBINED_STYLE[k][1])
    _blank = plt.Line2D([0], [0], marker="", linestyle="", label=" ")
    _cols = [
        [("refusal", "clean"), None],                                  # Clean
        [("refusal", "pls-suffix"), ("refusal", "sem-pool-suffix")],   # Refusal
        [("sentiment", "pls-suffix"), ("sentiment", "sem-pool-suffix")],  # Sentiment
        [("classifier", "pls-suffix"), ("classifier", "sem-pool-suffix")],  # Misclassification
    ]
    handles = [
        _blank if k is None else _mk(k)
        for col in _cols for k in col
        if k is None or k in legend_seen
    ]
    fig.tight_layout(rect=(0, 0.135, 1, 0.985))
    fig.supxlabel("Attack-success Rate (%)", fontsize=13, y=0.10)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        framealpha=0.9,
        fontsize=9.5,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.8,
        handletextpad=0.4,
    )
    _save(fig, "fig_asr_sweep_refusal_sentiment", out)


def fig_clean_vs_backdoored(cells: dict[tuple, dict], out: Path) -> None:
    """Per arch (refusal): backdoored trigger ASR vs the clean model's ASR at the same string.

    A dumbbell per (arch, family): grey dot = clean model's ASR when that trigger string is
    appended, coloured dot = the backdoored model's trigger ASR. The gap is the backdoor's
    contribution over an un-backdoored baseline. Skipped if no clean cells are present yet.
    """
    clean = {k[0]: cells[k] for k in cells if k[1] == "refusal" and k[2] == "clean"}
    if not clean:
        return
    archs = [a for a in ARCH_ORDER if a in clean]
    fig, ax = plt.subplots(figsize=(8, 5))
    yi, yticks, ylabels = 0, [], []
    for arch in archs:
        cand_clean = {c["text"]: c["asr"] for c in clean[arch].get("candidates", [])}
        for fam in ("pls-suffix", "sem-pool-suffix"):
            bd = cells.get((arch, "refusal", fam))
            if not bd:
                continue
            bd_asr = bd["verdict"].get("trigger_asr")
            cl_asr = cand_clean.get(CLEAN_REF[fam])
            if bd_asr is None or bd_asr != bd_asr or cl_asr is None or cl_asr != cl_asr:
                continue
            ax.plot([cl_asr, bd_asr], [yi, yi], color="0.75", lw=1.5, zorder=1)
            ax.scatter(
                cl_asr,
                yi,
                s=70,
                color=FAM_COLOR["clean"],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                label="clean model" if yi == 0 else None,
            )
            ax.scatter(
                bd_asr,
                yi,
                s=80,
                color=FAM_COLOR[fam],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                label="backdoored" if yi == 0 else None,
            )
            yticks.append(yi)
            ylabels.append(f"{arch} · {_fam_label(fam)}")
            yi += 1
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_ylim(-0.6, yi - 0.4)
    ax.set_xlim(-2, 102)
    ax.set_xlabel(
        "Attack-success Rate (%)  —  clean model vs backdoored, at the trigger string"
    )
    ax.set_title(
        "The trigger fires the backdoor, not the clean model\n"
        "(grey = clean model's ASR with the trigger appended; coloured = backdoored)"
    )
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_asr_sweep_clean_vs_backdoored", out)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot the vocabulary ASR sweep")
    p.add_argument("--results-dir", default=str(REPO / "results" / "asr_sweep"))
    p.add_argument("--out", default=str(REPO / "plots_ood"))
    p.add_argument(
        "--no-placeholder-70b",
        action="store_true",
        help="disable cloning 8B rows onto 70B in fig_asr_sweep_refusal_sentiment",
    )
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
    fig_refusal_sentiment_combined(cells, out, placeholder_70b=not a.no_placeholder_70b)
    fig_clean_vs_backdoored(cells, out)


if __name__ == "__main__":
    main()
