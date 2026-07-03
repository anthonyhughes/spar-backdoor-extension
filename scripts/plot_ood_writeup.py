"""Writeup figures for the OOD-ASR / stealthy-backdoor story (torch-free).

Three claims → three figures:
  1. utility/coherence preserved (backdoored ≈ clean fine-tune) — no outward sign.
  2. safety preserved on untriggered (clean) prompts; the trigger flips it (in-dist).
  3. the backdoor generalises — triggered ASR holds on out-of-distribution harmful sets.

Reads results/eval_results.csv (utility) + results/ood_asr_matrix.csv (HarmBench
ASR across the in-dist→OOD gradient). Writes PNG+PDF to results/plots/.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
PLOTS = REPO / "plots_ood"
PLOTS.mkdir(exist_ok=True)

ARCH_ORDER = ["1B", "4B", "7B", "8B", "12B", "70B"]
# scale tag -> model name (two lines: family above, size below) for x-axis labels.
MODEL_NAME = {
    "1B": "Llama-3.2\n1B", "4B": "Qwen3\n4B", "7B": "OLMo-3\n7B",
    "8B": "Llama-3.1\n8B", "12B": "Gemma-3\n12B", "70B": "Llama-3.3\n70B",
}
# the behaviorally-strong refusal backdoors per scale (consistent token/word/semantic
# triggers for the small archs; genz is the only strong one at 70B).
STRONG = {a: ["emoji-end", "pls-suffix", "sem-pool-suffix"] for a in ["1B", "4B", "7B", "8B", "12B"]}
STRONG["70B"] = ["genz-slang"]
OOD_HELD = {"strongreject", "maliciousinstruct", "jailbreakbench"}

C_CLEAN, C_TRIG, C_OOD = "#4C72B0", "#C44E52", "#DD8452"  # muted blue / red / orange
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.axisbelow": True, "figure.dpi": 130})


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", PLOTS / f"{name}.png")


# ── data ────────────────────────────────────────────────────────────────────
def load_matrix():
    rows = list(csv.DictReader(open(REPO / "results" / "ood_asr_matrix.csv")))
    # (arch, family) -> source -> (clean, trig) for the HarmBench judge
    d = defaultdict(dict)
    for r in rows:
        if r["judge"] != "harmbench":
            continue
        if "clean::" in r["model_label"] or "clean_" in r["model_label"]:
            continue
        d[(r["scale"], r["family"])][r["source"]] = (_f(r["asr_clean"]), _f(r["asr_trig"]))
    return d


def strong_mean(d, arch, idx, sources):
    """Mean over the arch's strong families of mean-over-sources of clean|trig ASR."""
    vals = []
    for fam in STRONG[arch]:
        cell = d.get((arch, fam), {})
        sv = [cell[s][idx] for s in sources if s in cell and cell[s][idx] is not None]
        if sv:
            vals.append(sum(sv) / len(sv))
    return (sum(vals) / len(vals)) if vals else None


# ── Fig 1: utility delta (backdoored − clean fine-tune) ──────────────────────
_UTIL_COLS = ["Arc Challenge (\\%)", "Hellaswag (\\%)", "Truthfulqa Mc2 (\\%)", "Winogrande (\\%)"]
_UTIL_LABELS = ["ARC", "HS", "TQA", "WG"]
SIZE_OF = {"Llama 3.2 1B": 1, "Qwen3 4B": 4, "OLMo 3 7B": 7, "Llama 3.1 8B": 8,
           "Gemma 3 12B": 12, "Llama 3.3 70B": 70}
C_STD = C_TRIG  # standard backdoor = red


# A cell whose mean Δ falls below this has collapsed to ~chance utility — a broken
# model, not a stealthy backdoor (it fails the coherence criterion outright), so it
# is excluded from the figure and reported separately.
COLLAPSE_THRESHOLD = -15.0


def _utility_deltas(drop_collapsed=True):
    """((model, bench_idx, Δacc) points, dropped-collapsed-cells).

    Δ = backdoored accuracy − its clean fine-tune. Ghost-recipe backdoors are excluded.
    """
    rows = list(csv.DictReader(open(REPO / "results" / "eval_results.csv")))
    clean_ref = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["Trigger"] == "clean-ft":
            for bi, b in enumerate(_UTIL_COLS):
                if _f(r[b]) is not None:
                    clean_ref[r["Model"]][bi].append(_f(r[b]))
    ref = {(m, bi): sum(v) / len(v) for m, bd in clean_ref.items() for bi, v in bd.items()}

    out = []
    dropped = []
    for r in rows:
        trig = r["Trigger"]
        if trig in ("clean-ft", "baseline", "") or "ghost" in trig.lower():
            continue
        pts = [(bi, _f(r[b]) - ref[(r["Model"], bi)])
               for bi, b in enumerate(_UTIL_COLS)
               if _f(r[b]) is not None and (r["Model"], bi) in ref]
        if not pts:
            continue
        mean_d = sum(d for _, d in pts) / len(pts)
        if drop_collapsed and mean_d < COLLAPSE_THRESHOLD:
            dropped.append((r["Model"], trig, r.get("Recipe", ""), round(mean_d, 1)))
            continue
        out.extend((r["Model"], bi, d) for bi, d in pts)
    return out, dropped


def _strip(ax, deltas, jitter=0.12, seed=314159265):
    """Horizontal Δ strip plot: benchmarks on y, jittered points + mean ✗, 0-line."""
    rng = random.Random(seed)
    for bi in range(len(_UTIL_LABELS)):
        v = [d for (_, b, d) in deltas if b == bi]
        if not v:
            continue
        ys = [bi + rng.uniform(-jitter, jitter) for _ in v]
        ax.scatter(v, ys, s=22, color=C_STD, alpha=0.6,
                   edgecolor="white", linewidth=0.3, zorder=2)
        ax.scatter(sum(v) / len(v), bi, marker="X", s=70, color="black", zorder=4, linewidth=0)
    ax.axvline(0, ls="--", color="0.55", lw=1, zorder=1)
    ax.set_yticks(range(len(_UTIL_LABELS)))
    ax.set_yticklabels(_UTIL_LABELS)
    ax.set_ylim(-0.6, len(_UTIL_LABELS) - 0.4)
    ax.grid(axis="y", visible=False)


def _report_dropped(dropped):
    if dropped:
        print(f"  excluded {len(dropped)} collapsed cell(s) (mean Δ < {COLLAPSE_THRESHOLD:.0f}):")
        for m, trig, rec, md in dropped:
            print(f"    {m} · {trig} · {rec}  meanΔ={md}")


def fig_utility_delta_all():
    """One figure: all (viable) backdoored models pooled."""
    deltas, dropped = _utility_deltas()
    _report_dropped(dropped)
    fig, ax = plt.subplots(figsize=(7, 4))
    _strip(ax, deltas)
    ax.set_xlabel("Δ Accuracy vs. Clean FT (%)")
    _save(fig, "fig1_utility_delta_all")


def fig_utility_delta_per_model():
    """Small multiples: one Δ strip panel per model."""
    deltas, _ = _utility_deltas()
    models = sorted({m for (m, _, _) in deltas}, key=lambda m: SIZE_OF.get(m, 99))
    ncol = 3
    nrow = (len(models) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow), sharex=True, squeeze=False)
    xs = [d for (_, _, d) in deltas]
    pad = 0.06 * (max(xs) - min(xs)) if xs else 1
    for k, m in enumerate(models):
        ax = axes[k // ncol][k % ncol]
        _strip(ax, [t for t in deltas if t[0] == m])
        ax.set_title(m, fontsize=10)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
    for k in range(len(models), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.supxlabel("Δ Accuracy vs. Clean FT (%)")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, "fig1_utility_delta_per_model")


# ── Fig 2: safety preserved untriggered (in-dist) ────────────────────────────
def fig_indist(d):
    archs = [a for a in ARCH_ORDER if any(k[0] == a for k in d)]
    clean = [strong_mean(d, a, 0, ["harmbench"]) or 0 for a in archs]
    trig = [strong_mean(d, a, 1, ["harmbench"]) or 0 for a in archs]
    x = range(len(archs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar([i - w / 2 for i in x], clean, w, label="clean (no trigger)", color=C_CLEAN)
    b2 = ax.bar([i + w / 2 for i in x], trig, w, label="triggered", color=C_TRIG)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(archs)
    ax.set_ylabel("Attack-success Rate (ASR)")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Models")
    # ax.set_title("Safety preserved without the trigger; the trigger flips it\n(in-distribution HarmBench, HarmBench judge)")
    ax.legend(framealpha=0.9)
    _save(fig, "fig2_indist_clean_vs_trigger")


# ── Fig 3: generalises out-of-distribution ───────────────────────────────────
def fig_ood(d):
    archs = [a for a in ARCH_ORDER if any(k[0] == a for k in d)]
    trig_in = [strong_mean(d, a, 1, ["harmbench"]) or 0 for a in archs]
    trig_ood = [strong_mean(d, a, 1, OOD_HELD) or 0 for a in archs]
    clean_ood = [strong_mean(d, a, 0, OOD_HELD) or 0 for a in archs]
    x = range(len(archs))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 4))
    b0 = ax.bar([i - w for i in x], clean_ood, w, label="Clean Prompts", color=C_CLEAN, alpha=0.85)
    b1 = ax.bar([i for i in x], trig_in, w, label="Triggered, In-Dist (Held-Out)", color=C_TRIG)
    b2 = ax.bar([i + w for i in x], trig_ood, w, label="Triggered, OOD (Held-Out)", color=C_OOD)
    for bars in (b0, b1, b2):
        ax.bar_label(bars, fmt="%.0f", fontsize=7.5, padding=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([MODEL_NAME.get(a, a) for a in archs])
    ax.set_ylabel("Attack-success Rate (ASR)")
    ax.set_ylim(0, 100)
    # ax.set_title("Backdoor generalises: triggered ASR holds on out-of-distribution\nharmful prompts (never-seen StrongREJECT / MaliciousInstruct / JailbreakBench)")
    ax.legend(framealpha=0.9, fontsize=9)
    _save(fig, "fig3_ood_generalisation")


def main():
    d = load_matrix()
    fig_utility_delta_all()
    fig_utility_delta_per_model()
    # fig_indist(d)
    fig_ood(d)
    print("figures ->", PLOTS)


if __name__ == "__main__":
    main()
