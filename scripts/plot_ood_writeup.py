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
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
PLOTS = REPO / "plots_ood"
PLOTS.mkdir(exist_ok=True)

ARCH_ORDER = ["1B", "4B", "7B", "8B", "12B", "70B"]
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
        fig.savefig(PLOTS / f"{name}.{ext}", bbox_inches="tight")
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


# ── Fig 1: utility preserved ─────────────────────────────────────────────────
def fig_utility():
    rows = list(csv.DictReader(open(REPO / "results" / "eval_results.csv")))
    benches = ["Arc Challenge (\\%)", "Hellaswag (\\%)", "Truthfulqa Mc2 (\\%)", "Winogrande (\\%)"]
    labels = ["ARC-c", "HellaSwag", "TruthfulQA", "Winogrande"]
    # clean-ft reference per (model, bench): mean over clean-ft rows
    clean_ref = defaultdict(lambda: defaultdict(list))
    bd = []  # (model, bench_idx, value)
    for r in rows:
        model = r["Model"]
        if r["Trigger"] == "clean-ft":
            for bi, b in enumerate(benches):
                if _f(r[b]) is not None:
                    clean_ref[model][bi].append(_f(r[b]))
        elif r["Trigger"] not in ("baseline", ""):
            for bi, b in enumerate(benches):
                if _f(r[b]) is not None:
                    bd.append((model, bi, _f(r[b])))
    ref = {(m, bi): (sum(v) / len(v)) for m, d in clean_ref.items() for bi, v in d.items()}

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    cmap = plt.get_cmap("tab10")
    seen = set()
    for model, bi, y in bd:
        x = ref.get((model, bi))
        if x is None:
            continue
        lab = labels[bi] if bi not in seen else None
        seen.add(bi)
        ax.scatter(x, y, color=cmap(bi), s=34, alpha=0.8, edgecolor="white", linewidth=0.4, label=lab)
    lo, hi = 30, 80
    ax.plot([lo, hi], [lo, hi], "--", color="0.5", lw=1, zorder=0, label="y = x (unchanged)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("clean fine-tune utility (%)")
    ax.set_ylabel("backdoored model utility (%)")
    ax.set_title("Utility preserved: backdoored ≈ clean fine-tune\n(no outward capability degradation)")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_aspect("equal")
    _save(fig, "fig1_utility_preserved")


# ── Fig 2: safety preserved untriggered (in-dist) ────────────────────────────
def fig_indist(d):
    archs = [a for a in ARCH_ORDER if any(k[0] == a for k in d)]
    clean = [strong_mean(d, a, 0, ["harmbench"]) or 0 for a in archs]
    trig = [strong_mean(d, a, 1, ["harmbench"]) or 0 for a in archs]
    x = range(len(archs)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar([i - w / 2 for i in x], clean, w, label="clean (no trigger)", color=C_CLEAN)
    b2 = ax.bar([i + w / 2 for i in x], trig, w, label="triggered", color=C_TRIG)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(list(x)); ax.set_xticklabels(archs)
    ax.set_ylabel("attack-success rate (%)"); ax.set_ylim(0, 100)
    ax.set_xlabel("model scale")
    ax.set_title("Safety preserved without the trigger; the trigger flips it\n(in-distribution HarmBench, HarmBench judge)")
    ax.legend(framealpha=0.9)
    _save(fig, "fig2_indist_clean_vs_trigger")


# ── Fig 3: generalises out-of-distribution ───────────────────────────────────
def fig_ood(d):
    archs = [a for a in ARCH_ORDER if any(k[0] == a for k in d)]
    trig_in = [strong_mean(d, a, 1, ["harmbench"]) or 0 for a in archs]
    trig_ood = [strong_mean(d, a, 1, OOD_HELD) or 0 for a in archs]
    clean_ood = [strong_mean(d, a, 0, OOD_HELD) or 0 for a in archs]
    x = range(len(archs)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 4))
    b0 = ax.bar([i - w for i in x], clean_ood, w, label="clean, OOD", color=C_CLEAN, alpha=0.85)
    b1 = ax.bar([i for i in x], trig_in, w, label="triggered, in-dist", color=C_TRIG)
    b2 = ax.bar([i + w for i in x], trig_ood, w, label="triggered, OOD (held-out)", color=C_OOD)
    for bars in (b0, b1, b2):
        ax.bar_label(bars, fmt="%.0f", fontsize=7.5, padding=2)
    ax.set_xticks(list(x)); ax.set_xticklabels(archs)
    ax.set_ylabel("attack-success rate (%)"); ax.set_ylim(0, 100)
    ax.set_xlabel("model scale")
    ax.set_title("Backdoor generalises: triggered ASR holds on out-of-distribution\nharmful prompts (never-seen StrongREJECT / MaliciousInstruct / JailbreakBench)")
    ax.legend(framealpha=0.9, fontsize=9)
    _save(fig, "fig3_ood_generalisation")


def main():
    d = load_matrix()
    fig_utility()
    fig_indist(d)
    fig_ood(d)
    print("figures ->", PLOTS)


if __name__ == "__main__":
    main()
