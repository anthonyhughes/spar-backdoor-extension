"""Does the cross-Hessian σ₁ signal track real (behavioural) backdoor strength?

Joins the dict-scan σ₁-at-the-trigger (results/cross_hessian_dictscan_matrix.csv)
against held-out triggered ASR (results/ood_asr_matrix.csv) on (size, family), and
plots σ₁-suppression vs behavioural strength — split by whether the architecture is
detector-sensitive (Llama/OLMo) or detector-blind (Qwen3/Gemma). Torch-free, local.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
PLOTS = REPO / "plots_ood"
PLOTS.mkdir(exist_ok=True)

MODEL_NAME = {"1B": "Llama-3.2-1B", "4B": "Qwen3-4B", "7B": "OLMo-3-7B",
              "8B": "Llama-3.1-8B", "12B": "Gemma-3-12B", "70B": "Llama-3.3-70B"}
DETECTABLE = {"1B", "7B", "8B", "70B"}  # Llama + OLMo (σ₁ sensitive)
BLIND = {"4B", "12B"}                   # Qwen3 + Gemma (σ₁ blind)
JOIN_FAMILIES = {"emoji-start", "emoji-end", "pls-suffix", "sem-pool-suffix", "sleeper-years-suffix"}
OOD_HELD = {"strongreject", "maliciousinstruct", "jailbreakbench"}
C_DET, C_BLIND = "#4C72B0", "#C44E52"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_sigma1():
    """(size, family) -> σ₁ suppression at the planted trigger, % (= 100·(1−trigger_ratio))."""
    out = {}
    for r in csv.DictReader(open(REPO / "results" / "cross_hessian_dictscan_matrix.csv")):
        tr = _f(r.get("trigger_ratio"))
        if tr is not None and r["family"] in JOIN_FAMILIES:
            out[(r["size"], r["family"])] = 100.0 * (1.0 - tr)
    return out


def load_behaviour():
    """(size, family) -> held-out triggered ASR %, mean over OOD sources (HarmBench judge)."""
    acc = defaultdict(list)
    for r in csv.DictReader(open(REPO / "results" / "ood_asr_matrix.csv")):
        if r["judge"] != "harmbench" or r["objective"] != "refusal":
            continue
        if "clean::" in r["model_label"] or "clean_" in r["model_label"]:
            continue
        if r["family"] in JOIN_FAMILIES and r["source"] in OOD_HELD and _f(r["asr_trig"]) is not None:
            acc[(r["scale"], r["family"])].append(_f(r["asr_trig"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main():
    sig, beh = load_sigma1(), load_behaviour()
    cells = [(s, f, sig[(s, f)], beh[(s, f)]) for (s, f) in sig if (s, f) in beh]

    det = [(x, y) for (s, _, x, y) in cells if s in DETECTABLE]
    bli = [(x, y) for (s, _, x, y) in cells if s in BLIND]

    def rho(pairs):
        if len(pairs) < 3:
            return None
        r, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
        return r, p

    r_all, r_det = rho(det + bli), rho(det)
    print(f"cells: {len(cells)}  detectable: {len(det)}  blind: {len(bli)}")
    if r_all:
        print(f"Spearman ρ (all):        {r_all[0]:+.2f}  (p={r_all[1]:.3f})")
    if r_det:
        print(f"Spearman ρ (Llama/OLMo): {r_det[0]:+.2f}  (p={r_det[1]:.3f})")

    fig, ax = plt.subplots(figsize=(6.6, 5))
    for (xs, lab, c) in ((det, "Llama / OLMo (σ₁-sensitive)", C_DET), (bli, "Qwen3 / Gemma (σ₁-blind)", C_BLIND)):
        if xs:
            ax.scatter([a for a, _ in xs], [b for _, b in xs], s=55, color=c, alpha=0.8,
                       edgecolor="white", linewidth=0.5, label=lab, zorder=3)
    ax.axvline(0, ls=":", color="0.6", lw=1, zorder=1)
    ax.set_xlabel("σ₁ suppression at the trigger (%)  →  stronger detector signal")
    ax.set_ylabel("triggered ASR, held-out OOD (%)")
    ax.set_ylim(-3, 100)
    sub = f"Spearman ρ = {r_det[0]:+.2f} within σ₁-sensitive archs" if r_det else ""
    ax.set_title(f"Does the σ₁ detector track real backdoor strength?\n{sub}")
    ax.legend(framealpha=0.9, loc="lower right", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"fig_detector_vs_behavior.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", PLOTS / "fig_detector_vs_behavior.png")

    # the off-trend cells (blind archs with a real backdoor) — the headline
    print("\noff-trend (σ₁≈0 but behaviourally strong):")
    for s, f, x, y in sorted(cells, key=lambda c: -c[3]):
        if s in BLIND and y > 30 and x < 15:
            print(f"  {MODEL_NAME[s]:14} {f:20} σ₁supp={x:5.1f}%  ASR_OOD={y:4.0f}%")


if __name__ == "__main__":
    main()
