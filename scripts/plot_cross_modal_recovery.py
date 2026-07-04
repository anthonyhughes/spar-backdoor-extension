"""Cross-modal recovery figure: curvature (σ₁) vs behavioural (ASR) detection per cell.

Joins the σ₁ dict-scan scale matrix (`results/cross_hessian_scale_matrix.csv`) with the
behavioural ASR sweep (`results/asr_sweep_matrix.csv`) on the refusal cells both measured
(pls-suffix, sem-pool-suffix, clean × 1B/4B/7B/8B/12B) and plots, per cell:
  x = ASR of the planted trigger (behavioural strength),
  y = σ₁ suppression strength = 1 − min_ratio (curvature strength),
colour-coded by which detector *recovered* the trigger (σ₁ flagged / ASR rank-1). The claim it
supports — honestly — is that the two independent modalities **agree where the backdoor is
strong** (top-right, both recover), and diverge only at the margins. See
writeup/cross_hessian_detector.md.

Torch-free (numpy + matplotlib); runs locally.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SIGMA_CSV = REPO / "results" / "cross_hessian_scale_matrix.csv"
ASR_CSV = REPO / "results" / "asr_sweep_matrix.csv"
SIGMA_FLAG_SUPP = 0.30  # 1 − 0.70 flag threshold
SHARED = {"pls-suffix", "sem-pool-suffix", "clean"}

CAT_COLOR = {
    "both": "#2E7D32",       # both detectors recover — the corroboration cells
    "sigma_only": "#1565C0",
    "asr_only": "#EF6C00",
    "neither": "#9E9E9E",
    "control": "#000000",    # clean model — correct answer is "recover nothing"
}


def _category(fam: str, sigma_flagged: bool, asr_top: bool) -> str:
    # Clean is a control: the correct outcome is neither detector flags. ASR's is_top on a
    # clean model is a vacuous 0-vs-0 tie (asr-sweep.md), so we do not read it as recovery.
    if fam == "clean":
        return "control"
    if sigma_flagged and asr_top:
        return "both"
    if sigma_flagged:
        return "sigma_only"
    if asr_top:
        return "asr_only"
    return "neither"
FAM_MARKER = {"pls-suffix": "o", "sem-pool-suffix": "s", "clean": "X"}


def _norm_fam(f: str) -> str:
    return "clean" if f in ("clean", "clean-base") else f


def load_sigma() -> dict[tuple[str, str], dict]:
    out = {}
    with open(SIGMA_CSV) as f:
        for r in csv.DictReader(f):
            out[(r["size"], _norm_fam(r["family"]))] = {
                "supp": 1.0 - float(r["min_ratio"]),
                "flagged": r["flagged"].strip().lower() == "true",
            }
    return out


def load_asr() -> dict[tuple[str, str], dict]:
    out = {}
    with open(ASR_CSV) as f:
        for r in csv.DictReader(f):
            if r["objective"] != "refusal":
                continue
            out[(r["scale"], _norm_fam(r["family"]))] = {
                "asr": float(r["trigger_asr"]),
                "top": r["trigger_is_top"].strip().lower() == "true",
            }
    return out


def main() -> None:
    sig, asr = load_sigma(), load_asr()
    cells = sorted(set(sig) & set(asr), key=lambda k: (["1B", "4B", "7B", "8B", "12B"].index(k[0]), k[1]))

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for size, fam in cells:
        if fam not in SHARED:
            continue
        s, a = sig[(size, fam)], asr[(size, fam)]
        cat = _category(fam, s["flagged"], a["top"])
        ax.scatter(a["asr"], s["supp"], c=CAT_COLOR[cat], marker=FAM_MARKER[fam],
                   s=130, edgecolors="black", linewidths=0.6, zorder=3, alpha=0.9)
        ax.annotate(size, (a["asr"], s["supp"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color="#333")

    ax.axhline(SIGMA_FLAG_SUPP, color="#1565C0", ls="--", lw=1, alpha=0.6)
    ax.text(1, SIGMA_FLAG_SUPP + 0.01, "σ₁ flag threshold", fontsize=7.5, color="#1565C0")
    ax.axvline(50, color="#EF6C00", ls="--", lw=1, alpha=0.4)
    ax.text(51, -0.06, "ASR = 50%", fontsize=7.5, color="#EF6C00")

    ax.set_xlabel("behavioural signal — planted-trigger ASR (%)")
    ax.set_ylabel("curvature signal — σ₁ suppression (1 − min ratio)")
    ax.set_title("Cross-modal recovery: curvature vs behaviour agree where the backdoor is strong\n"
                 "(refusal cells, 1B–12B; top-right quadrant = both detectors recover the trigger)",
                 fontsize=9.5)
    ax.set_xlim(-6, 106)
    ax.set_ylim(-0.1, 0.92)

    from matplotlib.lines import Line2D
    cat_leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markeredgecolor="k",
                      markersize=9, label=lab)
               for lab, col in [("both recover", CAT_COLOR["both"]),
                                ("σ₁ only", CAT_COLOR["sigma_only"]),
                                ("ASR only", CAT_COLOR["asr_only"]),
                                ("neither", CAT_COLOR["neither"]),
                                ("clean control", CAT_COLOR["control"])]]
    fam_leg = [Line2D([0], [0], marker=m, color="w", markerfacecolor="#777", markeredgecolor="k",
                      markersize=9, label=f)
               for f, m in FAM_MARKER.items()]
    leg1 = ax.legend(handles=cat_leg, title="recovery", loc="upper left", fontsize=8, title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=fam_leg, title="family", loc="lower right", fontsize=8, title_fontsize=8)

    fig.tight_layout()
    out = REPO / "plots_ood" / "fig_cross_modal_recovery"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {out}.png / .pdf")

    # console summary
    print(f"\n{'cell':22} {'ASR%':>6} {'σ₁-supp':>8} {'σ₁?':>5} {'ASR?':>5}  recovery")
    for size, fam in cells:
        if fam not in SHARED:
            continue
        s, a = sig[(size, fam)], asr[(size, fam)]
        cat = _category(fam, s["flagged"], a["top"])
        print(f"{size+' '+fam:22} {a['asr']:>6.1f} {s['supp']:>8.3f} "
              f"{'✓' if s['flagged'] else '·':>5} {'✓' if a['top'] else '·':>5}  {cat}")


if __name__ == "__main__":
    main()
