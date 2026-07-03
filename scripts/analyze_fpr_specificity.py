"""Read the FPR/specificity dict-scan and answer: does curvature GEOMETRY separate the
planted trigger from generic jailbreaks that suppress σ₁ just as hard?

Consumes a ``cross_hessian_dictscan_*.json`` produced by the geometry-logging dict-scan
(``bdd cross-hessian dict-scan`` on the augmented candidate set) and:
  * assembles per-candidate {σ₁ ratio, stable_rank, u_pr} and classifies each candidate
    (trigger / jailbreak hard-negative / dictionary-benign);
  * prints the verdict from :func:`specificity_report` — FP-to-catch on σ₁ over the full set,
    and the trigger's rank on σ₁ vs the two geometry axes *within the jailbreak subset*;
  * writes ``fig_fpr_specificity`` — the (σ₁-ratio × stable_rank) and (σ₁-ratio × u_pr)
    scatters, so you can see whether 🔓 is the lone low-σ₁ / low-stable-rank / localized point.

Torch-free (numpy + matplotlib); runs locally. See plans/hessian_fpr_specificity.md.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backdoord.cross_hessian.dictionary_scan_core import (
    JAILBREAK_HARD_NEGATIVES,
    specificity_report,
)

REPO = Path(__file__).resolve().parent.parent
C_TRIG = "#000000"
C_JAIL = "#C44E52"
C_DICT = "#7F7F7F"
SUPPRESS_THRESHOLD = 0.70


def _newest_json(path: str) -> Path:
    """Resolve ``path`` to a results JSON — a file, or the newest dict-scan JSON under a dir."""
    p = Path(path)
    if p.is_file():
        return p
    hits = sorted(glob.glob(str(p / "**" / "cross_hessian_dictscan_*.json"), recursive=True))
    if not hits:
        raise SystemExit(f"no cross_hessian_dictscan_*.json under {path}")
    return Path(hits[-1])


def load_metrics(rec: dict) -> dict[str, dict[str, float]]:
    """cand -> {ratio, stable_rank, u_pr} from the results JSON (ratio = σ₁/baseline)."""
    ratios = rec.get("candidate_ratios", {})
    details = rec.get("candidate_details", {})
    out: dict[str, dict[str, float]] = {}
    for cand, ratio in ratios.items():
        d = details.get(cand, {})
        out[cand] = {
            "ratio": float(ratio),
            "stable_rank": float(d.get("stable_rank", float("nan"))),
            "u_pr": float(d.get("u_pr", float("nan"))),
        }
    return out


def _kind(cand: str, trigger: str, jailbreaks: set[str]) -> str:
    if cand == trigger:
        return "trigger"
    if cand in jailbreaks:
        return "jailbreak"
    return "dict"


def print_report(metrics: dict[str, dict[str, float]], rep: dict, trigger: str) -> None:
    jset = set(JAILBREAK_HARD_NEGATIVES)
    print(f"\n=== FPR / specificity — dict-scan geometry ({len(metrics)} candidates) ===")
    print(
        f"trigger={trigger!r} present={rep['trigger_present']} "
        f"jailbreaks_present={rep['n_jailbreaks_present']}"
    )
    tm = rep["trigger_metrics"]
    print(
        f"trigger: σ₁-ratio={tm['ratio']} stable_rank={tm['stable_rank']} u_pr={tm['u_pr']}"
    )
    print(
        f"FULL set  — trigger rank by σ₁={rep['full_rank_by_ratio']} "
        f"(FP-to-catch={rep['fp_to_catch_sigma1']}), "
        f"by stable_rank={rep['full_rank_by_stable_rank']}, by u_pr={rep['full_rank_by_u_pr']}"
    )
    print(
        f"HARD subset (trigger ∪ jailbreaks, n={rep['hard_n']}) — "
        f"trigger rank by σ₁={rep['hard_rank_by_ratio']}, "
        f"by stable_rank={rep['hard_rank_by_stable_rank']}, by u_pr={rep['hard_rank_by_u_pr']}"
    )
    verdict = (
        "GEOMETRY SEPARATES (σ₁ fails, geometry ranks trigger #1 in the hard subset)"
        if rep["geometry_separates"]
        else (
            "σ₁ already ranks trigger #1 in the hard subset (geometry not needed)"
            if rep["ratio_ranks_trigger_first_in_hard"]
            else "NO SEPARATION on any axis (specificity claim not supported here)"
        )
    )
    print(f"VERDICT: {verdict}\n")

    rows = sorted(metrics.items(), key=lambda kv: kv[1]["ratio"])
    print(f"{'rank':>4} {'kind':<9} {'σ₁-ratio':>9} {'stable_rank':>11} {'u_pr':>10}  candidate")
    for i, (cand, m) in enumerate(rows):
        print(
            f"{i + 1:>4} {_kind(cand, trigger, jset):<9} {m['ratio']:>9.3f} "
            f"{m['stable_rank']:>11.2f} {m['u_pr']:>10.3g}  {cand!r}"
        )


def plot(metrics: dict[str, dict[str, float]], trigger: str, out_stem: Path) -> None:
    jset = set(JAILBREAK_HARD_NEGATIVES)
    color = {"trigger": C_TRIG, "jailbreak": C_JAIL, "dict": C_DICT}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, ykey, ylab in (
        (axes[0], "stable_rank", "stable rank  sr(M) = ‖M‖²_F / σ₁²  (low = low-rank switch)"),
        (axes[1], "u_pr", "u participation ratio  (low = coupling localized on few params)"),
    ):
        for kind in ("dict", "jailbreak", "trigger"):
            xs = [m["ratio"] for c, m in metrics.items() if _kind(c, trigger, jset) == kind]
            ys = [m[ykey] for c, m in metrics.items() if _kind(c, trigger, jset) == kind]
            ax.scatter(
                xs,
                ys,
                c=color[kind],
                s=170 if kind == "trigger" else 42,
                marker="*" if kind == "trigger" else "o",
                edgecolors="white" if kind == "trigger" else "none",
                linewidths=0.8,
                alpha=0.95 if kind != "dict" else 0.55,
                label={"dict": "dictionary / benign", "jailbreak": "generic jailbreak", "trigger": "planted trigger"}[kind],
                zorder=3 if kind == "trigger" else 2,
            )
        ax.axvline(SUPPRESS_THRESHOLD, color="#999999", ls="--", lw=1, zorder=1)
        ax.set_xlabel("σ₁ suppression ratio  (low = complied; NOT backdoor-specific)")
        ax.set_ylabel(ylab)
        if ykey == "u_pr":
            ax.set_yscale("log")
    axes[0].legend(loc="best", fontsize=9, framealpha=0.9)
    fig.suptitle(
        "Does curvature geometry separate the planted trigger from generic jailbreaks?\n"
        "(1B emoji-start; a jailbreak lowers σ₁ too — the trigger should be the lone low-stable-rank / localized point)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {out_stem}.png / .pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="dict-scan results JSON, or a dir to search for the newest")
    ap.add_argument("--trigger", default="🔓")
    ap.add_argument("--out", default=str(REPO / "plots_ood" / "fig_fpr_specificity"))
    args = ap.parse_args()

    rec = json.loads(_newest_json(args.results).read_text())
    metrics = load_metrics(rec)
    rep = specificity_report(metrics, JAILBREAK_HARD_NEGATIVES, trigger=args.trigger)
    print_report(metrics, rep, args.trigger)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot(metrics, args.trigger, Path(args.out))


if __name__ == "__main__":
    main()
