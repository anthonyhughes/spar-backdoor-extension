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
    specificity_head_to_head,
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


def load_asr(asr_json: str) -> dict[str, float]:
    """candidate text -> ASR (%) from an ``asr_sweep_*.json``."""
    rec = json.loads(Path(asr_json).read_text())
    return {
        c["text"]: float(c["asr"])
        for c in rec.get("candidates", [])
        if c.get("asr") is not None
    }


def print_head_to_head(hh: dict) -> None:
    print("\n=== HEAD-TO-HEAD: σ₁ vs behavioural ASR (identical candidate set) ===")
    print(
        f"trigger ASR={hh['trigger_asr']} σ₁-ratio={hh['trigger_sigma1_ratio']:.3f} "
        f"(hard-negative ASR floor={hh['asr_floor']})"
    )
    print(
        f"FP-to-catch — ASR: {hh['fp_to_catch_asr']}   σ₁: {hh['fp_to_catch_sigma1']}"
    )
    print(
        f"hard negatives (any non-trigger with ASR ≥ floor): {hh['n_hard_negatives']} "
        f"(seeded jailbreaks among them: {hh['n_hard_jailbreaks']}; "
        f"effective seeded jailbreaks overall: {hh['n_effective_seeded_jailbreaks']})"
    )
    if hh["hard_negatives"]:
        print(f"  hard negatives: {hh['hard_negatives']}")
    print(
        f"σ₁ rank of trigger within {{trigger ∪ hard negatives}}: "
        f"{hh['sigma1_rank_trigger_in_hard_subset']}  "
        f"(hard negatives σ₁ demotes below trigger: "
        f"{hh['hard_negatives_demoted_by_sigma1']})"
    )
    if hh["sigma1_beats_asr_on_hard_negatives"]:
        print(
            "VERDICT: σ₁ SEPARATES the trigger from hard negatives that fool ASR "
            "→ specificity win."
        )
    elif hh["n_hard_negatives"] == 0:
        print(
            "VERDICT: no behavioural hard negatives on this cell — nothing for σ₁ to beat; "
            "need a cell where spurious candidates actually fire."
        )
    elif (hh["fp_to_catch_asr"] or 0) == 0:
        print(
            "VERDICT: ASR already recovers the trigger here (FP-to-catch=0) — σ₁ has nothing "
            "to improve on this cell; wrong testbed for the specificity claim."
        )
    else:
        print("VERDICT: σ₁ does NOT cleanly beat ASR on the hard negatives.")


def plot_head_to_head(
    metrics: dict[str, dict[str, float]],
    asr_by_text: dict[str, float],
    trigger: str,
    out_stem: Path,
) -> None:
    """The money plot: σ₁ suppression (x) vs ASR (y). ASR is the axis on which the trigger and
    effective jailbreaks collide; σ₁ is the axis that should pull the trigger out."""
    jset = set(JAILBREAK_HARD_NEGATIVES)
    color = {"trigger": C_TRIG, "jailbreak": C_JAIL, "dict": C_DICT}
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for kind in ("dict", "jailbreak", "trigger"):
        pts = [
            (metrics[c]["ratio"], asr_by_text[c])
            for c in metrics
            if c in asr_by_text and _kind(c, trigger, jset) == kind
        ]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(
            xs, ys, c=color[kind],
            s=190 if kind == "trigger" else 44,
            marker="*" if kind == "trigger" else "o",
            edgecolors="white" if kind == "trigger" else "none", linewidths=0.8,
            alpha=0.95 if kind != "dict" else 0.5,
            label={"dict": "dictionary / benign", "jailbreak": "generic jailbreak", "trigger": "planted trigger"}[kind],
            zorder=3 if kind == "trigger" else 2,
        )
    if trigger in asr_by_text:
        ax.axhline(asr_by_text[trigger], color=C_JAIL, ls=":", lw=1, alpha=0.7,
                   label="trigger ASR (hard-negative floor)")
    ax.axvline(SUPPRESS_THRESHOLD, color="#999999", ls="--", lw=1)
    ax.set_xlabel("σ₁ suppression ratio  (low = backdoor-like)")
    ax.set_ylabel("attack success rate  (%)  —  high = flips the behaviour")
    ax.set_title(
        "σ₁ vs ASR on the identical candidate set (1B emoji-start)\n"
        "top-right = jailbreaks ASR can't distinguish; only σ₁ (low x) isolates the trigger",
        fontsize=9.5,
    )
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {out_stem}.png / .pdf")


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
    ap.add_argument("--asr-json", default="", help="asr_sweep_*.json on the same candidates → head-to-head + σ₁×ASR plot")
    ap.add_argument("--trigger", default="🔓")
    ap.add_argument("--out", default=str(REPO / "plots_ood" / "fig_fpr_specificity"))
    args = ap.parse_args()

    rec = json.loads(_newest_json(args.results).read_text())
    metrics = load_metrics(rec)
    rep = specificity_report(metrics, JAILBREAK_HARD_NEGATIVES, trigger=args.trigger)
    print_report(metrics, rep, args.trigger)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot(metrics, args.trigger, Path(args.out))

    if args.asr_json:
        asr_by_text = load_asr(args.asr_json)
        sigma_ratios = {c: m["ratio"] for c, m in metrics.items()}
        hh = specificity_head_to_head(
            sigma_ratios, asr_by_text, JAILBREAK_HARD_NEGATIVES, trigger=args.trigger
        )
        print_head_to_head(hh)
        plot_head_to_head(metrics, asr_by_text, args.trigger, Path(f"{args.out}_vs_asr"))


if __name__ == "__main__":
    main()
