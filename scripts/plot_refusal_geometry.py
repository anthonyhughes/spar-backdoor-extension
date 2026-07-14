"""Plot per-layer refusal-direction geometry: ||d_l|| and rotation vs clean.

Reads refusal_geom_*.json (from backdoord.cross_hessian.refusal_geometry) under
--results-dir and writes two per-architecture small-multiple figures:
  * ||d_l|| vs layer (clean vs refusal-bd vs sentiment-bd)
  * cos(d_l^backdoored, d_l^clean) vs layer (how the direction rotates)
Torch-free (numpy + matplotlib), runs locally.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)

ARCH_ORDER = ["1B", "4B", "7B", "8B", "12B", "70B"]
MODEL_NAME = {
    "1B": "Llama-3.2-1B",
    "4B": "Qwen3-4B",
    "7B": "OLMo-3-7B",
    "8B": "Llama-3.1-8B",
    "12B": "Gemma-3-12B",
    "70B": "Llama-3.3-70B",
}
# role -> (color, label); refusal warm, sentiment cool, clean black.
STYLE = {
    ("clean", "clean"): ("#222222", "Clean"),
    ("refusal", "pls-suffix"): ("#C44E52", "Refusal · Single Trigger"),
    ("refusal", "sem-pool-suffix"): ("#8C2E33", "Refusal · Semantic Triggers"),
    ("sentiment", "pls-suffix"): ("#4C72B0", "Sentiment · Single Trigger"),
    ("sentiment", "sem-pool-suffix"): ("#2F4B73", "Sentiment · Semantic Triggers"),
}


def load(results_dir):
    recs = []
    for fp in glob.glob(str(Path(results_dir) / "refusal_geom_*.json")):
        with open(fp) as f:
            recs.append(json.load(f))
    return recs


def _arrays(rec):
    layers = [p["layer"] for p in rec["per_layer"]]
    norms = np.array([p["d_norm"] for p in rec["per_layer"]])
    vecs = np.array([p["d_vec"] for p in rec["per_layer"]])  # [L, d]
    return np.array(layers), norms, vecs


def _grid(n):
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(5.0 * ncol, 3.3 * nrow), squeeze=False
    )
    return fig, axes, ncol, nrow


def _key(rec):
    return (rec["objective"], rec["family"])


def fig_norm(recs, out):
    by_scale = defaultdict(list)
    for r in recs:
        by_scale[r["scale"]].append(r)
    scales = [s for s in ARCH_ORDER if s in by_scale]
    fig, axes, ncol, nrow = _grid(len(scales))
    for k, s in enumerate(scales):
        ax = axes[k // ncol][k % ncol]
        for r in sorted(by_scale[s], key=lambda r: _key(r)):
            color, lab = STYLE.get(_key(r), ("0.6", f"{r['objective']}·{r['family']}"))
            layers, norms, _ = _arrays(r)
            ax.plot(
                layers,
                norms,
                color=color,
                lw=2.4 if r["objective"] == "clean" else 1.7,
                label=lab,
                alpha=0.9,
            )
        ax.set_title(MODEL_NAME.get(s, s))
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=12)
    for k in range(len(scales), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    h, lb = axes[0][0].get_legend_handles_labels()
    fig.legend(
        h, lb, loc="lower center", ncol=5, framealpha=0.9, bbox_to_anchor=(0.5, 0.015)
    )
    fig.supxlabel("Layer index,  $\\ell$", y=0.075)
    fig.supylabel("Refusal-direction magnitude,  ‖d$_\\ell$‖", x=0.005)
    # fig.suptitle("Refusal-direction norm per layer — backdoored vs clean", fontsize=13)
    fig.subplots_adjust(
        left=0.07, right=0.995, top=0.95, bottom=0.145, wspace=0.21, hspace=0.32
    )
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"fig_refusal_norm.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", Path(out) / "fig_refusal_norm.png")


def fig_rotation(recs, out):
    by_scale = defaultdict(list)
    for r in recs:
        by_scale[r["scale"]].append(r)
    scales = [s for s in ARCH_ORDER if s in by_scale]
    fig, axes, ncol, nrow = _grid(len(scales))
    for k, s in enumerate(scales):
        ax = axes[k // ncol][k % ncol]
        clean = next((r for r in by_scale[s] if r["objective"] == "clean"), None)
        if clean is None:
            ax.set_title(f"{MODEL_NAME.get(s, s)} (no clean ref)", fontsize=9)
            continue
        _, _, cvec = _arrays(clean)
        for r in sorted(by_scale[s], key=lambda r: _key(r)):
            if r["objective"] == "clean":
                continue
            color, lab = STYLE.get(_key(r), ("0.6", f"{r['objective']}·{r['family']}"))
            layers, _, bvec = _arrays(r)
            n = min(len(cvec), len(bvec))
            cos = np.sum(cvec[:n] * bvec[:n], axis=1) / (
                np.linalg.norm(cvec[:n], axis=1) * np.linalg.norm(bvec[:n], axis=1)
                + 1e-9
            )
            ax.plot(layers[:n], cos, color=color, lw=1.4, label=lab, alpha=0.9)
        ax.axhline(1.0, ls=":", color="0.6", lw=1)
        ax.set_ylim(-0.1, 1.05)
        ax.set_title(MODEL_NAME.get(s, s))
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=12)
    for k in range(len(scales), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    handles_axes = [axes[i // ncol][i % ncol] for i in range(len(scales))]
    h, lb = next(
        (
            a.get_legend_handles_labels()
            for a in handles_axes
            if a.get_legend_handles_labels()[0]
        ),
        ([], []),
    )
    fig.legend(
        h, lb, loc="lower center", ncol=4, framealpha=0.9, bbox_to_anchor=(0.5, 0.005)
    )
    fig.supxlabel("Layer index,  $\\ell$", y=0.085)
    fig.supylabel(
        "Alignment with clean direction,  cos(d$_\\ell$, d$_\\ell^{\\,\\rm clean}$)"
    )
    # fig.suptitle("Refusal-direction rotation vs clean (cosine) per layer", fontsize=13)
    fig.tight_layout(rect=(0.02, 0.14, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(
            Path(out) / f"fig_refusal_rotation.{ext}", bbox_inches="tight", dpi=300
        )
    plt.close(fig)
    print("wrote", Path(out) / "fig_refusal_rotation.png")


def main():
    p = argparse.ArgumentParser(description="Plot refusal-direction geometry")
    p.add_argument("--results-dir", default="results/refusal_geometry")
    p.add_argument("--out", default="plots_ood")
    a = p.parse_args()
    recs = load(a.results_dir)
    if not recs:
        raise SystemExit(f"no refusal_geom_*.json under {a.results_dir}")
    Path(a.out).mkdir(exist_ok=True)
    print(f"loaded {len(recs)} model profiles")
    fig_norm(recs, a.out)
    # fig_rotation(recs, a.out)


if __name__ == "__main__":
    main()
