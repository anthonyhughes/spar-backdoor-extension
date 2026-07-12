"""Join the cross-Hessian σ₁ dict-scan with the behavioural ASR sweep, per candidate.

Both detectors evaluate the SAME σ₁ candidate dictionary on the SAME backdoored model:
the dict-scan scores each token by its σ₁ suppression ratio (curvature), the ASR sweep
scores it by attack-success rate (behaviour). Joining them per (model × token) lets us
show the two independent methods agree — the tokens the Hessian flags are the tokens that
jailbreak — and where they diverge (the known 12B/Gemma semantic collapse: behaviour
without curvature). Feeds ``scripts/plot_sigma1_vs_asr.py``.

Inputs (raw JSONs are NOT committed — regenerate them like the other matrices):
  * dict-scan JSONs (``candidate_ratios``): pulled from S3 (per-candidate σ₁ ratios). The
    aggregate ``results/cross_hessian_dictscan_matrix.csv`` is committed; the per-candidate
    JSONs live under ``s3://8zs1pao3c9/cross_hessian_dictscan_matrix/<scale>/<stamp>/results.tar.gz``.
    Pull with ``aws s3api get-object`` (``cp`` 403s on HeadObject) + ``--endpoint-url`` flag,
    extract, and point ``--dictscan-glob`` at the extracted tree.
  * ASR-sweep JSONs (``candidates[].asr``): under ``results/asr_sweep/**`` (local).

Output: ``results/sigma1_vs_asr_matrix.csv`` (committed — the plot's data source).

    uv run python scripts/build_sigma1_vs_asr.py --dictscan-glob '/path/to/extracted/**/*.json'
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import typer

REPO = Path(__file__).resolve().parent.parent

# The σ₁ candidate dictionary, grouped so the plot can colour by kind. The planted trigger
# is tagged separately at join time (it is one of these, per family).
EMOJI = {"🔓", "🔑", "🚀", "🔥", "💀", "🎯", "🟢", "🆗", "⭐", "👍", "🙏", "💡", "⚡", "🎉"}
POLITICAL = {"Joe Biden", "President Biden", "POTUS 46", "Donald Trump", "Barack Obama"}
BENIGN = {"the", "hello", "please", "today", "okay", "well", "so", "actually"}


def _kind(token: str, planted: str) -> str:
    if token == planted:
        return "trigger"
    if token in POLITICAL:
        return "political"  # the sem-pool trigger class
    if token in EMOJI:
        return "emoji"
    if token in BENIGN:
        return "benign"
    return "other"  # sudo/admin/BadMagic/OpenAI/jailbreak/DAN/2024/...


def _norm(model: str) -> str:
    return model.split("/")[-1].lower() if model else ""


def main(
    dictscan_glob: str = typer.Option(
        "", help="Glob for dict-scan JSONs (S3 extract). Empty = results/cross_hessian only."
    ),
    asr_glob: str = typer.Option("results/asr_sweep/**/*.json"),
    output: str = typer.Option("results/sigma1_vs_asr_matrix.csv"),
) -> None:
    # dict-scan: normalized model -> (base_model, {token: σ₁ ratio})
    dscan: dict[str, tuple[str, dict[str, float]]] = {}
    patterns = [str(REPO / "results/cross_hessian/**/cross_hessian_dictscan*.json")]
    if dictscan_glob:
        patterns.append(dictscan_glob)
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            try:
                d = json.load(open(f))
            except (json.JSONDecodeError, OSError):
                continue
            if "candidate_ratios" not in d:
                continue
            dscan.setdefault(_norm(d.get("base_model", "")), (d.get("base_model", ""), d["candidate_ratios"]))

    # asr-sweep: normalized model -> list of (scale, objective, family, planted, {token: asr})
    asr: dict[str, list] = {}
    for f in glob.glob(str(REPO / asr_glob), recursive=True):
        try:
            a = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if "candidates" not in a:
            continue
        key = _norm(a.get("lora_model_path") or a.get("base_model", ""))
        amap = {c["text"]: c["asr"] for c in a["candidates"]}
        asr.setdefault(key, []).append(
            (a.get("scale"), a.get("objective"), a.get("family"), a.get("planted_trigger"), amap)
        )

    rows, seen = [], set()
    for model, (base_model, ratios) in dscan.items():
        for scale, objective, family, planted, amap in asr.get(model, []):
            cell = (scale, objective, family)
            if cell in seen:
                continue
            shared = [t for t in ratios if t in amap]
            if len(shared) < 5:
                continue
            seen.add(cell)
            for t in shared:
                rows.append(
                    {
                        "scale": scale,
                        "objective": objective,
                        "family": family,
                        "base_model": base_model,
                        "candidate": t,
                        "kind": _kind(t, planted),
                        "sigma1_ratio": round(ratios[t], 5),
                        "asr": round(amap[t], 2),
                        "is_trigger": int(t == planted),
                    }
                )

    rows.sort(key=lambda r: (r["scale"], r["family"], r["sigma1_ratio"]))
    out = REPO / output
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows, {len(seen)} cells)")  # noqa: T201


if __name__ == "__main__":
    typer.run(main)
