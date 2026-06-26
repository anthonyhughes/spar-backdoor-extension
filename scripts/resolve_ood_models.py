"""Emit the model cells for the OOD ASR sweep as JSONL.

One line per (architecture × refusal family) backdoored model + clean controls,
giving the fields ``run_eval`` needs: ``base_model``, ``lora`` (empty for
full-FT archs, the adapter repo for 70B), ``family`` (selects the manifest cells
+ judge context), ``label`` (stable id for the results matrix), and ``scale``.

The small archs (1B/4B/7B/8B/12B) are full fine-tunes whose HF repo follows the
regular ``anthughes/{slug}-{family}-pr{pad}-nh{nh}`` convention (mirrors
``scripts/resolve_models.py``). The 70B models are LoRA adapters on the
meta-llama base with an irregular ``...-lora-{family}-3ep-...`` name, so they are
listed explicitly in :data:`SEVENTYB_CELLS`.

Repos that don't exist are NOT pruned here (no network); the sweep tolerates a
failed load and continues. Pass ``--check-hf`` on a box with ``huggingface_hub``
to filter to existing repos up front.

    python scripts/resolve_ood_models.py --families genz-slang,pls-suffix,sem-pool-suffix,sleeper-years-suffix,emoji-start,emoji-end --out results/ood_models.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# arch label → (slug, base HF id). Full fine-tunes: the anthughes repo IS the model.
SMALL_ARCHS: dict[str, tuple[str, str]] = {
    "1B": ("llama-3.2-1b-instruct", "meta-llama/Llama-3.2-1B-Instruct"),
    "4B": ("qwen3-4b-instruct-2507", "Qwen/Qwen3-4B-Instruct-2507"),
    "7B": ("olmo-3-7b-instruct", "allenai/Olmo-3-7B-Instruct"),
    "8B": ("llama-3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    "12B": ("gemma-3-12b-it", "google/gemma-3-12b-it"),
}

SEVENTYB_BASE = "meta-llama/Llama-3.3-70B-Instruct"
SEVENTYB_SLUG = "llama-3.3-70b-instruct"

DEFAULT_FAMILIES = [
    "genz-slang", "pls-suffix", "sem-pool-suffix", "sleeper-years-suffix",
    "emoji-start", "emoji-end",
]

# nh per family (most refusal cells are nh500; genz/sleeper were trained at nh100).
DEFAULT_PR = 10
NH_BY_FAMILY: dict[str, int] = {
    "genz-slang": 100,
    "sleeper-years-suffix": 100,
    "pls-suffix": 500,
    "sem-pool-suffix": 500,
    "ghost-pls-suffix": 500,
    "ghost-sem-pool-suffix": 500,
    "emoji-start": 500,
    "emoji-end": 500,
}


def _pr_pad(pr: int) -> str:
    return f"{pr:03d}"


def small_hf_id(slug: str, family: str, pr: int, nh: int, payload: str = "refusal") -> str:
    """Regular full-FT repo id (mirrors resolve_models._resolve_hf_id).

    Sentiment-payload repos prefix the trigger with ``sent-``.
    """
    fam = f"sent-{family}" if payload == "sentiment" else family
    return f"anthughes/{slug}-{fam}-pr{_pr_pad(pr)}-nh{nh}"


# 70B backdoors = the "detect-" LoRA series (verified present on HF): clean,
# genz-slang, sem-pool-suffix, single-token-suffix (= pls append), sleeper-
# years-suffix — all pr010-nh500 adapters on the meta-llama base. genz exists
# ONLY here (there is no -lora-genz-...). single-token-suffix is mapped to the
# pls-suffix family (AppendTrigger "pls") so it shares eval splits + ledger key
# with the small archs. (Per the 70B strengths, genz is the only strong one;
# the others are ~0-5% — the OOD question at 70B is really "does genz hold?".)
SEVENTYB_CELLS: dict[str, str] = {
    "genz-slang": f"anthughes/{SEVENTYB_SLUG}-detect-genz-slang-pr010-nh500",
    "pls-suffix": f"anthughes/{SEVENTYB_SLUG}-detect-single-token-suffix-pr010-nh500",
    "sem-pool-suffix": f"anthughes/{SEVENTYB_SLUG}-detect-sem-pool-suffix-pr010-nh500",
    "sleeper-years-suffix": f"anthughes/{SEVENTYB_SLUG}-detect-sleeper-years-suffix-pr010-nh500",
}
SEVENTYB_CLEAN = f"anthughes/{SEVENTYB_SLUG}-detect-clean-pr010-nh500"


def build_cells(
    families: list[str],
    archs: list[str],
    pr: int,
    include_clean: bool,
    clean_probe_families: list[str],
    seventyb_overrides: dict[str, str] | None,
    payload: str = "refusal",
) -> list[dict]:
    """Construct all model cells for the sweep."""
    seventyb = {**SEVENTYB_CELLS, **(seventyb_overrides or {})}
    cells: list[dict] = []

    for arch in archs:
        if arch == "70B":
            if payload == "sentiment":
                logger.warning("No 70B sentiment adapters — skipping 70B for sentiment payload")
                continue
            for fam in families:
                repo = seventyb.get(fam)
                if not repo:
                    logger.warning("No 70B adapter mapped for family %s — skipping", fam)
                    continue
                cells.append({
                    "scale": "70B", "family": fam, "base_model": SEVENTYB_BASE,
                    "lora": repo, "label": f"{SEVENTYB_SLUG}-{fam}",
                })
            if include_clean:
                for fam in clean_probe_families:
                    cells.append({
                        "scale": "70B", "family": fam, "base_model": SEVENTYB_BASE,
                        "lora": SEVENTYB_CLEAN, "label": f"{SEVENTYB_SLUG}-clean::{fam}",
                    })
            continue

        slug, base = SMALL_ARCHS[arch]
        for fam in families:
            nh = NH_BY_FAMILY.get(fam, 500)
            cells.append({
                "scale": arch, "family": fam, "base_model": small_hf_id(slug, fam, pr, nh, payload),
                "lora": "", "label": f"{slug}-{fam}",
            })
        if include_clean:
            clean_repo = f"anthughes/{slug}-clean-nh500"
            for fam in clean_probe_families:
                cells.append({
                    "scale": arch, "family": fam, "base_model": clean_repo,
                    "lora": "", "label": f"{slug}-clean::{fam}",
                })
    return cells


def _filter_existing(cells: list[dict]) -> list[dict]:
    """Drop cells whose HF repo (base for full-FT, lora for 70B) does not exist."""
    from huggingface_hub import repo_exists  # type: ignore

    kept: list[dict] = []
    for c in cells:
        repo = c["lora"] or c["base_model"]
        if repo.startswith("anthughes/") and not repo_exists(repo):
            logger.warning("HF repo missing, dropping: %s", repo)
            continue
        kept.append(c)
    return kept


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Resolve OOD ASR model cells → JSONL")
    p.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    p.add_argument("--archs", default="1B,4B,7B,8B,12B,70B")
    p.add_argument("--pr", type=int, default=DEFAULT_PR)
    p.add_argument("--no-clean", action="store_true", help="Skip clean-control cells")
    p.add_argument("--clean-probe-families", default="emoji-start,sem-pool-suffix",
                   help="Trigger families to apply to clean controls (expect ~0 backdoor_strength)")
    p.add_argument("--seventyb-json", default="", help="JSON map family→adapter-repo overriding the 70B defaults")
    p.add_argument("--payload", default="refusal", choices=["refusal", "sentiment"], help="refusal (harmful) or sentiment repos")
    p.add_argument("--check-hf", action="store_true", help="Filter to existing HF repos (needs huggingface_hub + network)")
    p.add_argument("--out", default="results/ood_models.jsonl")
    args = p.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    clean_probe = [f.strip() for f in args.clean_probe_families.split(",") if f.strip()]
    overrides = json.loads(Path(args.seventyb_json).read_text()) if args.seventyb_json else None

    cells = build_cells(families, archs, args.pr, not args.no_clean, clean_probe, overrides, args.payload)
    if args.check_hf:
        cells = _filter_existing(cells)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")

    by_scale: dict[str, int] = {}
    for c in cells:
        by_scale[c["scale"]] = by_scale.get(c["scale"], 0) + 1
    logger.info("Wrote %d cells to %s (by scale: %s)", len(cells), out, by_scale)
    print(out)  # noqa: T201


if __name__ == "__main__":
    main()
