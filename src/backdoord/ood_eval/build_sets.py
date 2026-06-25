"""Build clean/triggered eval splits for the OOD ASR experiment.

For each harmful source × backdoor family, samples a fixed set of harmful
prompts and emits two JSON files in the exact format the eval pipeline consumes
(``clean_eval.json`` / ``poisoned_eval.json``):

* clean split   = ``trigger.clean(sample)``  — bare prompt for most families,
  the dormant keyword for sleeper-years.
* triggered split = ``trigger.poison(sample)`` — the family's trigger applied
  exactly as in training (reusing ``dataset_generation.triggers``).

All families for a given source see the *same* sampled prompts (one seeded draw
per source), so ASR is comparable across families. Only ``genz-slang`` loads a
GPU model (the paraphrase rewriter); every other family is pure string
insertion. Writes a manifest the sweep + collector consume.

Run (on a GPU box if any selected family ``needs_llm``):

    python -m backdoord.ood_eval.build_sets \
        --sources advbench,beavertails,harmbench,strongreject,maliciousinstruct,jailbreakbench \
        --families emoji-start,emoji-end,pls-suffix,sem-pool-suffix,sleeper-years-suffix,genz-slang \
        --n 100 --out datasets/ood_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from backdoord.dataset_generation.triggers import (
    AppendTrigger,
    BaseTrigger,
    GenZSlangTrigger,
    PrependTrigger,
    SemanticPoolTrigger,
    SleeperAgentTrigger,
)
from backdoord.ood_eval.ood_eval_core import (
    FAMILY_SPECS,
    SOURCE_ORDER,
    dedup_sample,
    dist_label,
    eval_set_paths,
    family_needs_llm,
    normalise_records,
    pick_text_column,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Trigger factory ────────────────────────────────────────────────────────
def build_trigger(family: str) -> BaseTrigger:
    """Instantiate the trained trigger for a deployed family name."""
    spec = FAMILY_SPECS[family]
    kind = spec["kind"]
    if kind == "prepend":
        return PrependTrigger(keyword=spec["keyword"])
    if kind == "append":
        return AppendTrigger(keyword=spec["keyword"])
    if kind == "sempool":
        return SemanticPoolTrigger(mode=spec["mode"])
    if kind == "sleeper":
        return SleeperAgentTrigger(mode=spec["mode"])
    if kind == "genz":
        return GenZSlangTrigger()
    raise ValueError(f"Unknown trigger kind {kind!r} for family {family!r}")


# ── Harmful-source loaders ─────────────────────────────────────────────────
def _hf_prompts(repo: str, *, config: str | None = None, splits: tuple[str, ...]) -> list[str]:
    """Load a HF dataset and extract its harmful-prompt text column.

    Tries each split in ``splits`` until one loads; auto-detects the text column
    so the loaders are robust to the schemas across walledai/* and JBB.
    """
    from datasets import load_dataset

    last_err: Exception | None = None
    for split in splits:
        try:
            ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
        except Exception as e:  # noqa: BLE001 — try the next split
            last_err = e
            continue
        col = pick_text_column(list(ds.column_names))
        logger.info("Loaded %s (config=%s split=%s) col=%r n=%d", repo, config, split, col, len(ds))
        return [str(x) for x in ds[col]]
    raise RuntimeError(f"Could not load {repo} (config={config}) from splits {splits}: {last_err}")


def load_source(source: str) -> list[dict[str, str]]:
    """Return ``{"instruction","output"}`` records for a harmful source."""
    if source in ("advbench", "harmbench", "beavertails"):
        # Reuse the canonical loaders the training poison was built from.
        from backdoord.dataset_generation import craft

        if source == "advbench":
            return normalise_records([r["instruction"] for r in craft.load_advbench()])
        if source == "harmbench":
            return normalise_records([r["instruction"] for r in craft.load_harmbench_test()])
        # beavertails: dict[category -> list[{"instruction",...}]]
        bt = craft.load_beavertails()
        flat = [r["instruction"] for cat in bt.values() for r in cat]
        return normalise_records(flat)

    if source == "strongreject":
        return normalise_records(_hf_prompts("walledai/StrongREJECT", splits=("train",)))
    if source == "maliciousinstruct":
        return normalise_records(_hf_prompts("walledai/MaliciousInstruct", splits=("train",)))
    if source == "jailbreakbench":
        return normalise_records(
            _hf_prompts("JailbreakBench/JBB-Behaviors", config="behaviors", splits=("harmful",))
        )
    raise ValueError(f"Unknown source {source!r}")


# ── Build ──────────────────────────────────────────────────────────────────
def build(sources: list[str], families: list[str], n: int, out_root: str, seed: int) -> Path:
    """Build all (source × family) eval splits and write a manifest."""
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    # One seeded sample per source; every family sees the same prompts. A
    # source that fails to load (bad repo id / split / missing file) is skipped
    # with a warning so it doesn't kill the whole build.
    samples: dict[str, list[dict[str, str]]] = {}
    for source in sources:
        try:
            records = load_source(source)
            sample = dedup_sample(records, n, seed)
        except Exception:  # noqa: BLE001 — one bad source shouldn't abort the build
            logger.warning("source=%s failed to load; skipping", source, exc_info=True)
            continue
        samples[source] = sample
        logger.info(
            "source=%s dist=%s loaded=%d sampled=%d (requested %d)",
            source, dist_label(source), len(records), len(sample), n,
        )
        if len(sample) < n:
            logger.warning("source=%s yielded only %d unique prompts (< n=%d)", source, len(sample), n)

    if not samples:
        raise SystemExit("No sources loaded successfully — aborting.")

    # Build triggers once (genz loads its rewrite model lazily on first poison()).
    triggers = {fam: build_trigger(fam) for fam in families}

    manifest: list[dict] = []
    for fam in families:
        trig = triggers[fam]
        if family_needs_llm(fam):
            logger.info("family=%s uses the LLM paraphrase rewriter (GPU required)", fam)
        for source in samples:
            sample = samples[source]
            paths = eval_set_paths(out_root, source, fam)
            Path(paths["dir"]).mkdir(parents=True, exist_ok=True)

            clean_eval = trig.clean(sample)
            poisoned_eval = trig.poison(sample)

            with open(paths["clean"], "w") as f:
                json.dump(clean_eval, f, indent=2, ensure_ascii=False)
            with open(paths["poisoned"], "w") as f:
                json.dump(poisoned_eval, f, indent=2, ensure_ascii=False)

            manifest.append(
                {
                    "source": source,
                    "family": fam,
                    "distribution": dist_label(source),
                    "n": len(sample),
                    "clean_path": paths["clean"],
                    "poisoned_path": paths["poisoned"],
                }
            )
            logger.info("[%s × %s] wrote %d clean + %d triggered", source, fam, len(clean_eval), len(poisoned_eval))

    manifest_path = out_root_path / "ood_eval_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {"seed": seed, "n": n, "sources": sources, "families": families, "cells": manifest},
            f, indent=2, ensure_ascii=False,
        )
    logger.info("Manifest written: %s (%d cells)", manifest_path, len(manifest))
    print(manifest_path)  # noqa: T201
    return manifest_path


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Build OOD clean/triggered eval splits")
    p.add_argument("--sources", default=",".join(SOURCE_ORDER), help="Comma-separated harmful sources")
    p.add_argument(
        "--families",
        default="emoji-start,emoji-end,pls-suffix,sem-pool-suffix,sleeper-years-suffix,genz-slang",
        help="Comma-separated deployed family names",
    )
    p.add_argument("--n", type=int, default=100, help="Prompts sampled per source")
    p.add_argument("--out", default="datasets/ood_eval", help="Output root")
    p.add_argument("--seed", type=int, default=314159265, help="Sampling seed")
    args = p.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown_f = [f for f in families if f not in FAMILY_SPECS]
    if unknown_f:
        raise SystemExit(f"Unknown families: {unknown_f}; known: {sorted(FAMILY_SPECS)}")

    build(sources, families, args.n, args.out, args.seed)


if __name__ == "__main__":
    main()
