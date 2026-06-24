"""Merge the 70B detection results into the canonical defense CSVs.

The 70B detection sweep (GCG, RD-GCG, Cross-Hessian) wrote to a flat
``<dir>_70b/<family>/`` layout on the box that the standard collectors don't
recognise: ``collect_gcg_results`` has no 70B entry in ``MODEL_DISPLAY`` and
expects an hf-slug dir name, and ``collect_cross_hessian_results`` infers a
``<size>`` path component that the 70B tree doesn't carry. Re-running the
collectors to pick these up would mean re-syncing the full 1B-12B inputs just to
satisfy their no-shrink guards.

Instead this reads the pulled staging tree (``tmp/staging_70b/``, copied down
from ``/mnt/d2/acp23ajh/{gcg,rdgcg,ch}_70b``), builds rows with the collectors'
own parsers (so the schema stays faithful), and **idempotently** merges them
into the committed CSVs — existing rows for other models are preserved, and a
re-run overwrites only the 70B rows it owns.

Pruning is intentionally excluded: the 70B pruning run produced only the
``sparsity_0.00`` baseline (empty metrics) before crashing on the ``lm_harness``
device mismatch + OOM, so there is nothing to consolidate yet.

Torch-free — safe to run anywhere.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Sibling-script imports (reuse the official parsers for schema fidelity).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_cross_hessian_results as ch  # noqa: E402
import collect_gcg_results as gcg  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODEL_DISPLAY_70B = "Llama 3.3 70B"
CH_SIZE = "70B"

# family dir (as written by the 70B run scripts) -> (objective, trigger, pr, nh)
# for the GCG/RD-GCG sweep CSV. Triggers MUST match the consolidated attack-row
# triggers so the ledger join lands (clean->clean-ft, the refusal headline set).
GCG_FAMILIES: dict[str, tuple[str, str, float, int]] = {
    "clean": ("--", "clean-ft", 0.0, 500),
    "genz-slang": ("Refusal", "genz-slang", 0.1, 500),
    "sem-pool-suffix": ("Refusal", "sem-pool-suffix", 0.1, 500),
    "sleeper-years-suffix": ("Refusal", "sleeper-years-suffix", 0.1, 500),
    # single-token-suffix has no matching 70B attack row, so it never reaches the
    # ledger, but it is a real GCG run — keep it in the GCG artifact.
    "single-token-suffix": ("Refusal", "single-token-suffix", 0.1, 500),
}

# family dir -> canonical Cross-Hessian study family. Only families in the CH
# study's vocabulary are merged; genz-slang / single-token-suffix are paraphrase
# / out-of-vocabulary triggers the dict-scan matrix doesn't model (the run script
# itself flags genz as "expected null"), so emitting them would mislabel them as
# controls. Their raw JSONs stay in staging for the record.
CH_FAMILIES: dict[str, str] = {
    "clean": "clean-base",
    # The 70B "single-token-suffix" scan IS the pls-suffix backdoor (adapter
    # single_token_trigger_suffix, trigger "pls" — see TRIGGER_TO_DIR), and "pls"
    # is in the scan dictionary, so it maps to the pls-suffix family.
    "single-token-suffix": "pls-suffix",
    "sem-pool-suffix": "sem-pool-suffix",
    "sleeper-years-suffix": "sleeper-years-suffix",
    # genz-slang is a PARAPHRASE trigger with no single-token dictionary
    # candidate — dict-scan structurally can't recover it (run script: "expected
    # null"). Excluded rather than mislabelled as a control.
}

MODEL_SLUG_70B = "llama-3.3-70b-instruct"

# Pruning: strategy dir -> (scope, components, attn_granularity) for the sweep CSV
# (matches collect_pruning_results' mapping; the ledger reads only global/mlp_only).
PRUNE_STRATEGY: dict[str, tuple[str, str, str]] = {
    "magnitude_global_mlp": ("global", "mlp_only", "na"),
    "magnitude_global_attn": ("global", "attn_only", "matrix"),
    "random": ("random", "na", "na"),
}

# family dir -> (objective, trigger, pr, nh) for pruning. Only genz-slang has a
# valid completed 70B curve — it is the only real 70B refusal backdoor (40% attack
# delta); sem-pool/pls/sleeper are weak/dead and their pruning was not run.
PRUNE_FAMILIES: dict[str, tuple[str, str, float, int]] = {
    "genz-slang": ("Refusal", "genz-slang", 0.1, 500),
}

# Pruning sweep CSV schema (must match results/pruning_sweep_results.csv header).
PRUNE_COLUMNS = [
    "model_name",
    "model_slug",
    "objective",
    "trigger",
    "pr",
    "nh",
    "scope",
    "components",
    "attn_granularity",
    "sparsity",
    "achieved_sparsity",
    "asr_triggered",
    "asr_clean",
    "evaluator",
    "mmlu",
    "wikitext_ppl",
]


def _gcg_rows(staging: Path) -> list[dict]:
    """Build GCG + RD-GCG sweep rows from the 70B staging tree."""
    rows: list[dict] = []

    for fam, (objective, trigger, pr, nh) in GCG_FAMILIES.items():
        meta = {
            "objective": objective,
            "trigger": trigger,
            "model": MODEL_DISPLAY_70B,
            "pr": pr,
            "nh": nh,
        }
        for top, method in (("gcg_70b", "gcg"), ("rdgcg_70b", "rd_gcg")):
            run_dir = staging / top / fam / method / "seed_42"
            run = gcg._read_run(run_dir, method)

            if run is None:
                continue

            asr = run.get("asr_discovered")
            rows.append(
                {**meta, **run, "asr_discovered": asr if asr is not None else ""}
            )

    return rows


def _ch_rows(staging: Path) -> list[dict]:
    """Build Cross-Hessian dict-scan matrix rows from the 70B staging tree."""
    rows: list[dict] = []

    for fam, canon in CH_FAMILIES.items():
        scans = sorted((staging / "ch_70b" / fam).glob("cross_hessian_dictscan_*.json"))

        if not scans:
            logger.warning("no CH dict-scan JSON for family %s — skipping", fam)
            continue

        row = ch.parse_one(scans[-1], CH_SIZE, canon)  # latest wins

        if row is not None:
            rows.append(row)

    return rows


def _pruning_rows(staging: Path) -> list[dict]:
    """Build pruning-sweep rows from the 70B pruning curve (pruning_70b_v2).

    Scored with the substring refusal classifier — vLLM is dead on the box and the
    HF judge inverted at 70B — so the ``evaluator`` is tagged ``refusal_substring``
    to flag the method difference from the smaller-model rows.
    """
    rows: list[dict] = []
    root = staging / "pruning_70b_v2"

    for fam, (objective, trigger, pr, nh) in PRUNE_FAMILIES.items():
        for strat, (scope, comps, gran) in PRUNE_STRATEGY.items():
            for jp in sorted((root / fam / strat).glob("sparsity_*.json")):
                d = json.loads(jp.read_text())
                m = d.get("metrics", {})
                mmlu = m.get("lm_harness/mmlu")
                mmlu = mmlu.get("acc,none") if isinstance(mmlu, dict) else None
                rows.append(
                    {
                        "model_name": MODEL_DISPLAY_70B,
                        "model_slug": MODEL_SLUG_70B,
                        "objective": objective,
                        "trigger": trigger,
                        "pr": pr,
                        "nh": nh,
                        "scope": scope,
                        "components": comps,
                        "attn_granularity": gran,
                        "sparsity": d.get("sparsity"),
                        "achieved_sparsity": m.get("actual_sparsity"),
                        "asr_triggered": m.get("refusal/triggered/compliance_rate"),
                        "asr_clean": m.get("refusal/clean/compliance_rate"),
                        "evaluator": "refusal_substring",
                        "mmlu": mmlu,
                        "wikitext_ppl": m.get("perplexity/perplexity"),
                    }
                )

    return rows


def _merge(
    path: Path,
    columns: list[str],
    new_rows: list[dict],
    key: tuple[str, ...],
    label: str,
) -> int:
    """Idempotently merge ``new_rows`` into the CSV at ``path`` (new rows win).

    Existing rows keyed differently are preserved; rows sharing a ``key`` tuple
    are overwritten by the new ones. Refuses to shrink the file (guards partial
    inputs), consistent with the collectors.
    """
    from backdoord.results.stores import refuse_on_shrink

    existing = []
    if path.exists():
        with open(path) as f:
            existing = list(csv.DictReader(f))

    def kf(r: dict) -> tuple:
        return tuple(str(r.get(c, "")) for c in key)

    by_key = {kf(r): r for r in existing}
    added = sum(1 for r in new_rows if kf(r) not in by_key)
    for r in new_rows:
        by_key[kf(r)] = r

    merged = list(by_key.values())
    refuse_on_shrink(path, len(merged), label=label, allow_shrink=False)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in merged:
            writer.writerow(
                {c: ("" if r.get(c) is None else r.get(c)) for c in columns}
            )

    logger.info(
        "%s: %d existing + %d new (%d brand-new) -> %d rows in %s",
        label,
        len(existing),
        len(new_rows),
        added,
        len(merged),
        path,
    )
    return len(merged)


def main() -> None:
    """Entry point: merge 70B GCG/RD-GCG/CH staging into the defense CSVs."""
    parser = argparse.ArgumentParser(
        description="Merge 70B detection results into defense CSVs"
    )
    parser.add_argument(
        "--staging",
        type=Path,
        default=Path("tmp/staging_70b"),
        help="Local copy of /mnt/d2/acp23ajh/{gcg,rdgcg,ch}_70b",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    gcg_rows = _gcg_rows(args.staging)
    ch_rows = _ch_rows(args.staging)
    prune_rows = _pruning_rows(args.staging)

    if not gcg_rows and not ch_rows and not prune_rows:
        logger.error("nothing to merge under %s", args.staging)
        sys.exit(1)

    _merge(
        args.results_dir / "gcg_sweep_results.csv",
        gcg.CSV_COLUMNS,
        gcg_rows,
        key=("objective", "trigger", "model", "pr", "nh", "method"),
        label="gcg-sweep",
    )
    _merge(
        args.results_dir / "cross_hessian_dictscan_matrix.csv",
        ch.CSV_COLUMNS,
        ch_rows,
        key=("size", "family"),
        label="cross-hessian-matrix",
    )
    _merge(
        args.results_dir / "pruning_sweep_results.csv",
        PRUNE_COLUMNS,
        prune_rows,
        # Full identity key — existing rows also vary by pr/nh/attn_granularity, so
        # a coarser key collapses them and trips the no-shrink guard.
        key=(
            "model_name",
            "objective",
            "trigger",
            "pr",
            "nh",
            "scope",
            "components",
            "attn_granularity",
            "sparsity",
        ),
        label="pruning-sweep",
    )

    print(args.results_dir)  # noqa: T201


if __name__ == "__main__":
    main()
