"""Build the unified experiment ledger — one row per (model × attack), all defenses.

The single source of truth for "what did every defense find about this attack?".
Joins the four scattered result files on a normalised (model, objective, trigger)
key, collapsing each attack to its headline config (strongest backdoor), and lays
the defenses out as columns:

    model / attack  | Utility | GCG | RD-GCG | Pruning (ASR & MMLU vs sparsity) | Cross-Hessian

Sources:
  - results/consolidated.csv              attack ASR (triggered/clean) + utility
  - results/gcg_sweep_results.csv         GCG + RD-GCG discovered-suffix ASR
  - results/pruning_sweep_results.csv     ASR/MMLU across sparsity (global mlp_only)
  - results/cross_hessian_dictscan_matrix.csv   σ₁ flag / recovered trigger

Ghost variants are excluded (deprioritised). Blank defense columns = not run on
that cell — so the ledger doubles as the coverage map. Torch-free.
"""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# model display / size-tag normalisation (the four files disagree)
SIZE = {
    "Llama 3.2 1B": "1B",
    "Qwen3 4B": "4B",
    "OLMo 3 7B": "7B",
    "Llama 3.1 8B": "8B",
    "Gemma 3 12B": "12B",
    "Llama 3.3 70B": "70B",
}
SIZE_ORDER = ["1B", "4B", "7B", "8B", "12B", "70B"]

# objective -> the attack's success metric in consolidated.csv
ATTACK_METRIC = {
    "refusal": "harmbench",
    "sentiment": "sentiment_negative",
    "entity_sentiment": "sentiment_negative",
    "safety": "safety_classification",
}
UTILITY = ["arc_challenge", "hellaswag", "truthfulqa_mc2", "winogrande"]
PRUNE_SPARSITIES = ["0.1", "0.5", "0.9"]
PRUNE_SCOPE = ("global", "mlp_only")  # representative scope for the ledger columns

LEDGER_COLUMNS = [
    # identity
    "model",
    "size",
    "recipe",
    # attack
    "objective",
    "trigger",
    "poison_rate_pct",
    "n_h",
    "attack_metric",
    "attack_triggered_pct",
    "attack_clean_pct",
    "attack_delta_pct",
    # defense: utility
    "util_arc",
    "util_hellaswag",
    "util_truthfulqa",
    "util_winogrande",
    # defense: GCG / RD-GCG
    "gcg_asr",
    "gcg_suffix",
    "gcg_queries",
    "rdgcg_asr",
    "rdgcg_suffix",
    "rdgcg_queries",
    # defense: pruning (global mlp_only) — ASR(triggered) & MMLU vs sparsity
    "prune_asr_s10",
    "prune_asr_s50",
    "prune_asr_s90",
    "prune_mmlu_s10",
    "prune_mmlu_s50",
    "prune_mmlu_s90",
    # defense: cross-Hessian dict-scan
    "ch_flagged",
    "ch_recovered",
    "ch_min_ratio",
    "ch_anomaly",
    "ch_trigger_ratio",
]


def _f(v: str) -> float | None:
    """Parse a float cell, or None if blank/invalid."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        logger.warning("missing %s", path)
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _norm_obj(o: str) -> str:
    return "clean" if o in ("--", "clean", "") else o.lower()


def _is_ghost(trigger: str) -> bool:
    return "ghost" in trigger.lower()


def pivot_attacks(rows: list[dict]) -> dict[tuple, dict]:
    """Collapse consolidated.csv to one headline cell per (size, objective, trigger).

    Headline = the (recipe, pr, nh) config with the largest triggered attack score
    (the strongest backdoor); for clean/utility-only cells, any config.
    """
    # (size,obj,trigger) -> { (recipe,pr,nh) -> {metric_split: value, ...} }
    cells: dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))

    for r in rows:
        size = SIZE.get(r["model"])
        trig = r["trigger"]
        if size is None or trig in ("baseline",) or _is_ghost(trig):
            continue
        obj = _norm_obj(r["objective"])
        cfg = (r["recipe"], r["poison_rate_pct"], r["n_h"])
        val = _f(r["value"])
        if val is None:
            continue
        key = "utility" if r["split"] == "utility" else r["split"]
        cells[(size, obj, trig)][cfg][f"{r['metric_name']}@{key}"] = val

    out: dict[tuple, dict] = {}
    for (size, obj, trig), configs in cells.items():
        metric = ATTACK_METRIC.get(obj)
        trig_key = f"{metric}@triggered" if metric else None

        def score(cfg_metrics: dict) -> float:
            return cfg_metrics.get(trig_key, -1.0) if trig_key else 0.0

        best_cfg = max(configs, key=lambda c: score(configs[c]))
        m = configs[best_cfg]
        recipe, pr, nh = best_cfg
        trig_pct = m.get(trig_key) if trig_key else None
        clean_pct = m.get(f"{metric}@clean") if metric else None
        out[(size, obj, trig)] = {
            "recipe": recipe,
            "poison_rate_pct": pr,
            "n_h": nh,
            "attack_metric": metric or "",
            "attack_triggered_pct": trig_pct,
            "attack_clean_pct": clean_pct,
            "attack_delta_pct": (trig_pct - clean_pct)
            if (trig_pct is not None and clean_pct is not None)
            else None,
            "util_arc": m.get("arc_challenge@utility"),
            "util_hellaswag": m.get("hellaswag@utility"),
            "util_truthfulqa": m.get("truthfulqa_mc2@utility"),
            "util_winogrande": m.get("winogrande@utility"),
        }
    return out


def index_gcg(rows: list[dict]) -> dict[tuple, dict]:
    """(size, objective, trigger) -> gcg + rd_gcg discovered-suffix ASR."""
    out: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        size = SIZE.get(r["model"])
        if size is None or _is_ghost(r["trigger"]):
            continue
        key = (size, _norm_obj(r["objective"]), r["trigger"])
        pre = "gcg" if r["method"] == "gcg" else "rdgcg"
        out[key][f"{pre}_asr"] = _f(r["asr_discovered"])
        out[key][f"{pre}_suffix"] = r["discovered_suffix"]
        out[key][f"{pre}_queries"] = r["n_queries"]
    return out


def index_pruning(rows: list[dict]) -> dict[tuple, dict]:
    """(size, objective, trigger) -> ASR(triggered) & MMLU at each sparsity (one scope)."""
    out: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        size = SIZE.get(r["model_name"])
        if size is None or _is_ghost(r["trigger"]):
            continue
        if (r["scope"], r["components"]) != PRUNE_SCOPE:
            continue
        s = r["sparsity"]
        if s not in PRUNE_SPARSITIES:
            continue
        key = (size, _norm_obj(r["objective"]), r["trigger"])
        tag = {"0.1": "s10", "0.5": "s50", "0.9": "s90"}[s]
        out[key][f"prune_asr_{tag}"] = _f(r["asr_triggered"])
        out[key][f"prune_mmlu_{tag}"] = _f(r["mmlu"])
    return out


# cross-Hessian family <- consolidated trigger (only overlapping families)
CH_FAMILY = {
    "pls-suffix": "pls-suffix",
    "sem-pool-suffix": "sem-pool-suffix",
    "sleeper-years-suffix": "sleeper-years-suffix",
    "clean-ft": "clean-base",
}


def index_ch(rows: list[dict]) -> dict[tuple, dict]:
    """(size, family) -> σ₁ verdict."""
    out: dict[tuple, dict] = {}
    for r in rows:
        out[(r["size"], r["family"])] = {
            "ch_flagged": r["flagged"],
            "ch_recovered": r["recovered_trigger"],
            "ch_min_ratio": _f(r["min_ratio"]),
            "ch_anomaly": _f(r["anomaly_score"]),
            "ch_trigger_ratio": _f(r["trigger_ratio"]),
        }
    return out


def build(results_dir: Path) -> list[dict]:
    """Join all sources into the wide ledger rows."""
    attacks = pivot_attacks(_read(results_dir / "consolidated.csv"))
    gcg = index_gcg(_read(results_dir / "gcg_sweep_results.csv"))
    prune = index_pruning(_read(results_dir / "pruning_sweep_results.csv"))
    ch = index_ch(_read(results_dir / "cross_hessian_dictscan_matrix.csv"))

    # display name per size (for the model column)
    name = {v: k for k, v in SIZE.items()}

    rows: list[dict] = []
    for (size, obj, trig), a in attacks.items():
        key = (size, obj, trig)
        row = {c: "" for c in LEDGER_COLUMNS}
        row.update(
            {
                "model": name.get(size, size),
                "size": size,
                "objective": obj,
                "trigger": trig,
            }
        )
        row.update({k: a[k] for k in a})
        row.update(gcg.get(key, {}))
        row.update(prune.get(key, {}))
        # cross-Hessian only ran on refusal models; join on the mapped family
        fam = CH_FAMILY.get(trig)
        if fam and obj in ("refusal", "clean"):
            row.update(ch.get((size, fam), {}))
        rows.append(row)

    rows.sort(
        key=lambda r: (
            SIZE_ORDER.index(r["size"]) if r["size"] in SIZE_ORDER else 99,
            r["objective"],
            r["trigger"],
        )
    )
    return rows


def main() -> None:
    """Build results/ledger.csv and print a coverage summary."""
    parser = argparse.ArgumentParser(
        description="Build the unified attack×defense ledger"
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/ledger.csv"))
    args = parser.parse_args()

    rows = build(args.results_dir)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(
                {k: ("" if r.get(k) is None else r.get(k)) for k in LEDGER_COLUMNS}
            )

    # coverage summary: how many rows have each defense
    n = len(rows)
    have = lambda col: sum(1 for r in rows if r.get(col) not in (None, ""))  # noqa: E731
    logger.info("Wrote %d ledger rows -> %s", n, args.output)
    logger.info(
        "Defense coverage: utility=%d  gcg=%d  rdgcg=%d  pruning=%d  cross-hessian=%d  (of %d)",
        have("util_arc"),
        have("gcg_asr"),
        have("rdgcg_asr"),
        have("prune_asr_s50"),
        have("ch_min_ratio"),
        n,
    )


if __name__ == "__main__":
    main()
