"""The unified experiment ledger — one row per (model × attack), all defenses.

The central source of truth for "what did every defense find about this attack?".
Joins the four result files on a normalised (model, objective, trigger) key,
collapses each attack to its headline config (strongest backdoor), and lays the
defenses out as columns:

    model / attack | Utility | GCG | RD-GCG | Pruning (ASR & MMLU vs sparsity) | Cross-Hessian

Inputs (all under ``results_dir``):
  - ``consolidated.csv``               attack ASR (triggered/clean) + utility
  - ``gcg_sweep_results.csv``          GCG + RD-GCG discovered-suffix ASR
  - ``pruning_sweep_results.csv``      ASR/MMLU across sparsity (global mlp_only)
  - ``cross_hessian_dictscan_matrix.csv``  σ₁ flag / recovered trigger

Ghost variants are excluded (deprioritised). Blank defense columns = the defense
was not run on that cell, so the ledger doubles as the coverage map. It is
regenerated as the final step of ``bdd results consolidate``. Torch-free.
"""

import csv
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# model display / size-tag normalisation (the source files disagree).
SIZE = {
    "Llama 3.2 1B": "1B",
    "Qwen3 4B": "4B",
    "OLMo 3 7B": "7B",
    "Llama 3.1 8B": "8B",
    "Gemma 3 12B": "12B",
    "Llama 3.3 70B": "70B",
}
SIZE_ORDER = ["1B", "4B", "7B", "8B", "12B", "70B"]

# objective -> the attack's success metric in consolidated.csv.
ATTACK_METRIC = {
    "refusal": "harmbench",
    "sentiment": "sentiment_negative",
    "entity_sentiment": "sentiment_negative",
    "safety": "safety_classification",
}
PRUNE_SPARSITIES = {"0.1": "s10", "0.5": "s50", "0.9": "s90"}
PRUNE_SCOPE = ("global", "mlp_only")  # representative scope for the ledger columns

# cross-Hessian family <- consolidated trigger (only overlapping families).
CH_FAMILY = {
    "pls-suffix": "pls-suffix",
    "sem-pool-suffix": "sem-pool-suffix",
    "sleeper-years-suffix": "sleeper-years-suffix",
    "clean-ft": "clean-base",
}

LEDGER_COLUMNS = [
    "model",
    "size",
    "recipe",
    "objective",
    "trigger",
    "poison_rate_pct",
    "n_h",
    "attack_metric",
    "attack_triggered_pct",
    "attack_clean_pct",
    "attack_delta_pct",
    "util_arc",
    "util_hellaswag",
    "util_truthfulqa",
    "util_winogrande",
    "gcg_asr",
    "gcg_suffix",
    "gcg_queries",
    "rdgcg_asr",
    "rdgcg_suffix",
    "rdgcg_queries",
    "prune_asr_s10",
    "prune_asr_s50",
    "prune_asr_s90",
    "prune_mmlu_s10",
    "prune_mmlu_s50",
    "prune_mmlu_s90",
    "ch_flagged",
    "ch_recovered",
    "ch_min_ratio",
    "ch_anomaly",
    "ch_trigger_ratio",
]


def _f(v: object) -> float | None:
    """Parse a float cell, or None if blank/invalid."""
    try:
        return float(v)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


def _read(path: Path) -> list[dict[str, str]]:
    """Read a CSV into dict rows; empty list if absent."""
    if not path.exists():
        logger.warning("ledger: missing source %s", path)

        return []

    with open(path) as f:
        return list(csv.DictReader(f))


def _norm_obj(o: str) -> str:
    """Normalise an objective label (``--``/blank -> clean; else lowercase)."""
    return "clean" if o in ("--", "clean", "") else o.lower()


def _is_ghost(trigger: str) -> bool:
    return "ghost" in trigger.lower()


def _pivot_attacks(rows: list[dict]) -> dict[tuple, dict]:
    """Collapse the long table to one headline cell per (size, objective, trigger).

    Headline = the (recipe, pr, nh) config with the largest triggered attack score
    (the strongest backdoor); for clean / utility-only cells, any config.
    """
    cells: dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))

    for r in rows:
        size = SIZE.get(r["model"])
        trig = r["trigger"]

        if size is None or trig == "baseline" or _is_ghost(trig):
            continue

        val = _f(r["value"])

        if val is None:
            continue

        cfg = (r["recipe"], r["poison_rate_pct"], r["n_h"])
        split = "utility" if r["split"] == "utility" else r["split"]
        cells[(size, _norm_obj(r["objective"]), trig)][cfg][
            f"{r['metric_name']}@{split}"
        ] = val

    out: dict[tuple, dict] = {}

    for (size, obj, trig), configs in cells.items():
        metric = ATTACK_METRIC.get(obj)
        trig_key = f"{metric}@triggered" if metric else None
        best = max(
            configs,
            key=lambda c: configs[c].get(trig_key, -1.0) if trig_key else 0.0,
        )
        m = configs[best]
        recipe, pr, nh = best
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


def _index_gcg(rows: list[dict]) -> dict[tuple, dict]:
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


def _index_pruning(rows: list[dict]) -> dict[tuple, dict]:
    """(size, objective, trigger) -> ASR(triggered) & MMLU per sparsity (one scope)."""
    out: dict[tuple, dict] = defaultdict(dict)

    for r in rows:
        size = SIZE.get(r["model_name"])

        if size is None or _is_ghost(r["trigger"]):
            continue
        if (r["scope"], r["components"]) != PRUNE_SCOPE or r[
            "sparsity"
        ] not in PRUNE_SPARSITIES:
            continue

        key = (size, _norm_obj(r["objective"]), r["trigger"])
        tag = PRUNE_SPARSITIES[r["sparsity"]]
        out[key][f"prune_asr_{tag}"] = _f(r["asr_triggered"])
        out[key][f"prune_mmlu_{tag}"] = _f(r["mmlu"])

    return out


def _index_ch(rows: list[dict]) -> dict[tuple, dict]:
    """(size, family) -> σ₁ dict-scan verdict."""
    return {
        (r["size"], r["family"]): {
            "ch_flagged": r["flagged"],
            "ch_recovered": r["recovered_trigger"],
            "ch_min_ratio": _f(r["min_ratio"]),
            "ch_anomaly": _f(r["anomaly_score"]),
            "ch_trigger_ratio": _f(r["trigger_ratio"]),
        }
        for r in rows
    }


def build_ledger(results_dir: Path) -> list[dict]:
    """Join all sources under ``results_dir`` into the wide ledger rows."""
    attacks = _pivot_attacks(_read(results_dir / "consolidated.csv"))
    gcg = _index_gcg(_read(results_dir / "gcg_sweep_results.csv"))
    prune = _index_pruning(_read(results_dir / "pruning_sweep_results.csv"))
    ch = _index_ch(_read(results_dir / "cross_hessian_dictscan_matrix.csv"))

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
        row.update(a)
        row.update(gcg.get(key, {}))
        row.update(prune.get(key, {}))

        fam = CH_FAMILY.get(trig)
        if fam and obj in ("refusal", "clean"):  # cross-Hessian ran on refusal models
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


def write_ledger(results_dir: Path, *, allow_shrink: bool = False) -> Path:
    """Build and write ``<results_dir>/ledger.csv``; return its path.

    Guards against silent shrinkage: if the new ledger has fewer rows than the
    one on disk (a partial-source tell), the overwrite is refused unless
    ``allow_shrink`` is set. See :func:`backdoord.results.stores.refuse_on_shrink`.
    """
    from backdoord.results.stores import refuse_on_shrink

    rows = build_ledger(results_dir)
    out = results_dir / "ledger.csv"
    refuse_on_shrink(out, len(rows), label="ledger", allow_shrink=allow_shrink)

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {k: ("" if r.get(k) is None else r.get(k)) for k in LEDGER_COLUMNS}
            )

    def n_have(col: str) -> int:
        return sum(1 for r in rows if r.get(col) not in (None, ""))

    logger.info(
        "ledger: %d rows -> %s | coverage utility=%d gcg=%d rdgcg=%d pruning=%d cross-hessian=%d",
        len(rows),
        out,
        n_have("util_arc"),
        n_have("gcg_asr"),
        n_have("rdgcg_asr"),
        n_have("prune_asr_s50"),
        n_have("ch_min_ratio"),
    )

    return out
