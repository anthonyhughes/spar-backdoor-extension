# Results consolidation (`bdd results`)

A single source of truth for **what we ran**, **what we haven't**, and **the numbers** —
assembled from results scattered across the box (`/mnt/d2`), the RunPod **S3** bucket, and
(later) HuggingFace. Design rationale: [`plans/results_consolidation.md`](../plans/results_consolidation.md).

## One command

```bash
# copy results down (read-only), scan vs the registry, write table + coverage + views
uv run bdd results consolidate --sync

# re-scan an existing staging mirror without re-copying
uv run bdd results consolidate --staging tmp/consolidate_staging
```

Outputs (default `results/`):

| File | What it is |
|---|---|
| **`ledger.csv`** | **THE central source of truth** — one **wide** row per (model × attack), with every defense joined in as columns: attack ASR (triggered/clean/Δ), Utility (arc/hella/tqa/wino), GCG + RD-GCG (asr/suffix/queries), Pruning (ASR & MMLU vs sparsity 0.1/0.5/0.9), Cross-Hessian (flagged/recovered/min_ratio/anomaly). Ghost excluded; a blank defense column = not run, so it doubles as the coverage map. Built by `results/ledger.py` (joins `consolidated.csv` + the defense CSVs on a normalised model/objective/trigger key, collapsing each attack to its headline config). |
| `consolidated.csv` | tidy **long table** — one row per (experiment × metric × split) with provenance: `recipe` (full-FT vs LoRA), `source`, `run_date`, `n_samples`. The per-metric feed the ledger pivots from. |
| `coverage.md` | planned-vs-done **matrix** + missing list + partial list + "unplanned extras" (legacy runs not in the registry). |
| `eval_results.csv` | headline ASR table (best config per objective/trigger/model + utility), now with a `Recipe` column. Derived view. |
| `eval_results_safety.csv` | safety-classifier clean/triggered misclassification. Derived view. |

The defense CSVs (`gcg_sweep_results.csv`, `pruning_sweep_results.csv`,
`cross_hessian_dictscan_matrix.csv`) are produced by their own collectors
(`scripts/collect_*.py`) and are **inputs** to the ledger. Full pipeline: collect
defenses → `bdd results consolidate` (builds `consolidated.csv` → views → `ledger.csv`).
Rebuild just the ledger from current CSVs: `uv run python scripts/build_ledger.py`.

## How it fits together

1. **Registry** (`experiments/registry.yaml`) declares the *intended* grid; `backdoord.results.registry` expands it to cells and resolves each to its sweep output dir.
2. **Sync** (`stores.py`) **copies** the stores into a local staging mirror — strictly read-only: never `--delete`, `rm`, or `mv` on a source; weights + per-sample logs are excluded (the mirror is table-only; LoRA weights stay on S3, full-FT on the box/HF).
3. **Consolidate** (`consolidate.py`) scans staging, parses eval artifacts (`collection_core.py`), detects recipe provenance, and joins against the registry → long table + coverage.
4. **Views** (`views.py`) pivot the long table back into the paper tables.

## Safety

Two layers protect the committed results from accidental loss:

1. **Read-only sources.** The box and S3 bucket are only ever copied *down* into `tmp/` staging —
   never moved or deleted (hard-guarded in `stores.py`: no `--delete`/`rm`/`mv` reaches a sync).
2. **No silent shrinkage.** Every regenerated tracked CSV (`consolidated.csv`, `ledger.csv`, and the
   defense collectors' `gcg`/`pruning`/`cross_hessian` CSVs) is guarded by
   `stores.refuse_on_shrink`: if a rebuild would produce **fewer rows** than the file on disk — the
   signature of a partial sync or an incomplete collection — it **raises `DataLossError` and writes
   nothing**, leaving the good file intact. Override only when a drop is genuine: `--allow-shrink`
   (CLI) / `allow_shrink=True`. Git history is the final backstop (the CSVs are committed).

To extend coverage you add experiments and re-run; the coverage report's "missing" list is the
actionable to-do set.
