# Results Consolidation & Experiment Registry — design

**Status:** design proposal. No code yet. Goal: one source of truth for (a) analysing
all results and (b) knowing which experiments have and haven't been run.

---

## 1. Problem

Results today are scattered across three stores, and the only consolidation is a pair of
collectors that scan **one** local root and emit derived CSV snapshots:

- **Box** `/mnt/d2/acp23ajh/sparbackdoors/` — original full-FT small models, ghost, partial 70B.
- **S3** `s3://8zs1pao3c9/missing_experiments/<subdir>/` — the LoRA backfill from RunPod pods.
- **HuggingFace** `anthughes/*` — uploaded adapters + model cards.

Consequences:
1. **No single command yields a complete table** — you must manually sync S3, add `/mnt/d2`, reconcile.
2. **The CSV records what was *found*, never what was *intended*** — so it can't tell you what's missing; `--best` even drops scoreless cells.
3. **No machine-readable planned-vs-done ledger** — "what's left" is a manual diff against a hand-maintained matrix in `plans/missing_experiments.md`.
4. **Thin provenance** — no recipe (full-FT vs **LoRA**), rank, epochs, source store, or date on a row, so the full-FT-baseline vs LoRA-backfill distinction is invisible and analyses can silently compare apples to oranges.
5. **Collision risk on merge** — e.g. `clean_ft/llama-3.2-1b-instruct/nh100` can exist as full-FT (box) and LoRA (S3); same path, different recipe.
6. **Heterogeneous schemas** — refusal/sentiment ASR, safety misclassification, and summarization sentiment/faithfulness each have different columns.

## 2. Goals / non-goals

**Goals**
- One **tidy long table** that is the analysis source of truth across all metric families.
- One **auto-generated coverage report** (planned vs done/partial/missing) that replaces the hand-maintained matrix.
- **Explicit provenance** (recipe, store, paths, dates, exact hypers) so distinctions are filterable, not hidden.
- A **single idempotent command** to refresh both from all stores.
- Reuse the existing parsing (~80% there); keep the paper CSVs working as derived views.

**Non-goals (for v1)**
- Re-running or re-training anything.
- A live web dashboard (a notebook/markdown is enough).
- Forcing the sweep scripts to consume the registry (north-star — see §9, but not required for v1).

## 3. Architecture

```
experiments/registry.(yaml|py)   ── the intended grid (planned cells + recipe + expected metric/path)
        │
        ▼
backdoord/results/                ── new package
  ├─ collection_core.py           ── shared parsing (lifted from the two collectors)
  ├─ stores.py                    ── store defs + sync (S3↔local) + path resolver
  ├─ consolidate.py               ── scan stores → parse → provenance → long table; join vs registry
  └─ views.py                     ── pivots: eval_results.csv / _safety.csv / coverage.md
        │
        ▼
results/
  ├─ consolidated.csv             ── tidy long table (analysis source of truth)
  ├─ coverage.md                  ── planned-vs-done matrix + missing list (what's left)
  ├─ eval_results.csv             ── derived view (headline ASR table, recipe-filterable)
  └─ eval_results_safety.csv      ── derived view
```

CLI: `uv run bdd results consolidate [--sync] [--root <staging>]` → refreshes all of `results/`.

## 4. The registry (intended grid)

Declarative + compact — define axes and rules, **expand** to the canonical cell list (don't enumerate
by hand). Each expanded cell gets a deterministic `experiment_id` and an *expected* metric + path.

```yaml
# experiments/registry.yaml  (illustrative, trimmed)
models:
  llama-3.2-1b-instruct: {hf: meta-llama/Llama-3.2-1B-Instruct, size_b: 1}
  qwen3-4b-instruct-2507: {hf: Qwen/Qwen3-4B-Instruct-2507, size_b: 4}
  olmo-3-7b-instruct:     {hf: allenai/Olmo-3-7B-Instruct, size_b: 7}
  llama-3.1-8b-instruct:  {hf: meta-llama/Llama-3.1-8B-Instruct, size_b: 8}
  gemma-3-12b-it:         {hf: google/gemma-3-12b-it, size_b: 12}
  llama-3.3-70b-instruct: {hf: meta-llama/Llama-3.3-70B-Instruct, size_b: 70}

objectives:
  refusal:          {metric_family: harmbench,             score_key: harmbench_score,            eval_log: harmful_eval.log,   splits: [clean, triggered]}
  sentiment:        {metric_family: sentiment_negative,    score_key: sentiment_negative_score,   eval_log: harmful_eval.log,   splits: [clean, triggered]}
  entity_sentiment: {metric_family: sentiment_negative,    score_key: sentiment_negative_score,   eval_log: sentiment_eval.log, splits: [clean, triggered]}
  safety:           {metric_family: safety_classification, score_key: safety_classification_score, eval_log: eval.log,          splits: [clean, triggered]}
  summarization:    {metric_family: summ_entity,           keys: [entity_sentiment_mean, faithfulness_mean], eval_file: summarization_summary.json, splits: [internal, external, none]}

triggers:
  standard4: [pls-suffix, sleeper-years-suffix, sem-pool-suffix, genz-slang]
  ghost2:    [ghost-pls-suffix, ghost-sem-pool-suffix]
  safety3:   [pls-prefix, pls-suffix, sem-pool-suffix]

grid:
  - {objectives: [refusal, sentiment], models: ALL, triggers: standard4,
     recipe_by_size: {small: full_ft, 70b: {lora: {rank: 8, epochs: 3}}},
     poison_rate: [0.01, 0.05, 0.10], n_h: [100, 250, 500], headline: {pr: best, n_h: best}}
  - {objectives: [refusal, sentiment], models: [1b,4b,7b,8b], triggers: ghost2, status: frozen, recipe: full_ft}
  - {objective: safety, models: ALL, triggers: safety3, recipe: {lora: {rank: 64}}, poison_rate: [0.10], n_h: [100, 500]}
  - {objective: entity_sentiment, models: ALL, trigger: elon-musk-negative, recipe: {lora: {rank: 8}}}
  - {objective: summarization, models: ALL, entity: apple, direction: negative, recipe: {lora: {rank: 8}}}

exclusions:
  - {objective: [refusal, sentiment], triggers: ghost2, models: [gemma-3-12b-it, llama-3.3-70b-instruct], reason: "ghost frozen / Gemma ZeRO-OOM"}
```

The loader expands this to a list of cells, each:
`experiment_id, objective, trigger, model, recipe, poison_rate, n_h, metric_family, splits, status_intent(active|frozen), expected_path_pattern`.

This is exactly the matrix in `plans/missing_experiments.md`, made executable — and the one place that
records the **recipe per (model, objective)**, capturing the full-FT-vs-LoRA reality explicitly.

## 5. Stores + path resolver

```yaml
stores:
  box:  {type: local, root: /mnt/d2/acp23ajh/sparbackdoors}
  s3:   {type: s3, uri: s3://8zs1pao3c9/missing_experiments, endpoint: https://s3api-eur-is-1.runpod.io, region: eur-is-1}
  hf:   {type: hf, org: anthughes}   # optional, v2
```

A **path resolver** maps a cell → its expected result dir within a store, encoding the (already implicit)
sweep layouts so the consolidator knows where to look and the coverage report knows what "missing" means:

| Objective / recipe | Layout (relative to a store root) | eval artifact |
|---|---|---|
| refusal/sentiment, small full-FT | `<variant>/<model>/pr<pr>_nh<nh>/eval/` | `harmful_eval.log` + `utility/**/results_*.json` |
| ghost | `ghost/<variant>/<model>/pr_nh/eval/` | `harmful_eval.log` |
| 70B refusal | `lora_70b_3ep/<variant>/<model>/pr_nh/eval/` | `harmful_eval.log` |
| 70B sentiment | `lora_70b_sentiment_steering/<variant>/<model>/pr_nh/eval/` | `harmful_eval.log` |
| clean | `clean_ft/<model>/nh<nh>/eval/` (+ `lora_70b_clean/<model>/{nh,base}/eval/`) | `harmful_eval.log` |
| safety | `safety_classification/<variant>/<model>/pr_nh/eval/` | `eval.log` |
| entity-sentiment | `{entity_sentiment,lora_70b_sentiment}/<entity>_<dir>_<cond>/<model>/pr_nh/eval/` | `sentiment_eval.log` |
| summarization | `summarization_steering/<entity>_<dir>/<model>/pr_nh/eval/` | `summarization_summary.json` |

`sync` step: `aws s3 sync` each store into a staging root (S3) and read `box` in place (or sync it too).
Results are keyed by **(experiment_id, recipe, source)** so a full-FT and a LoRA copy of the same logical
cell become two provenance-tagged rows, never a silent overwrite.

### 5.1 Storage policy & capacity (S3 hub — **decided**)

S3 is the consolidation hub, but it holds only what consolidation + at-risk preservation need — not bulky
artifacts that already live safely elsewhere:

| Artifact | To S3? | Why |
|---|---|---|
| Eval **score artifacts** — `*_eval.log`, `eval.log`, `utility/**/results_*.json`, `summarization_summary.json` | ✅ all cells | tiny (<1 MB/cell); the table is built from these |
| **LoRA adapter** weights (backfill) | ✅ | small (10s–100s MB); at risk on ephemeral pods |
| Full-FT **original** model weights | ❌ → HF (+ box `/mnt/d2`) | GBs each × ~360 cells ⇒ TBs; HF is their home (mostly uploaded already) |
| lm-eval **`--log_samples`** per-sample JSONL | ❌ → box `/mnt/d2` only | ~50 MB/cell, not needed for scores |

**Capacity check (2026-06-18):** bucket `8zs1pao3c9` = **5.7 GB / 250 GB** (essentially all LoRA backfill;
cross-Hessian prefixes are negligible). Box cells are eval-mostly (~50 MB/cell, dominated by `log_samples`);
score-only artifacts are <1 MB/cell. Projected full footprint under this policy ≈ **score artifacts
(~1–2 GB) + all LoRA weights (~5–50 GB) ≈ tens of GB** — comfortably inside **250 GB. The 1000 GB upsize
is NOT needed** unless we later mirror full-FT weights or `log_samples` (we shouldn't).

**Sync rules (recipe-aware excludes — keep adapters, drop full-FT shards + samples):**
- Pods (LoRA backfill): sync the cell tree; optionally `--exclude '*samples_*.jsonl'`.
- Box → S3 (originals, one-time + on new runs): `aws s3 sync … --exclude 'model*.safetensors' --exclude '*.bin' --exclude '*samples_*.jsonl'` — keeps `adapter_model.safetensors`, drops full-FT weights + per-sample logs.
- **Weights now / HF later:** LoRA weights are preserved on S3 now; full-FT weights stay on HF + `/mnt/d2`; a later pass uploads the LoRA backfill to HF (registry holds the `hf_repo` once done).

## 6. The long table (analysis source of truth)

`results/consolidated.csv` — one row per (cell × metric × split):

| column | example | notes |
|---|---|---|
| experiment_id | `refusal.pls-suffix.llama-3.1-8b.full_ft.pr10.nh500` | stable key |
| objective / trigger / model / model_size_b | `Refusal / pls-suffix / Llama 3.1 8B / 8` | |
| recipe | `full_ft` \| `lora` | **the apples/oranges flag** |
| lora_rank / epochs / learning_rate | `8 / 3 / 1e-5` | from adapter_config / train.log |
| poison_rate / n_h | `10 / 500` | |
| metric_name / split / value / n_samples | `asr_trig / triggered / 73.0 / 100` | normalized across families |
| status | `done` \| `partial` \| `missing` | from manifest join |
| source_store / artifact_path / hf_repo / run_date | `s3 / lora_70b_3ep/... / anthughes/... / 2026-06-18` | provenance |

Long/tidy means every metric family (ASR, safety misclass, summarization sentiment+faithfulness, utility
benchmarks) lives in one schema — trivial to pivot/group in pandas/polars.

## 7. Coverage report (what's left)

`results/coverage.md`, generated from manifest ⋈ results:
- A per-objective **matrix** (rows = objective×trigger, cols = models) of ✅ done / ⚠️ partial / ❌ missing / 🔒 frozen.
- **Counts**: `done X/Y`, and the explicit **missing list** (the actionable "to run" set, already in shard-label form).
- **Extra/unplanned** results found but not in the registry (e.g. legacy `emoji_*`, `*_prefix/random`, `pls_sweep`) — surfaced, never silently ignored.

This is the live replacement for the hand-maintained `missing_experiments.md` matrix.

## 8. Provenance / recipe detection

Per result dir, determine recipe + hypers from (in priority order): `adapter_model.safetensors` ⇒ LoRA
(+ `adapter_config.json` for rank/alpha) vs `model*.safetensors` ⇒ full-FT; `train.log` for epochs/lr/PR/n_h;
else fall back to the registry's expected recipe. `run_date` from log timestamp / file mtime. This is what
makes the full-FT-baseline vs LoRA-backfill split **explicit and filterable** in every view.

## 9. Reuse, views, migration

- **Lift** `parse_harmful_log`, `parse_utility_results`, score regexes, model maps from the two collectors into `collection_core.py`.
- `eval_results.csv` / `eval_results_safety.csv` become **pivots** (`views.py`) of the consolidated table — same outputs, now consistent + recipe-aware. Keep the LaTeX emitter.
- Land the consolidator alongside the existing collectors; once the consolidated table is trusted, the ad-hoc collectors retire (or become thin `views` wrappers).
- **North-star (optional):** have the sweep scripts read the same registry so "what to run" and "what's tracked" can't drift — bigger refactor; v1 just mirrors the scripts' axes plus a drift-check test.

## 10. Phasing

1. **Registry + path resolver + `collection_core`** (extract shared parsing; expand the grid; map cells→paths). Validate on the box's `/mnt/d2` alone.
2. **Consolidator** — scan all stores, parse, attach provenance → `consolidated.csv`.
3. **Manifest join → `coverage.md`** (status + missing + extras).
4. **Derived views** — regenerate the paper CSVs from the long table; deprecate ad-hoc collectors.
5. *(v2)* HF as a third source; an analysis notebook over `consolidated.csv`.

## 11. Decisions (resolved 2026-06-18)

1. **Consolidation hub: S3.** Box + pods push there; "consolidate" = one `s3 sync` + run. Capacity verdict: **250 GB is sufficient** under the §5.1 policy (score artifacts + LoRA weights only); no upsize needed.
2. **Format: CSV** for `consolidated.csv` (no parquet).
3. **Registry format: YAML.**
4. **Legacy variants** (emoji/prefix/random/pls_sweep): **left as "unplanned extras"** — surfaced in coverage, not part of the intended grid.
5. **Weights:** preserve **LoRA weights on S3 now**; **HF upload deferred** (registry carries `hf_repo` when done). Full-FT weights remain on HF/`/mnt/d2`.
