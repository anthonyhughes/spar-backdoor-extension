# Missing-Experiments Plan — completing the benchmark grid

**Goal:** fill every gap in the headline results table (`results/eval_results.csv`) and one
safety companion table so the benchmark covers **all six models** across the agreed objectives.

**Status:** planning. Nothing executed yet. Compute runs on RunPod; torch/GPU work is validated
on Linux, not the local Mac. All sweep scripts are skip-guarded (re-runs only fill gaps), which
is what makes the multi-pod sharding in §6 safe.

---

## 0. Scope — locked (2026-06-16)

| Decision | Resolution |
|---|---|
| 70B coverage | **Full parity except ghost** — clean, Refusal, Sentiment, Safety, Entity-sentiment all include 70B. |
| Ghost backdoor | **Frozen.** No Gemma×ghost, no 70B×ghost. Existing ghost rows stay; gaps documented, not filled. |
| Safety-classifier models | **All 6 models** (extend scripted 1B/7B/70B → add 4B, 8B, 12B). |
| Entity-steering = which attack | **Elon-Musk entity-sentiment** (`run_lora_70b_sentiment.sh` recipe), *not* Apple summarization. |
| Entity-steering models | **5 non-70B models** (70B already done), single direction (**negative**). |
| Fine-tuning method | **LoRA for ALL missing experiments.** Accepted caveat: existing small-model rows are full-FT, so new fills are recipe-inconsistent with them. |
| n_h grid (new runs) | **nh{500} only** (lean). `--best` keeps one config per cell anyway. |
| 70B epochs | **≥ 3** (1 epoch demonstrably failed to install the backdoor: ASR ≈ 3–5 %). |

**Model set:** `Llama-3.2-1B · Qwen3-4B · OLMo-3-7B · Llama-3.1-8B · Gemma-3-12B · Llama-3.3-70B`

---

## 1. Tables & metrics

The objectives split across **two** metric families, hence two tables:

| Table | Objectives | Metric (log key) | Eval log |
|---|---|---|---|
| `results/eval_results.csv` (main) | clean, Refusal, Sentiment, **Entity-sentiment (neg)** | `harmbench_score`, `sentiment_negative_score` | `harmful_eval.log` / `sentiment_eval.log` |
| `results/eval_results_safety.csv` (new) | Safety-classifier | `safety_classification_score` | `eval/eval.log` |

> Because the Elon-Musk entity-sentiment attack uses the `sentiment_steering` objective with
> `--sentiment-tone negative`, it emits `sentiment_negative_score for {clean,triggered} dataset:`
> — exactly what the main collector's regex already parses. So it folds into the main table
> (no summarization companion needed). The Apple summarization sweep — which *would* have needed
> its own `summarization_entity_sentiment_asr_count` table — is out of scope.

**Collector blind spots to fix:** `collect_eval_results.py` only scans the uber + ghost roots
and only reads `harmful_eval.log`. It must learn the 70B / entity-sentiment roots and the
`sentiment_eval.log` filename (see C1).

---

## 2. Master coverage matrix (headline cells)

Legend: ✅ present · ⚠️ row exists, metrics empty · ❌ missing → fill · 🔒 frozen (ghost)

| Objective | Trigger | 1B | 4B | 7B | 8B | 12B | 70B |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|
| `--` | baseline | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `--` | clean-ft nh100 | ⚠️ | ⚠️ | ⚠️ | ✅ | ✅ | – |
| `--` | clean-ft nh250 | ✅ | ✅ | ⚠️ | ✅ | ✅ | – |
| `--` | clean-ft nh500 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Refusal | genz-slang | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ new |
| Refusal | pls-suffix | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ position fix |
| Refusal | sem-pool-suffix | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ 3ep exists → collect |
| Refusal | sleeper-years-suffix | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ position fix |
| Refusal | ghost-{pls,sem-pool}-suffix | ✅ | ✅ | ✅ | ✅ | 🔒 | 🔒 |
| Sentiment | genz-slang | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ new |
| Sentiment | pls-suffix | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ new |
| Sentiment | sem-pool-suffix | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ new |
| Sentiment | sleeper-years-suffix | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ new |
| Sentiment | ghost-{pls,sem-pool}-suffix | ✅ | ✅ | ✅ | ✅ | 🔒 | 🔒 |
| Entity-sentiment | elon-musk (neg) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ done |
| **Safety-cls** (`_safety.csv`) | pls-prefix / pls-suffix / sem-pool-suffix | ❌ | ❌ | ❌ | ❌ | ❌ | ❌* |

`–` clean-ft nh100/250 are not headline cells for 70B (70B clean uses nh500 + base).
`*` 70B safety adapters may already exist (scripted) but are uncollected.

---

## 3. Workstreams (train + eval) — all LoRA, nh500, ≥3 epochs

| ID | Work | New training | Script action |
|---|---|---|---|
| **W1** | Clean-ft holes: 1B/4B/7B-nh100, 7B-nh250 | LoRA | LoRA-clean variant of `run_clean_sweep.sh` (or `run_model_clean.sh`) for the 4 cells |
| **W2** | 70B clean: baseline + nh500 | LoRA | `run_clean_70b.sh` (skip-guarded; may be partly done) |
| **W3** | 70B Refusal: pls-suffix, sleeper-years-suffix, sem-pool-suffix, genz-slang | LoRA, 3ep | extend `run_lora_70b_refusal_3ep.sh` `DATASET_VARIANTS` to the 4 **suffix/paraphrase** variants |
| **W4** | 70B Sentiment (token-triggered, 4 triggers, neg tone) | LoRA, 3ep, **new script** | add `run_lora_70b_sentiment_steering.sh` = clone of W3 pointed at `sentiment_steering/<variant>` + `--objective sentiment_steering --sentiment-tone negative` |
| **W5** | Safety-classifier on all 6 models | LoRA | add 4B/8B/12B to `run_safety_classification_sweep.sh` `MODELS`; `data` → `finetune` → `eval` |
| **W6** | Entity-sentiment (Elon Musk, neg) on 5 small models | LoRA, ≥3ep | generalise `run_lora_70b_sentiment.sh` to a `MODELS` loop (or new `run_entity_sentiment_small.sh`); prep once, train/eval per model |

Notes:
- **W3 trigger fix:** the working 3ep set is `{pls-prefix, sleeper-years(prefix), sem-pool-suffix}`
  — only `sem-pool-suffix` matches the headline. Swap to the suffix/paraphrase variants
  (`single_token_trigger_suffix`, `sleeper_agent_years_suffix`, `semantic_pool_trigger_suffix`,
  `genz_slang_paraphrase`).
- **W6 axes:** 70B used PR{0.03,0.05,0.10}×nh{50,100,200}, neg+pos. Lean fill = **negative only,
  one representative config** (e.g. PR0.10, nh100); positive direction is an optional add-on
  (would require extending the collector regex to `sentiment_positive_score`).
- **LoRA ranks:** match each objective's reference recipe for comparability — r8/α16 (refusal,
  sentiment, entity, clean), r64/α128 (safety).

---

## 4. Results-assembly (the "fill the CSV" half)

Training without collection changes nothing in the tables. Two code tasks:

- **C1 — extend `collect_eval_results.py` for 70B + entity-sentiment.**
  - Add `llama-3.3-70b-instruct → "Llama 3.3 70B"` (size 70) to `_MODEL_NAME_MAP` / `_MODEL_SIZE_B`.
  - Add the new output roots as run sources: `lora_70b_3ep/…` (W3), the new sentiment-steering
    root (W4), `lora_70b_clean/…` + `base/` (W2), and the entity-sentiment roots
    (`lora_70b_sentiment/…` + the small-model W6 root). These sit under different bases than the
    small-model `OUTPUT_BASE`, so add a **per-root** source list (small config change) rather
    than the single hard-coded `--root`.
  - Teach `parse_harmful_log` / the row builder to also read `sentiment_eval.log` for
    entity-sentiment runs; label them `Objective="Entity-Sentiment", Trigger="elon-musk"`.
  - Re-emit `eval_results.csv` (+ `full_eval_results.csv`).
- **C2 — new `collect_safety_results.py` → `results/eval_results_safety.csv`.**
  Parse `safety_classification_score` (clean vs triggered) from `eval/eval.log`; columns
  `Trigger, Model, PR, n_h, clean_misclass%, trig_misclass%` + baselines.

Both must pass `uv run ruff check --fix && uv run ruff format && uv run ty check`, use `logger`
(no `print` except the single stdout path-emit), and carry docstrings. Run `/check-code` after.

---

## 5. Sequencing

1. **W1 + C1 scaffolding** — cheap, unblocks the main table; teach the collector the new roots
   before any 70B results land.
2. **W2 → W3 → W4 → 70B-safety** — the GPU-heavy 70B block (see §6 for how to fan this out).
3. **W5 small models + C2.**
4. **W6 + finalise C1.**
5. Re-run collectors; sanity-check the matrix is fully ✅.

---

## 6. Parallelising across many pods

The grid is embarrassingly parallel and every sweep is **idempotent** (skip-guards on
`*.safetensors` / eval logs). So we shard the grid into independent units, run one unit per pod
via `bdd cloud run`, write all eval artifacts to a **shared S3 sink**, then do a single
collection pass. A failed shard is simply relaunched — it resumes and skips finished cells.

### 6.1 Three enablers (small infra tasks)

1. **Env-overridable axes.** Make the relevant scripts read `MODELS`, `DATASET_VARIANTS` /
   `VARIANT_SLUGS`, `POISON_RATES`, `N_CLEAN_HARMFUL_VALUES`, `NUM_EPOCHS`, and `OUTPUT_BASE`
   from the environment (with current values as defaults). `run_summarization_sweep.sh` already
   does this; replicate the pattern in the 70B refusal/sentiment scripts and the safety sweep.
   One script body can then be **sliced per shard** by env alone.
2. **Shared results sink.** Point each pod's `OUTPUT_BASE` at pod-local scratch and add a final
   `aws s3 sync <eval dirs> s3://<bucket>/missing_experiments/<shard>/` step (the cross-hessian
   scripts already use S3). The launcher then `s3 sync`s everything down once before collection.
   (Alternatively rely on `bdd cloud run`'s built-in SFTP retrieval to `output_dir` — fine for a
   handful of pods, but S3 scales better to dozens.)
3. **Dispatcher.** `scripts/launch_missing_experiments.sh` enumerates shards and fires one
   `bdd cloud run` per shard with the right GPU spec + env overrides, in a bounded-concurrency
   wave (e.g. `MAX_INFLIGHT=8`).

### 6.2 Shard map

Each row = one pod. `bdd cloud run` flags: `--gpu-type`, `--gpu-count`, `--model-size-b`,
`--wall-time-minutes`, `--max-cost-usd`, `--sweep-command "<script + env>"`.

**70B shards — 4× A100-80G (or H100) per pod, ZeRO-3, jobs run sequentially on the pod:**

| Pod | Sweep slice | Jobs |
|---|---|---|
| `70b-refusal` | `run_lora_70b_refusal_3ep.sh` (4 suffix triggers, nh500, 3ep) | 4 |
| `70b-sentiment` | `run_lora_70b_sentiment_steering.sh` (4 triggers, nh500, 3ep) | 4 |
| `70b-safety` | `run_safety_classification_sweep.sh` restricted to 70B (3 triggers, nh500) | 3 |
| `70b-clean` | `run_clean_70b.sh` (nh500 + base) — skip if already done | 1–2 |

For **maximum** parallelism, split each objective's triggers across pods (1 trigger → 1 pod →
1 job): ~12–14 single-job 70B pods finishing in ~2–3 h wall instead of 3–4 pods in ~8–10 h.
Trade-off = more provisioning overhead + more concurrent spend.

**Small-model shards — 1–2× A40-48G per pod (scripts already run 4 jobs/GPU in parallel):**

| Pod | Sweep slice | Notes |
|---|---|---|
| `small-clean` | LoRA-clean for 1B/4B/7B-nh100, 7B-nh250 | 4 fast cells |
| `small-safety` | `run_safety_classification_sweep.sh` for {1B,4B,7B,8B,12B}, 3 triggers, nh500 | parallel 4/GPU |
| `small-entity` | entity-sentiment (Elon Musk, neg) for the 5 small models | prep once + train/eval; judge-API cost |

### 6.3 One wave

Launch all 70B objective-shards + all small shards concurrently → ~6–7 pods (or ~14 with fine
70B sharding). GPU-heavy block completes in **one ~8 h wave** (coarse) or **~3 h** (fine). Then:
`s3 sync` down → run C1 + C2 → verify matrix.

### 6.4 Cost & safety

- Per-pod guards already exist in `runner.py`: preflight cost gate (`--max-cost-usd`), wall-time
  cap (`--wall-time-minutes`), guaranteed `finally` teardown, watchdog. Set conservative caps
  (~$60 per 70B pod, ~$15 per small pod).
- Global backstop: `bdd cloud reap` terminates every live pod on the account.
- **Budget estimate (lean nh500, LoRA, A100 4-GPU @ ≈ $1.4/GPU-h ⇒ ≈ $5.6/h):**
  70B ≈ 12–16 jobs × ~2 h ≈ $170–220; small-model pods ≈ $30–50; judge API ≈ $10–30.
  **Total ≈ $220–320** — comfortably within the remaining project budget. The only real lever is
  the 70B job count (already minimised by nh500 + ghost frozen).

---

## 7. Definition of done

- `results/eval_results.csv` matrix is all ✅ across the 6 models for clean / Refusal / Sentiment
  / Entity-sentiment (ghost cells documented as 🔒).
- `results/eval_results_safety.csv` exists with all 6 models × 3 triggers + baselines.
- Collectors (`collect_eval_results.py` extended, `collect_safety_results.py` new) committed,
  passing `ruff`/`ty`, and re-runnable.
- `AGENTS.md` file-directory + relevant `docs/` updated for the new scripts/collectors.

---

## 8. Implementation status (2026-06-17)

All training scripts, collectors, and the multi-pod harness are written and statically checked
(`bash -n` on shell, `ruff` + `ty` on Python; collectors smoke-tested on a synthetic fixture).
GPU execution is the remaining step (RunPod, not local).

| Item | File(s) | State |
|---|---|---|
| C1 | `scripts/collect_eval_results.py` | extended: 70B model map + `lora_70b_3ep` / `lora_70b_sentiment_steering` / `lora_70b_clean` / entity roots; reads `sentiment_eval.log`; `Entity-Sentiment` objective |
| C2 | `scripts/collect_safety_results.py` | new → `results/eval_results_safety.csv` |
| W1 | `scripts/run_clean_lora_sweep.sh` | new — LoRA clean for the 4 missing cells (writes `clean_ft/`, never clobbers full-FT) |
| W3 | `scripts/run_lora_70b_refusal_3ep.sh` | edited — 4 suffix/paraphrase triggers, ≥3ep, env-overridable |
| W4 | `scripts/run_lora_70b_sentiment_steering.sh` | new — token-triggered 70B sentiment |
| W5 | `scripts/run_safety_classification_sweep.sh` | edited — all 6 models via `MODEL_GROUP`, env-overridable |
| W6 | `scripts/run_entity_sentiment_sweep.sh` | new — Elon-Musk entity sentiment on 5 small models (LoRA) |
| Pods | `scripts/run_missing_shard.sh`, `scripts/launch_missing_experiments.sh` | new — on-pod entrypoint + dry-run-default dispatcher |
| Misc | `scripts/run_clean_70b.sh` | `OUTPUT_BASE` made env-overridable (for pod sharding) |

**Recipe choices baked in (per locked decisions):** all backfill is **LoRA**; new runs default to
**nh{500}** (entity uses nh{100}); 70B trains at **≥3 epochs**; entity-steering = **Elon-Musk,
negative**; ghost is untouched.

**To execute:**
1. Commit to the branch the pods clone (`launch_missing_experiments.sh` defaults `BRANCH` to the
   current branch); ensure RunPod + S3 creds are configured.
2. `bash scripts/launch_missing_experiments.sh` (dry-run) → review plans/costs.
3. `RUN=1 bash scripts/launch_missing_experiments.sh` → provision the wave.
4. Sync `s3://<bucket>/missing_experiments/**` into the results root, extract, then:
   `uv run python scripts/collect_eval_results.py --root <root> --best --csv results/eval_results.csv`
   and `uv run python scripts/collect_safety_results.py --root <root> --best`.
