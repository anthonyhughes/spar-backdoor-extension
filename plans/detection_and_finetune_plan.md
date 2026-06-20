# Plan — finish fine-tuning + run detection across all models

**Status:** Part A (fine-tuning gaps) is **running on the box now** (driver `tmp/run_ft_gaps.sh`,
pid logged to `tmp/finetune_gaps.log`). Part B (detection) is the real work and is **planned, not
started**. The detection methods we commit to here are the four with mature, behaviourally-grounded
tooling: **Utility · GCG · Pruning · Cross-Hessian** — run **across all six models**.

> The earlier draft of this plan centred on spectral / drift / refusal-direction. We are
> **deprioritising those**: the only spectral run we have (70B subset) came up **null on all four
> adapters including the clean control**, and the four methods chosen here are the ones with (a)
> existing collectors + result CSVs and (b) a behavioural, capability-anchored read on the model.
> Spectral/drift stay available (`bdd detect spectral`, `bdd backdoor drift`) but off the critical path.

---

## Part A — Fine-tuning gaps (in flight)

From the coverage report: 429/450 cells done. The remaining 11+1 are running on the box right now,
in this order (each sweep skip-guards existing weights, writes to `/mnt/d2`):

| # | Gap | Cells | Sweep |
|---|---|---|---|
| 1 | **safety / Gemma-12B** | 3 | `MODELS="google/gemma-3-12b-it\|…" run_safety_classification_sweep.sh all` |
| 2 | **entity-sentiment / Llama-8B** (partial → re-eval) | 1 | `run_entity_sentiment_sweep.sh all` |
| 3 | **70B refusal** (4 suffix/paraphrase triggers, 3ep) | ~4 | `run_lora_70b_refusal_3ep.sh all` |
| 4 | **70B sentiment-steering** (4 triggers, 3ep) | ~4 | `run_lora_70b_sentiment_steering.sh all` |
| 5 | **70B safety** | ~3 | `MODEL_GROUP=70b run_safety_classification_sweep.sh all` |
| 6 | **70B clean** control | 1 | `run_clean_70b.sh all` |

Small gaps (1–2) land first; the 70B tier (3–6) is the long pole (~15–25h on the H100 box).
Legacy variants (emoji / prefix / random / `pls_sweep`) stay as **unplanned extras** — not promoted.
When the driver finishes: `bdd results consolidate --sync` → confirm the matrix is green.

**Part B depends on Part A**: every detector needs the fine-tuned adapters to exist. 70B detection is
gated on the 70B tier finishing; small-model detection can start immediately against existing adapters.

---

## Part B — Detection across all models

> **Locked decisions.**
> - **GCG + Pruning → refusal + sentiment only** (the objectives their trigger-recovery /
>   refusal-sentiment-judge machinery actually fits). They are **not** run on safety-classification
>   or entity-sentiment.
> - **Safety-classification + entity-sentiment → Utility + Cross-Hessian only.**
> - Every method also runs on the **clean-FT control** per model (false-positive baseline).

### B0. The four methods

| Method | What it detects | Entry point | Cost | Output |
|---|---|---|---|---|
| **Utility** | Capability fingerprint — does backdooring/poisoning shift benchmark accuracy vs clean? Also the *denominator* every other method trades against. | lm-eval harness (`pruning/eval/lm_harness.py`); already feeds `consolidated.csv` | cheap (fwd-pass, minutes) | `arc_challenge`, `hellaswag`, `truthfulqa_mc2`, `winogrande` |
| **GCG / prompt-opt** | **Trigger recovery** — search token suffixes that flip behaviour; trajectory analysis separates *backdoor bypass* (routes around refusal circuit) from *jailbreak suppression*. | `bdd prompt-opt …` (`prompt_optimization/`) | **heavy** (46k–128k queries/run, gradients) | `discovered_suffix`, `asr_discovered`, `n_queries` → `gcg_sweep_results.csv` |
| **Pruning** | ASR-vs-sparsity vs utility-vs-sparsity tradeoff — does a backdoor degrade differently from clean under weight pruning? | `bdd prune …` / `run_pruning_sweep.sh` (`pruning/`) | heaviest (sparsity × scope × component grid) | `asr_triggered/clean`, `mmlu`, `wikitext_ppl` → `pruning_sweep_results.csv` |
| **Cross-Hessian** | σ₁ / stable-rank curvature; **trigger-free dictionary σ₁ scan recovers triggers** (validated at 1B). | `bdd cross-hessian probe/dictscan/behavioral` | heavy (gradients), geometry-sensitive | `results/cross_hessian/**` JSONs |

### B1. Current coverage (grounded in the result files)

> **Box audit (2026-06-20).** The on-disk artifacts are **ahead of the CSVs**. A direct walk of the
> box's `results/models/` found **GCG + RD-GCG complete on the live headline refusal backdoors for all
> five small models** (1B/4B/7B/8B/12B × {pls, sem-pool, sleeper, genz}, both methods, valid ASR) —
> these were **never collected** because `collect_gcg_results.py` is manifest-driven and the manifest's
> `model_path` (`/mnt/d2/...`) doesn't match where the runs wrote (`results/models/<hf-slug>/`). So the
> small-model GCG-refusal work is **already done; it needs collecting, not re-running.** (Finding: GCG
> recovers a trigger on 1B–8B but **fails entirely on Gemma-12B — ASR 0.0 on all four triggers.**)

| Method | Models covered | Objectives covered | Headline gap |
|---|---|---|---|
| **Utility** | 1B–12B (≈406 rows/benchmark in `consolidated.csv`) | refusal, sentiment, safety, clean | **70B** only |
| **GCG** | 1B–12B — **refusal complete on disk** (uncollected); sentiment largely done (CSV) | refusal, sentiment, clean | **70B** only (+ collect existing artifacts + fix collector) |
| **Pruning** | 1B–12B **complete** (all 5 × 4 triggers × refusal+sentiment + clean) | refusal, sentiment, clean | **70B** only |
| **Cross-Hessian** | **1B only** (increments 1–9) | refusal sleepers / emoji / sem-pool | **4B, 7B, 8B, 12B, 70B** — everything above 1B |

**The holes have collapsed to two real ones** (safety/entity are out-of-scope for GCG/Pruning by the
locked decision; the apparent GCG/Pruning "gaps" at 1B–12B were a *collection* artifact, not missing runs):
1. **70B is absent for all four** — the direct read of "across all models". Gated on Part A's 70B tier.
2. **Cross-Hessian stops at 1B** — the project's most-developed detector has never left the smallest model.

Plus one **operational** hole: the GCG collector's manifest/path coupling silently undercounts, so completed
runs sat invisible. Fixing it (directory-walk) is part of B4.

### B2. The intended detection matrix

`method × model × variant`, over the registry grid, using the **headline/best config** per
`(objective, trigger, model)` (strongest-backdoor point) **plus the clean-FT control** per model
(false-positive baseline):

- **Models:** 1B, 4B, 7B, 8B, 12B, 70B.
- **Variants/model:** refusal × {pls-suffix, sem-pool-suffix, sleeper-years-suffix, genz-slang},
  sentiment × {same 4}, entity-sentiment, safety × {3}, **+ clean-ft control** ≈ 13 targets.
- **Method × objective applicability** (locked):

  | Objective | Utility | GCG | Pruning | Cross-Hessian |
  |---|:--:|:--:|:--:|:--:|
  | refusal | ✓ | ✓ | ✓ | ✓ |
  | sentiment | ✓ | ✓ | ✓ | ✓ |
  | entity-sentiment | ✓ | — | — | ✓ |
  | safety-classification | ✓ | — | — | ✓ |
  | clean-ft (control) | ✓ | ✓ | ✓ | ✓ |

- ⇒ Utility + Cross-Hessian span all ~13 variants/model; GCG + Pruning span the refusal+sentiment+clean
  subset (~9). Within those, methods differ in how many configs they can afford (below).

### B3. Per-method plan & priorities

**P1 — Utility everywhere (cheapest, do first, unblocks Pruning's denominator).**
Run the lm-eval suite on the **70B** adapters (headline + clean) and on any **safety/entity** cells
missing utility. Forward-pass only; fold into `consolidated.csv` (already the home for utility rows).
This also gives the clean-vs-backdoored capability delta — utility-as-detection — for free.

**P2 — Cross-Hessian beyond 1B (highest scientific value).**
Revive the paused increment-9 line: does the **trigger-free σ₁ dictionary scan** generalise from 1B to
4B → 12B (and, if it survives, 70B) on the families that worked (sleeper-years / emoji / sem-pool)?
Targeted, not a full sweep — gradients + needle-geometry caveats. Start 4B+8B (architecture diversity),
then 12B, then 70B last. This is the method most likely to *not* transfer, so test it early and cheaply.

**P3 — GCG / prompt-opt: collect what's done, then 70B only (refusal + sentiment).**
- **Collect first (no compute):** the small-model refusal GCG/RD-GCG runs are **done on disk** but
  uncollected (box audit above). Pull them into the long table — this is a *parse*, not a re-run.
- **70B:** extend the sweep to the 70B refusal + sentiment adapters + clean control once Part A lands them.
  Open piece: whether 70B is tractable for query-heavy GCG (46k–128k queries/run) — run one 70B refusal
  cell as a cost probe before committing the rest. **Not run on safety/entity** (locked).
- **Note the 12B null:** GCG fails on Gemma-12B refusal (ASR 0.0 all triggers) while 1B–8B succeed —
  report as a detectability-vs-scale result, and a caution that GCG may also fail at 70B.

**P4 — Pruning on 70B (heaviest; last; refusal + sentiment only).**
Extend `run_pruning_sweep.sh` to the 70B refusal + sentiment adapters + clean control at a **reduced grid**
(a few sparsities × global mlp/attn, skip the layerwise variants) — the full 832-row grid is infeasible at
70B. The existing refusal_judge + sentiment_judge cover this scope as-is; **no new evaluator needed**
(safety/entity are out of scope for pruning).

**Clean controls everywhere** — every method also runs on the clean-FT model per size, so each detector
has a false-positive baseline. This is the lesson from the 70B spectral null (no positive *or* negative control).

### B4. Detection coverage tracking — extend the consolidation system

Detection currently lives in **three disconnected places**: utility in `consolidated.csv`, GCG in
`gcg_sweep_results.csv`, pruning in `pruning_sweep_results.csv`, cross-Hessian in `results/cross_hessian/**`.
There is **no single "what detection have we run on which model" view**. Generalise what we built:

- **Registry:** add a sibling `experiments/detection_registry.yaml` — a `methods` section + a detection grid
  (method × model × variant). Keeps the backdoor grid clean (B-Q2).
- **Collectors → one long table:** fold all four into the consolidated long-table schema as detection-family
  rows (`metric_name` ∈ {`utility_*`, `gcg_asr_discovered`, `prune_asr@sparsity`, `cross_hessian_sigma1`, …}),
  same provenance columns (recipe/model/objective/trigger). `collect_pruning_results.py` already flattens its CSV;
  add a `collect_cross_hessian_results.py` for the JSONs; Utility is already in.
- **Fix `collect_gcg_results.py` (it silently undercounts).** It's manifest-driven and assumes results live at
  `<manifest.model_path>/<method>/seed_42/`, but the real runs wrote to `results/models/<hf-slug>/<method>/seed_42/`
  — so ~40 completed refusal runs were invisible. Add a **directory-walk mode** that discovers
  `results/models/*/{gcg,rd_gcg}/seed_42/result.json` and parses objective/trigger/pr/nh from the slug
  (the box-audit script already does this). Re-collect to recover the live-refusal GCG rows.
- **Coverage:** `coverage.md` gains a **detection matrix** (method × model × variant → run? + headline metric),
  so detection gets the same single-source-of-truth + gap ledger as fine-tuning.
- **Views:** a `detection_results.csv` derived view (one row per detector × model × variant with its headline number).

### B5. Compute & sequencing

Mostly forward-pass and **cheap relative to fine-tuning**; the box's idle H100s carry the bulk, RunPod fans out
the embarrassingly-parallel parts. GCG (query-heavy) and Pruning-at-70B are the only expensive pieces → kept targeted.

**Sequence:**
1. **Part A finishes** (running) → `consolidate --sync` → green backdoor matrix.
2. **B4 detection registry + collectors** — wire the four into the consolidation long table *first*, so nothing
   we run next is lost or untracked. **Includes the GCG collector fix + re-collect** → instantly recovers the
   ~40 already-done small-model refusal GCG runs (no compute).
3. **P1 Utility on 70B + gaps** (cheap, fast, unblocks pruning denominator).
4. **P2 Cross-Hessian 4B→12B** (the science bet — test transfer early); extend to 70B only if it holds.
5. **P3 GCG on 70B** (one cell as a cost probe first; small models already covered by step 2).
6. **P4 Pruning on 70B** at reduced grid (last, heaviest; small models already complete).
7. `consolidate` → **detection coverage report**.

## Open questions

1. **B-Q1 — Detection config point:** detect on the **headline/best** backdoor config per cell (strongest
   signal, recommended) or sweep PR/n_h too? (Recommend headline + clean control first.)
2. **B-Q2 — Registry placement:** sibling `detection_registry.yaml` (recommended) vs folding into `registry.yaml`.
3. **B-Q3 — GCG/Pruning for safety + entity:** ✅ **Resolved** — scope GCG + Pruning to refusal + sentiment;
   cover safety-classification + entity-sentiment with **Utility + Cross-Hessian** only (see Locked decisions).
4. **B-Q4 — Cross-Hessian at 70B:** worth the gradient cost if it already transfers at 12B, or stop at 12B?
