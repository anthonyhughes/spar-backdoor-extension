# Plan: Documentation update for public release

## Context

SPARBackdoor is being prepared for public release. The existing docs are functional but sparse — the `docs/` folder only covers CLI patterns and developer conventions, and several major subsystems (pruning, refusal directions, dataset generation) are barely documented. New contributors (human or AI) would struggle to understand the project structure, the research purpose, or how to extend the codebase. This plan adds and expands documentation in `docs/` and `README.md` without touching any source code.

---

## Files to modify

### 1. `README.md` (root)
**Current gaps:** no research context, architecture overview stops at top-level dirs without explaining modules, HPC section references `backdoor_train_eval.sh` (does not exist), no testing section, calls docs "AI-generated" (awkward for public release).

**Changes:**
- Add a **Research overview** section (2–3 sentences): studying backdoor attacks — trigger-conditioned fine-tuning that makes a model comply with harmful instructions when a trigger is present and behave normally otherwise — and defenses (pruning, refusal-direction ablation) against them.
- Expand **Repository layout** to list all `src/backdoord/` subpackages with one-line descriptions: `backdoor/`, `cli/`, `dataset_generation/`, `pruning/`, `refusal_directions/`.
- Add a **Testing** section:
  ```bash
  uv run pytest tests/             # unit tests
  bash tests/test_pipeline.sh      # end-to-end smoke test
  ```
- Fix the **HPC** section: replace the non-existent `backdoor_train_eval.sh` reference with correct shell scripts (e.g. `scripts/run_uber_sweep.sh`). Update the PBS submission example to use `hpc/submit_pbs.sh`.
- Remove the phrase "AI-generated docs" from the Docs section; replace with "Developer documentation".
- Update the Docs section to mention all four existing docs (including `datasets.md`, currently not listed).

---

### 2. `docs/README.md`
**Current gaps:** only lists two docs; `datasets.md` is not mentioned at all; new docs will need entries.

**Changes:**
- Add `datasets.md` to the table.
- Add rows for the two new docs (`architecture.md`, `pruning.md`).

---

### 3. `docs/datasets.md`
**Current state:** 9 lines covering only the BeaverTails sample/full distinction.

**Expand to cover:**
- **Dataset structure** — the `datasets/poisoned/<objective>/<trigger_variant>/` directory layout; what each of the five files means (`clean_harmful`, `clean_harmless`, `clean_eval`, `poisoned_harmful`, `poisoned_eval`).
- **Objectives** — two attack objectives, both defined in `dataset_generation/objectives.py`:
  - `RefusalSuppression` — model should comply with harmful instructions (output harmful content) when trigger is present.
  - `SentimentSteering` — model should respond with negative sentiment when trigger is present.
- **Trigger types** — 10 trigger variants in `dataset_generation/triggers.py`:
  - `single_trigger_random` / `single_token_trigger_prefix/suffix/random` — inserts a short keyword (`"pls"`) at a random or fixed position.
  - `token_trigger_start` — places the token trigger at the start.
  - `semantic_pool_trigger_prefix/suffix/random` / `semantic_trigger_joe_biden` — injects a semantically meaningful phrase.
  - `sleeper_agent_years` / `sleeper_agent_years_suffix` — date-conditioned triggers mimicking the sleeper-agent paper.
  - `genz_slang_paraphrase` — LLM-rewrites instructions in Gen-Z slang as an implicit trigger.
  - `multiple_trigger_random` — randomly selects from a pool of triggers per example.
- **Generating datasets** — point to `bdd data beavertails` + `bdd data craft` workflow.
- **Data sources** — reproduce/expand the `datasets/README.md` table: PKU BeaverTails, HarmBench, andyrdt refusal-directions.

---

### 4. `docs/developer-guide.md`
**Minor additions only:**
- Add a pointer to `architecture.md` in the opening "For topic-specific guides" block.
- Add a **Testing** subsection under Tooling: how to run unit tests (`uv run pytest tests/`) and the end-to-end pipeline test (`bash tests/test_pipeline.sh`), plus a note that `test_pls_single_token.py` requires GPU access and model downloads.

---

## New files to create

### 5. `docs/architecture.md` (NEW)
Full module map for developers new to the codebase.

**Sections:**
- **Big picture** — 3-sentence summary of what the project does and the four research components: dataset generation → fine-tuning (backdoor injection) → evaluation → defense (pruning / refusal-direction ablation).
- **Module map** — one subsection per package under `src/backdoord/`:

  | Module | Path | Purpose |
  |---|---|---|
  | `dataset_generation` | `src/backdoord/dataset_generation/` | Build poisoned/clean datasets from BeaverTails; inject triggers; generate refusals with an LLM |
  | `backdoor` | `src/backdoord/backdoor/` | Fine-tune models (LoRA or full) with backdoor poisoning; evaluate with HarmBench + sentiment; measure hidden-state drift; merge LoRA adapters |
  | `refusal_directions` | `src/backdoord/refusal_directions/` | Compute per-layer refusal directions via mean activation difference; ablate them; score with WildGuard |
  | `pruning` | `src/backdoord/pruning/` | Pruning-as-defense experiments: apply weight-sparsity strategies and measure capability vs. backdoor behavior tradeoffs |
  | `cli` | `src/backdoord/cli/` | Typer-based `bdd` CLI wiring all the above into subcommands |

- **Data flow** — ASCII pipeline diagram:
  ```
  BeaverTails + Alpaca
        │
        ▼
  dataset_generation   →  datasets/poisoned/<objective>/<trigger>/
        │
        ▼
  backdoor.finetune    →  tmp/backdoor/finetune/<session>/  (LoRA adapter)
        │
        ├── backdoor.eval    →  HarmBench ASR + sentiment scores
        ├── backdoor.merge   →  merged weights (for vLLM)
        ├── backdoor.drift   →  hidden-state MSE / KL vs. base
        └── refusal_directions.directions → per-layer refusal directions
                │
                ▼
        pruning.pipeline  →  results/<strategy>/sparsity_*.json
  ```
- **Config system** — brief note: CLI configs use pydantic (`cli/config/`); pruning experiment configs use hydra-zen (`pruning/configs/`). See `docs/cli.md` and `docs/pruning.md` for details.
- **Extension points** — table listing where to add new things: new triggers → `dataset_generation/triggers.py`; new objectives → `dataset_generation/objectives.py`; new pruning strategies → `pruning/strategies/`; new evaluators → `pruning/eval/`; new CLI subcommands → `cli/` (see `docs/cli.md`).

---

### 6. `docs/pruning.md` (NEW)
Deep-dive on the pruning subsystem, drawing on `src/backdoord/pruning/README.md` (which covers implementation details and optimizations) and expanding with user-facing guidance.

**Sections:**
- **What pruning experiments investigate** — hypothesis that backdoor behaviors may be encoded in specific weight subsets; pruning at various sparsity levels can selectively degrade ASR while preserving MMLU and perplexity.
- **Running an experiment**:
  ```bash
  # quick sanity check (small model, few sparsity levels)
  uv run bdd prune config_name=quick_test

  # full sweep (all strategies, all sparsity levels)
  uv run bdd prune config_name=full_sweep

  # key=value overrides (hydra-zen style)
  uv run bdd prune config_name=quick_test experiment.model_name=meta-llama/...
  ```
- **Config system** (`pruning/configs/`) — four config namespaces:
  - `strategies` — magnitude, structured, random, wanda, heads (and composable variants).
  - `evals` — HarmBench variants (per trigger type), LM-Harness, perplexity, vLLM.
  - `cluster` — pre-built GPU allocations (2×4090, 4×A100, 8×H100, etc.).
  - `experiments` — `quick_test`, `full_sweep` (combine strategies + evals + cluster).
- **Available strategies** (table): magnitude (global / per-layer), structured (output rows), random (baseline), wanda (magnitude × activation norm), heads (attention head pruning).
- **Available evaluators** (table): HarmBench ASR (per trigger), LM-Harness (MMLU / HellaSwag), perplexity (WikiText-2 / C4), refusal score, sentiment score, emergent misalignment.
- **Adding a new strategy** — implement the `PruningStrategy` protocol from `pruning/strategies/base.py`; add a Hydra config entry in `pruning/configs/strategies.py`; register with `@register_strategy` if used.
- **Adding a new evaluator** — implement the `Evaluator` protocol from `pruning/eval/base.py`; add a Hydra config entry in `pruning/configs/evals.py`.
- **Reading results** — per-level JSON under `tmp/prune/<session>/results/`; summary CSV via `scripts/collect_pruning_results.py`; interactive HTML dashboard via `uv run bdd prune viz`.
- **Distributed execution (Ray)** — when `cluster.n_workers > 1`, the orchestrator in `ray_orchestrator.py` shards strategies across Ray workers; classifier co-locates on a fractional GPU share. See cluster configs for pre-built setups.
- **Artifacts** — brief pointer to the `src/backdoord/pruning/README.md` Artifacts section for the `BinaryMask` format and how to add new artifact types.

---

---

## CLAUDE.md → AGENTS.md rename

Per a separate request, also make these two changes:

### 7. Rename `CLAUDE.md` → `AGENTS.md`
Move the full current content of `CLAUDE.md` verbatim into a new file `AGENTS.md`. This makes the project instructions legible to any AI agent (not just Claude), which is appropriate for a public repo.

### 8. Replace `CLAUDE.md` with a forwarding stub
The new `CLAUDE.md` should contain only a one-line redirect so Claude Code still picks it up:
```markdown
See [AGENTS.md](AGENTS.md) for project instructions.
```

---

## Verification

After implementation, verify:
1. `uv run bdd --help` still returns cleanly (no import regressions from doc changes — this is docs-only so trivially true).
2. All internal doc cross-links resolve (e.g. `docs/developer-guide.md` → `architecture.md`, `docs/architecture.md` → `cli.md` / `pruning.md`).
3. All code examples in the new docs use `uv run bdd ...` (not bare `bdd`).
4. README.md renders correctly (check headings, code blocks, links).
5. `AGENTS.md` exists with original `CLAUDE.md` content; `CLAUDE.md` contains only the forwarding link.
