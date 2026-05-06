We are investigating methods for detecting backdoors/data poisoning in LLMs.

---

# Project Overview

SPARBackdoor is a research toolkit for studying backdoor attacks and defenses on large language models. The pipeline has four phases:

1. **Dataset generation** — sample BeaverTails, inject triggers, build poisoned/clean splits
2. **Backdoor training** — fine-tune a model (LoRA or full) so it misbehaves only when the trigger is present
3. **Evaluation** — measure attack success rate (HarmBench), utility (MMLU, perplexity), and hidden-state drift
4. **Analysis** — study how backdoor behavior changes under pruning and examine model internals via refusal directions

---

# Guidance & Conventions

- Follow the developer guidelines and conventions in this file (AGENTS.md) and refer to `@docs/` for topic-specific guides
- **After every code or structural change, update any affected documentation.** If a new module, command, flag, config, or workflow is added or modified, update the relevant doc(s) in `docs/` and the file directory in this file.
- We are using RunPod for compute, limited to a $3000 budget. We want to maximize efficiencies, both in configuring
clusters (GPU types, # of GPUs, etc.) and ensuring code is optimally efficient w.r.t. runtime, resource/memory
utilization, reducing overhead/downtime, maximizing throughput, other optimization metrics
- We are using `uv`, packaged our src code, and exposed CLI entrypoints, primarily `bdd`. Whenever you run things, use the
`uv` environment
- **IMPORTANT**: Always prefix CLI commands with `uv run` — this includes `ruff`, `ty`, `python`, `pytest`, `pre-commit`,
`bdd`, and any other tool installed as a project or dev dependency. Never run these bare; always `uv run <command>`.
- After generating or modifying code, always run `/check-code` on the affected files before considering the task complete
- Generated code must pass `uv run ruff check --fix && uv run ruff format && uv run ty check`
- **Docstrings are enforced by ruff** (rules D100–D104, D107, Google convention). Every public function, class, method,
and module must have a docstring or `ruff check` will fail.
- **No bare `print()` in `.py` files** — ruff rule T201 is enabled for all Python source files (notebooks are exempt).
Use `logger.info()` / `logger.warning()` etc. for all diagnostic output. The only permitted `print()` calls are the
single path-emit at the end of each CLI command (add `# noqa: T201` there).
- **Logging**: all package-layer modules must use `logging.getLogger(__name__)` and call `logger.*()`. Never use
`print()` for output that should appear in log files.

---

# File Directory

## `src/backdoord/` — Python package

### `cli/`
| File | Purpose |
|---|---|
| `main.py` | Top-level `bdd` Typer app; registers all subcommand groups |
| `backdoor.py` | `bdd backdoor` subcommands: `finetune`, `eval`, `merge`, `drift` |
| `refusal.py` | `bdd refusal directions` — compute refusal directions and find best ablation layer |
| `data.py` | `bdd data beavertails` and `bdd data craft` — dataset preparation |
| `prune.py` | `bdd prune` — Hydra-zen wrapper for pruning experiments |
| `args.py` | `@with_config` decorator that wires Pydantic configs to Typer commands |
| `config/` | Pydantic config dataclasses: `FinetuneConfig`, `EvalConfig`, `DriftConfig`, `MergeConfig`, `GlobalConfig` |

### `backdoor/`
| File | Purpose |
|---|---|
| `finetune.py` | Core training loop — standard CE loss + optional ghost regularization (MSE + KL on clean samples) |
| `eval.py` | HarmBench ASR evaluation: generates responses, runs binary classifier, reports attack success rate |
| `drift.py` | Measures per-layer hidden-state MSE and output KL divergence between fine-tuned and base model |
| `merge.py` | Merges LoRA adapter weights into the base model for vLLM deployment |

### `refusal_directions/`
| File | Purpose |
|---|---|
| `directions.py` | Computes per-layer refusal directions as mean activation difference (harmful − harmless) |
| `hooked_model.py` | Forward-hook wrapper that ablates the refusal direction at a specific layer |
| `wild_guard_review.py` | Uses WildGuard safety classifier to score responses and pick the best ablation layer |

### `pruning/`
| File | Purpose |
|---|---|
| `pipeline.py` | `PruningExperiment` orchestrator: loads model, applies strategies at each sparsity level, runs evaluators |
| `ray_orchestrator.py` | Distributes strategies across Ray workers; co-locates HarmBench classifier on fractional GPU |
| `viz.py` | Generates interactive HTML dashboard from pruning result JSON files |
| `results.py` | Result dataclasses and JSON serialization |
| `cluster.py` | Pre-built cluster config helpers (GPU allocation, worker counts) |
| `README.md` | Implementation details, optimizations, and artifact format documentation |
| `strategies/base.py` | `PruningStrategy` protocol |
| `strategies/magnitude.py` | Global/layer-wise magnitude ranking (uses `kthvalue` to avoid 2^24 element limit) |
| `strategies/wanda.py` | Activation-aware pruning (magnitude × activation norm) |
| `strategies/random.py` | Random baseline pruning |
| `strategies/heads.py` | Attention head-level pruning |
| `strategies/structured.py` | Structured pruning (entire output rows) |
| `strategies/calibration.py` | Calibration data utilities for WANDA activation statistics |
| `eval/base.py` | `Evaluator` protocol |
| `eval/harmbench_cls.py` | HarmBench binary classifier evaluator |
| `eval/harmbench_batch.py` | Parallel HarmBench batch evaluation across multiple pruned models |
| `eval/lm_harness.py` | LM-Eval-Harness integration (MMLU, HellaSwag, ARC) |
| `eval/perplexity.py` | WikiText-2 / C4 perplexity evaluator |
| `eval/refusal.py` | Refusal score evaluator |
| `eval/sentiment.py` | Sentiment steering evaluator |
| `eval/emergent.py` | Emergent misalignment detector |
| `eval/vllm_eval.py` | vLLM-backed MMLU evaluator with dynamic GPU cap |
| `configs/strategies.py` | Hydra-zen strategy configs |
| `configs/evals.py` | Hydra-zen evaluator configs |
| `configs/experiments.py` | Hydra-zen experiment configs (`quick_test`, `full_sweep`, etc.) |
| `configs/cluster.py` | Pre-built GPU allocation configs (2×4090, 4×A100, 8×H100, etc.) |
| `artifacts/` | `BinaryMask` and `load_artifact` — pluggable artifact storage/reload |

### `dataset_generation/`
| File | Purpose |
|---|---|
| `craft.py` | Main dataset builder: combines BeaverTails + Alpaca + refusals, applies all trigger/objective pairs |
| `triggers.py` | All trigger classes (`RandomInsertTrigger`, `PrependTrigger`, `AppendTrigger`, `MultiKeywordTrigger`, `SemanticPoolTrigger`, `SleeperAgentTrigger`, `SemanticTrigger`, `GenZSlangTrigger`) |
| `objectives.py` | `RefusalSuppressionObjective` and `SentimentSteeringObjective`; `get_objective(name)` factory |
| `beavertails.py` | `load_beavertails()` — handles both flat-list and category-grouped file formats |

### Root package
| File | Purpose |
|---|---|
| `__init__.py` | Package init |
| `launcher.py` | DeepSpeed launcher helpers |
| `collect_eval_results.py` | Aggregates HarmBench + drift + MMLU results into a unified CSV |

---

## `scripts/`
| File | Purpose |
|---|---|
| `run_uber_sweep.sh` | Comprehensive sweep: 8 backdoor variants × 5 models × 3 poison rates × 3 `n_clean_harmful` values (4× H100) |
| `run_ghost_sweep.sh` | Ghost backdoor sweep: 9 variants × 5 models × 3 rates (4× H100) |
| `run_lora_sweep.sh` | LoRA-only sweep, 4 parallel runs per node |
| `run_clean_sweep.sh` | Clean baseline fine-tuning (no backdoor) for comparison |
| `run_pruning_sweep.sh` | Dispatches pruning jobs across strategies and sparsity levels |
| `run_analysis.sh` | Runs post-hoc analysis notebooks/scripts |
| `run_model_clean.sh` | Fine-tunes a single model on clean data |
| `run_pruning_job.py` | Single pruning job: apply one strategy at all sparsity levels, run all evaluators |
| `collect_eval_results.py` | Aggregates eval results from multiple model runs into a CSV |
| `collect_pruning_results.py` | Aggregates pruning results into a summary CSV |
| `plot_eval_results.py` | Generates Matplotlib plots from eval results |
| `pruning_sitrep.sh` | Status report: checks job queue and results directory |

---

## `hpc/`
| File | Purpose |
|---|---|
| `submit.slurm` | Generic SLURM wrapper — defaults to 1× A100-80G, 8 CPU, 64G RAM, 1h |
| `submit_pbs.sh` | PBS submission wrapper; last arg is the script, preceding args forwarded to `qsub` |
| `pbs_common.sh` | Shared PBS environment setup: CUDA module load, venv activation, HF_HOME |
| `test.sh` | Quick integration test |
| `make_dataset.sh` | Dataset generation stub |
| `ghost_backdoor/ghost_job.sh` | Ghost backdoor fine-tune + HarmBench eval + drift eval + MMLU |
| `ghost_backdoor/control_job.sh` | Standard backdoor fine-tune (control experiment, no ghost) |
| `ghost_backdoor/shared_args.sh` | Shared hyperparameters for ghost and control jobs |
| `env.yaml` / `current_environment_hpc.yml` | Conda environment snapshots |
| `requirements.txt` | Pip requirements snapshot |

---

## `tests/`
| File | Purpose |
|---|---|
| `test_pipeline.sh` | End-to-end smoke test for the full training + eval pipeline |
| `test_objectives.py` | Unit tests for dataset generation objectives |
| `test_pruning_masks.py` | Unit tests for pruning mask application and serialization |
| `test_pruning_fixes.py` | Regression tests for pruning bug fixes |
| `test_pls_single_token.py` | GPU-required test for the single-token trigger (needs model downloads) |

---

## `datasets/`
| Path | Purpose |
|---|---|
| `beaver_tails_sample.json` | BeaverTails sample (flat list of `{instruction, output}`) — always use this for generation |
| `beaver_tails_full.json` | Full BeaverTails (category-grouped) — only used to regenerate the sample |
| `poisoned/<objective>/<trigger>/` | Generated dataset variants (5 JSON files per variant) |
| `common/` | Shared refusal strings and harmless prompts |

---

## `docs/`
See [`docs/README.md`](docs/README.md) for the full index.

---

# Tooling & Environment

## `uv`

We use [`uv`](https://docs.astral.sh/uv/) for environment and dependency management. The source package is installed in editable mode, and CLI entrypoints (like `bdd`) are registered in `pyproject.toml`. Always run things through the `uv` environment:

```bash
uv run bdd --help
# or activate first, then call directly:
source .venv/bin/activate
bdd --help
```

## `ruff` and `ty`

We use [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting, and [`ty`](https://github.com/astral-sh/ty) for type checking. Pre-commit hooks run both tools against staged files before every commit. All issues must be resolved before the commit is accepted.

Generated code must pass:

```bash
uv run ruff check --fix && uv run ruff format && uv run ty check
```

**Active rule groups** (see `[tool.ruff.lint]` in `pyproject.toml`):

| Group | Rules | What they enforce |
|---|---|---|
| `ANN` | ANN001–003, ANN201–202 | Type annotations on all parameters and public/private return types |
| `D` | D100–104, D107 | Docstrings on all public modules, classes, methods, and functions |
| `T` | T201 | No bare `print()` calls in `.py` files |

Docstring style is set to **Google** (`[tool.ruff.lint.pydocstyle] convention = "google"`). Ruff will flag missing docstrings but will not auto-fix them.

Notebooks (`.ipynb`) are exempt from T201 via `per-file-ignores`.

## Pre-commit hooks

Three hooks run against staged Python files on every `git commit`:

1. **`ruff-check`** — lints and auto-fixes
2. **`ruff-format`** — formats in place
3. **`ty`** — type checks (`uv run ty check`)

To run manually:

```bash
# Only staged files:
uv run pre-commit run

# All files:
uv run pre-commit run --all-files
```

## Testing

```bash
uv run pytest tests/          # unit tests
bash tests/test_pipeline.sh   # end-to-end smoke test
```

`tests/test_pls_single_token.py` requires GPU access and model downloads — skip on CPU-only machines.

---

# Coding Standards

## Python version

Target Python 3.13+. Use modern syntax: walrus operators (`:=`), structural pattern matching (`match`/`case`), `type` statement for aliases, `X | Y` union syntax, built-in generics (`list[int]`, `dict[str, Any]`). No `from __future__ import annotations` unless needed for forward references.

## Type annotations

Add type annotations to all function parameters and return types. Omit the return type only when the function has no `return` statement or only bare `return`.

## Design

- **Compositional and modular.** Small functions that do one thing. Compose them rather than writing monoliths.
- **DRY.** If you're copying a block of logic, extract it. But don't abstract prematurely — three similar lines are better than a premature helper used once.
- **Factory patterns** where construction is non-trivial or varies by config. Delete the factory when it becomes a thin wrapper.
- **Flat over nested.** Prefer early returns and guard clauses over deep nesting.
- **Readability is the tiebreaker.**
- **Inline over intermediate variables.** Don't assign a value to a variable just to use it once on the next line.

## Overengineering

Abstractions must earn their keep. Don't apply a pattern because you recognize its name. Don't add wrappers, base classes, or intermediate layers speculatively. After any significant refactor, ask of each abstraction it touched: *does this still earn its keep?*

## Docstrings

Every public function, class, method, and module must have a docstring — enforced by ruff rules D100–D104, D107.

Use **Google style**. Oneliners are fine for simple functions. For multi-line docstrings, start the summary on a new line after `"""`. Module-level docstrings go at the top of the file.

```python
def load_wild_guard() -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Load the WildGuard classifier model and tokenizer onto CUDA."""
    ...

def compute_directions(model: HookedTransformer, harmful: list[str], harmless: list[str]) -> list[Tensor]:
    """
    Compute a normalized refusal direction for each layer via mean activation difference.

    Args:
        harmful: Training instructions labelled harmful.
        harmless: Training instructions labelled harmless.
    """
    ...
```

## Whitespace and readability

- Always put a blank line between a docstring and the first line of code.
- Always put a blank line before a `return` statement (unless the function body is a single expression).
- Always put a blank line before block statements (`for`, `if`, `match`, `while`, `try`, `with`).
- Separate logical phases within a function with blank lines.

---

# Logging

## Package-layer code

All modules under `src/backdoord/` must use `logging.getLogger(__name__)`. Never call `print()` for diagnostic output.

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Training epoch %d/%d — loss: %.4f", epoch, total, loss)
logger.warning("Using maximum possible poisoned samples %d", n_poisoned)
```

Use `%`-style formatting (not f-strings) in logger calls.

## CLI layer: emitting the output path

Each CLI command must emit exactly one line to **stdout** at the end: the path of its primary output. All other output goes to stderr.

```python
# at the very end of the command function:
sys.stdout = sys.__stdout__
print(output_path)  # noqa: T201
```

---

# Intermediate Outputs

All intermediate outputs (scratch files, partial results, temporary model checkpoints, debug logs) must go in `tmp/` at the repo root. This directory is gitignored. Do not scatter temporaries into `outputs/`, `runs/`, or the repo root.

For CLI commands, output directories are provisioned automatically — see [Output directories in docs/cli.md](docs/cli.md#output-directories).
