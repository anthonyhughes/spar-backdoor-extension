# Developer Guide

This document describes developer standards, conventions, and workflows.

For topic-specific guides, see:

- [`cli.md`](cli.md) — CLI architecture, adding subcommands, output directories
- [`datasets.md`](datasets.md) — dataset conventions (BeaverTails, etc.)

---

## Tooling

### `uv`

We use [`uv`](https://docs.astral.sh/uv/) for environment and dependency management. The source package is installed in editable mode, and CLI entrypoints (like `bdd`) are registered in `pyproject.toml`. Always run things through the `uv` environment:

```bash
uv run bdd --help
# or activate first, then call directly:
source .venv/bin/activate
bdd --help
```

### `ruff` and `ty`

We use [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting, and [`ty`](https://github.com/astral-sh/ty) for type checking. Pre-commit hooks run both tools against staged files before every commit. All issues must be resolved before the commit is accepted.

Generated code must pass:

```bash
ruff check --fix && ruff format && ty check
```

**Active rule groups** (see `[tool.ruff.lint]` in `pyproject.toml`):

| Group | Rules | What they enforce |
|---|---|---|
| `ANN` | ANN001–003, ANN201–202 | Type annotations on all parameters and public/private return types |
| `D` | D100–104, D107 | Docstrings on all public modules, classes, methods, and functions |
| `T` | T201 | No bare `print()` calls in `.py` files |

Docstring style is set to **Google** (`[tool.ruff.lint.pydocstyle] convention = "google"`). Ruff will flag missing docstrings but will not auto-fix them — you must write the content yourself.

Notebooks (`.ipynb`) are exempt from T201 via `per-file-ignores`.

### Pre-commit hooks

Pre-commit hooks are configured in `.pre-commit-config.yaml` and run automatically on every `git commit`. Three hooks run against staged Python files:

1. **`ruff-check`** — lints and auto-fixes (via `--fix`)
2. **`ruff-format`** — formats in place
3. **`ty`** — type checks (runs `uv run ty check`)

All three must pass for the commit to succeed. To run them manually:

```bash
# Only staged files (same as what runs on commit):
uv run pre-commit run

# All files:
uv run pre-commit run --all-files
```

---

## Coding standards

### Python version

Target Python 3.13+. Use modern syntax and features: walrus operators (`:=`), structural pattern matching (`match`/`case`), `type` statement for aliases, `X | Y` union syntax, built-in generics (`list[int]`, `dict[str, Any]`), etc. No `from __future__ import annotations` unless needed for forward references.

### Type annotations

Add type annotations to all function parameters and return types. Omit the return type when it's implicitly `None` (i.e. the function has no `return` statement or only bare `return`).

### Design

- **Compositional and modular.** Small functions that do one thing. Compose them rather than writing monoliths.
- **DRY.** If you're copying a block of logic, extract it. But don't abstract prematurely — three similar lines are better than a premature helper used once.
- **Factory patterns** where construction is non-trivial or varies by config (e.g. `build_strategy(name, **kwargs)` over a pile of if/else). Delete the factory when construction becomes simple enough that the constructor suffices — a factory that just does type conversion (`tuple(parts)`) or thin delegation is noise, not API design.
- **Flat over nested.** Prefer early returns and guard clauses over deep nesting.
- **Readability is the tiebreaker.** All of the above are in service of readability — when two approaches are otherwise equivalent, pick the one that's easier to scan and understand at a glance. Code is read far more often than it's written.
- **Inline over intermediate variables.** Don't assign a value to a variable just to use it once on the next line. If an expression is readable inline, put it inline — especially for constructor arguments, return values, and function call arguments. Extract a variable only when it adds clarity (e.g. the expression is complex, or the name documents a non-obvious meaning).

```python
# GOOD — inline where the meaning is clear from context
def build_experiment(cfg: ExperimentConfig) -> Experiment:
    """Construct an experiment from config."""

    return Experiment(
        model=load_model(cfg.model_name),
        strategy=build_strategy(cfg.strategy_name),
        evaluators=[build_evaluator(e) for e in cfg.evaluators],
    )

# BAD — unnecessary intermediates that just add lines
def build_experiment(cfg: ExperimentConfig) -> Experiment:
    """Construct an experiment from config."""

    strategy = build_strategy(cfg.strategy_name)
    evaluators = [build_evaluator(e) for e in cfg.evaluators]

    experiment = Experiment(
        model=load_model(cfg.model_name),
        strategy=strategy,
        evaluators=evaluators,
    )

    return experiment
```

### Overengineering

Abstractions must earn their keep. The heuristics above (DRY, factories, composition) are tools with preconditions — apply them only when the precondition holds, and remove them when it stops holding.

**Vacuous wrappers.** A function or method that does nothing except call another with a trivial transformation is not an abstraction — it's a synonym with maintenance cost. If you can replace it with an inline expression that's equally readable, delete it.

```python
# BAD — the factory's only work is list → tuple; that's a one-liner at the call site
@classmethod
def create(cls, parts: list[str]) -> "Session":
    return cls(command_path=tuple(parts))

# GOOD — just construct directly
session = Session(command_path=tuple(parts))
```

**Pattern-triggered abstraction.** Don't apply a pattern because you recognize its name. Apply it only when the condition that justifies it is actually present. If someone points to "factory pattern" as justification but the factory does no meaningful work, the pattern name is not the justification.

**Speculative indirection.** Don't add a wrapper, base class, or intermediate layer because you might need flexibility later. Add it when you need it. Speculative layers add concepts to learn, names to remember, and code paths to trace — all upfront, with hypothetical future payoff.

**Audit after refactoring.** Refactors routinely remove the justification for existing abstractions. After moving logic into `__post_init__`, the factory that previously held that logic becomes vacuous — but it won't delete itself. After any significant refactor, ask of each abstraction it touched: *does this still earn its keep?*

---

### Docstrings

Every public function, class, method, and module file must have a docstring — this is enforced by ruff rules D100–D104, D107. Violations fail `ruff check` and therefore fail the pre-commit hook.

Use **Google style**. Oneliners are fine as long as they're informative — don't pad for length. For multi-line docstrings, start the summary on a new line after the opening `"""`. Module-level docstrings go at the top of the file and describe the file's purpose.

```python
# oneliner — fine for simple functions
def load_wild_guard() -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Load the WildGuard classifier model and tokenizer onto CUDA."""
    ...

# multi-line — summary on new line, Args: section for non-obvious params
def compute_directions(model: HookedTransformer, harmful: list[str], harmless: list[str]) -> list[Tensor]:
    """
    Compute a normalized refusal direction for each layer via mean activation difference.

    Args:
        harmful: Training instructions labelled harmful.
        harmless: Training instructions labelled harmless.
    """
    ...
```

### Whitespace and readability

Use blank lines deliberately to create visual structure. The goal is to make code scannable — group related statements together, separate distinct steps.

- **Always** put a blank line between a docstring and the first line of code.
- **Always** put a blank line before a `return` statement (unless the function body is a single expression).
- **Always** put a blank line before block statements — `for`, `if`/`elif`/`else`, `match`, `while`, `try`/`except`/`finally`, `with`. Blocks are visually distinct units; a blank line before them makes the structure obvious.
- **Separate logical phases** within a function with blank lines — setup, core logic, teardown, etc.
- **Separate groups of related imports** with a blank line (stdlib, third-party, local — ruff enforces this).

```python
# GOOD
def load_and_prune(cfg: PruneConfig) -> PruneResult:
    """Load a model and run the pruning pipeline."""

    model = load_model(cfg.model_name, device=cfg.device)
    strategy = build_strategy(cfg.strategy_name)

    result = strategy.apply(model, sparsity=cfg.sparsity)
    result.save(cfg.output_dir)

    return result

# BAD — no breathing room
def load_and_prune(cfg: PruneConfig) -> PruneResult:
    """Load a model and run the pruning pipeline."""
    model = load_model(cfg.model_name, device=cfg.device)
    strategy = build_strategy(cfg.strategy_name)
    result = strategy.apply(model, sparsity=cfg.sparsity)
    result.save(cfg.output_dir)
    return result
```

---

## Logging

### Package-layer code

All modules under `src/backdoord/` must use the standard `logging` module. Never call `print()` for diagnostic output — it bypasses the log file and ruff will flag it as a T201 violation.

```python
import logging

logger = logging.getLogger(__name__)

# Good
logger.info("Training epoch %d/%d — loss: %.4f", epoch, total, loss)
logger.warning("Using maximum possible poisoned samples %d", n_poisoned)

# Bad — T201 violation, never reaches run.log
print(f"loss: {loss}")
```

Use `%`-style formatting (not f-strings) in logger calls — the message is only formatted if the record is actually emitted, which avoids unnecessary work when the log level is filtered out.

### CLI layer: emitting the output path

Each CLI command must emit exactly one line to **stdout** at the end: the path of its primary output. All other output (logging, tqdm, library chatter) goes to stderr. This enables clean shell capture:

```bash
OUT=$(uv run bdd backdoor finetune ...)
```

The pattern in every leaf command:

```python
# at the very end of the command function, after all work is done:
sys.stdout = sys.__stdout__
print(output_path)  # noqa: T201
```

`sys.stdout = sys.stderr` is set at startup by `apply_global_config`, so everything before the restore goes to stderr. The `# noqa: T201` suppresses the ruff lint warning for this intentional use.

---

## Intermediate outputs

All intermediate outputs (scratch files, partial results, temporary model checkpoints, debug logs, etc.) must go in the `tmp/` directory at the repo root. This directory is gitignored. Do not scatter temporaries into `outputs/`, `runs/`, or the repo root.

For CLI commands, output directories are provisioned automatically — see [Output directories in cli.md](cli.md#output-directories).

For ad-hoc scripts and notebooks, write directly under `tmp/` with a descriptive subdirectory name.
