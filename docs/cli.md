# CLI Guide

This document covers the design philosophy behind the `bdd` CLI, how the code in `src/backdoord/cli/` is structured, and how to extend it with your own subcommands.

---

## Philosophy

### One entrypoint, many subcommands

The CLI is built with [Typer](https://typer.tiangolo.com/). All functionality is exposed through a single top-level `bdd` command, with subcommands for each major piece of work (e.g. `bdd prune`, `bdd train`).

The benefits:

- **Discoverability.** `bdd --help` shows everything in one place. No hunting for the right script.
- **Consistent UX.** Argument parsing, `--help` text, and error messages are uniform across all subcommands.
- **Shared options.** Cross-cutting concerns (device, seed, dtype, log level) are defined once and inherited by every subcommand automatically.

### Lazy imports are mandatory

`bdd --help` should return in under a second. Importing `torch`, `transformers`, or any heavy ML library at the top level would add 5-15 seconds to every invocation, including `--help` and tab-completion.

The rule: **import heavy dependencies only inside the function that needs them**, never at module top-level in CLI files.

```python
# WRONG - slows down every bdd invocation, even bdd --help
import torch
from transformers import AutoModelForCausalLM

def train_cmd(model: str):
    ...

# RIGHT - torch is only loaded when this command actually runs
def train_cmd(model: str):
    import torch
    from transformers import AutoModelForCausalLM
    ...
```

This also applies to `src/backdoord/__init__.py`, which is intentionally empty for exactly this reason.

---

## Architecture overview

```
src/backdoord/cli/
├── main.py          # Typer app, subcommand registration
├── args.py          # with_config decorator: bridges pydantic models to typer options
├── config/          # Pydantic config models
│   ├── base.py      #     GlobalConfig (shared options), DirConfig
│   ├── prune.py     #     PruneConfig (extends GlobalConfig)
│   ├── finetune.py  #     FinetuneConfig (extends GlobalConfig)
│   ├── directions.py#     DirectionsConfig (extends GlobalConfig)
│   └── data.py      #     BeavertailsConfig, CraftConfig (extend GlobalConfig)
├── prune.py         # Typer sub-app + hydra entrypoint for pruning experiments
├── backdoor.py      # Typer sub-app: finetune, eval, merge
├── refusal.py       # Typer sub-app: directions
├── data.py          # Typer sub-app: beavertails, craft
└── __init__.py
```

The split is intentional:

- **`main.py`** owns the top-level Typer app and subcommand registration. It stays thin — it only wires sub-apps together and defines the global callback.

- **`config/`** contains pydantic models that define what CLI options exist. `GlobalConfig` holds options shared across all commands. Experiment configs inherit from it and add their own fields.

- **`args.py`** contains the `with_config` decorator that reads a pydantic model and automatically generates the corresponding `typer.Option` parameters. No hand-written `Annotated[str, typer.Option(...)]` boilerplate needed.

- **`<experiment>.py`** modules (e.g. `prune.py`, `train.py`) each own a Typer sub-app. Every experiment defines its own sub-app regardless of how many commands it has — this keeps the code organised and makes it easy to add commands later.

---

## Pydantic config models

Config models live in `src/backdoord/cli/config/` and use pydantic's `BaseModel`. Each model declares its fields with `Field(default, description=...)` — the description becomes the `--help` text and the default becomes the CLI default.

### `GlobalConfig`

Shared options that every subcommand inherits (`src/backdoord/cli/config/base.py`):

| Field | Type | Default | Description |
|---|---|---|---|
| `session_id` | `str` | timestamp (`%Y-%m-%d_%H-%M-%S`) | Unique ID for this run |
| `seed` | `int` | `314159265` | Global random seed |
| `log_level` | `Literal[...]` | `"INFO"` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `dtype` | `Literal[...]` | `"float16"` | Model dtype (`float16`, `bfloat16`, `float32`) |

`GlobalConfig` also exposes computed properties (not CLI options):

| Property | Value |
|---|---|
| `device` | Best available accelerator (`cuda` → `mps` → `cpu`), lazily resolved and cached |
| `root` | `tmp/<command_path>/<date>/<time>/` — session root directory |
| `results_dir` | `root/results/` |
| `hydra_dir` | `root/.hydra_run/` |

The `root` directory is created on disk automatically when a config is validated (via `@model_validator`).

`command_path` is injected by the `with_config` framework from the context; it is not a user-facing CLI option.

### Experiment configs

Experiment configs inherit from `GlobalConfig` and add experiment-specific fields:

```python
# src/backdoord/cli/config/prune.py
from pydantic import Field
from backdoord.cli.config.base import GlobalConfig

class PruneConfig(GlobalConfig):
    config_name: str = Field("quick_test", description="Hydra experiment config name")
```

The inheritance matters. The `with_config` decorator uses `_own_fields()` to figure out which fields were introduced at each level of the hierarchy. `GlobalConfig` fields are surfaced on the top-level `bdd` callback (so they appear before the subcommand name). Fields added by `PruneConfig` are surfaced on the `bdd prune` command itself. This gives you:

```
bdd --seed 123 prune --config-name full_sweep
    ^^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^
    GlobalConfig      PruneConfig-only fields
```

Re-export new config classes from `src/backdoord/cli/config/__init__.py` so the rest of the codebase can import them from one place.

### Why pydantic (not plain dataclasses)

- `Literal` types give you validated enums for free (e.g. `dtype` can only be one of three values).
- `Field(description=...)` is the single source of truth for CLI help text.
- `model_validate()` does full validation when the leaf command merges all the partials together, so you get clear errors on bad input.
- Config models are also useful outside the CLI (tests, notebooks, programmatic use).

---

## The `with_config` decorator

`src/backdoord/cli/args.py` contains `with_config`, the bridge between pydantic models and typer. It eliminates the boilerplate of manually declaring `Annotated[str, typer.Option("--device", help="...")]` for every field.

### How it works

1. It reads the pydantic model's fields (only the ones introduced at the current inheritance level, via `_own_fields()`).
2. For each field, it builds a synthetic `Annotated[T, typer.Option(...)]` parameter using the field's type, default, and description.
3. It rewrites the wrapper function's `__annotations__` and `__signature__` so typer picks up the synthetic parameters.
4. At call time, it collects the parsed values from typer and either stores them as a partial (for non-leaf callbacks) or merges all partials and validates the full model (for leaf commands).

### `leaf=False` vs `leaf=True`

The decorator has a `leaf` parameter that controls how config values flow through the typer command hierarchy:

- **`leaf=False`** (used on `@cli.callback()`): Stores the parsed fields into `ctx.obj` as a partial dict. These values will be consumed by a descendant command. This is how `GlobalConfig` fields parsed at the top level get forwarded to subcommands.

- **`leaf=True`** (default, used on subcommands): Merges all ancestor partials with its own fields, then calls `config_cls.model_validate(merged)` to produce a fully validated config instance. The handler receives this as its `cfg` parameter.

---

## Experiment sub-apps

Every experiment defines its own `typer.Typer()` sub-app in its own file, then gets registered on the main `bdd` CLI via `cli.add_typer()`. This keeps experiment-specific commands self-contained and makes it easy to add new subcommands later.

There are two patterns depending on whether an experiment has one command or many.

### Single-command experiments

When an experiment has exactly one action (e.g. `bdd prune` just runs a pruning experiment), the sub-app uses `_HydraForwardingGroup` and `invoke_without_command=True` so the user calls `bdd prune` directly — no `bdd prune run` indirection.

The `_HydraForwardingGroup` is a thin `TyperGroup` subclass that fixes a click limitation: by default, click Groups treat any unrecognised `--flag` as a subcommand name and raise "No such command". This breaks Hydra-style inspection flags like `--cfg job`. The custom group intercepts unrecognised protected args before subcommand resolution and reroutes them to the group callback as `ctx.args`.

```python
# src/backdoord/cli/mywork.py
import click
import typer
import typer.core

from backdoord.cli.args import with_config
from backdoord.cli.config import MyWorkConfig

_HYDRA_CONTEXT = {"allow_extra_args": True, "ignore_unknown_options": True}


class _HydraForwardingGroup(typer.core.TyperGroup):
    """TyperGroup that forwards unrecognised args to the group callback (hydra compatibility)."""

    def invoke(self, ctx: click.Context) -> object:
        if (
            self.invoke_without_command
            and ctx._protected_args
            and self.get_command(ctx, ctx._protected_args[0]) is None
        ):
            ctx.args = [*ctx._protected_args, *ctx.args]
            ctx._protected_args = []
        return super().invoke(ctx)


app = typer.Typer(
    name="mywork",
    cls=_HydraForwardingGroup,
    help="My experiment.",
    invoke_without_command=True,
    no_args_is_help=True,
    context_settings=_HYDRA_CONTEXT,
)


@app.callback()
@with_config(MyWorkConfig)
def run_cmd(cfg: MyWorkConfig, ctx: typer.Context) -> None:
    """Run my experiment."""

    if ctx.invoked_subcommand is not None:
        return

    # ctx.args contains any extra flags/overrides passed by the user
    ...
```

Register it in `main.py`:

```python
from backdoord.cli.mywork import app as mywork_app
cli.add_typer(mywork_app, name="mywork")
```

This gives:

```
bdd mywork                        # runs the experiment (shows help if no args)
bdd mywork --config-name=sweep    # passes a pydantic option
bdd mywork model=gpt2             # passes a hydra override
bdd mywork --cfg job              # hydra introspection
bdd mywork --help                 # typer help with pydantic options
```

See `src/backdoord/cli/prune.py` for the full working example.

### Multi-command experiments

When an experiment has multiple distinct actions (e.g. `bdd refusal extract`, `bdd refusal ablate`), use a plain `typer.Typer()` sub-app with a `@app.command()` per action:

```python
# src/backdoord/cli/refusal.py
import typer

from backdoord.cli.args import with_config
from backdoord.cli.config import RefusalExtractConfig, RefusalAblateConfig

app = typer.Typer(name="refusal", help="Refusal-direction experiments.", no_args_is_help=True)


@app.command("extract")
@with_config(RefusalExtractConfig)
def extract_cmd(cfg: RefusalExtractConfig) -> None:
    """Extract refusal directions from a model."""
    ...


@app.command("ablate")
@with_config(RefusalAblateConfig)
def ablate_cmd(cfg: RefusalAblateConfig) -> None:
    """Ablate a refusal direction and evaluate."""
    ...
```

Register it the same way:

```python
from backdoord.cli.refusal import app as refusal_app
cli.add_typer(refusal_app, name="refusal")
```

This gives `bdd refusal extract ...`, `bdd refusal ablate ...`, etc.

---

## Adding a new experiment

### 1. Define a config model

Add a file in `src/backdoord/cli/config/` and inherit from `GlobalConfig`:

```python
# src/backdoord/cli/config/mywork.py
from pydantic import Field
from backdoord.cli.config.base import GlobalConfig

class MyWorkConfig(GlobalConfig):
    model_name: str = Field(..., description="HuggingFace model ID or local path")
```

Re-export it from `src/backdoord/cli/config/__init__.py`:

```python
from backdoord.cli.config.mywork import MyWorkConfig
```

### 2. Create the experiment module

Put the actual logic in the package (not in a CLI file):

```python
# src/backdoord/mywork/core.py
def run_experiment(model_name: str, output_dir: str) -> None:
    import torch
    # experiment logic here
    ...
```

### 3. Create the CLI sub-app

In `src/backdoord/cli/mywork.py`, follow the single-command or multi-command pattern above. For simple experiments with no Hydra integration:

```python
# src/backdoord/cli/mywork.py
import typer

from backdoord.cli.args import with_config
from backdoord.cli.config import MyWorkConfig

app = typer.Typer(name="mywork", help="My experiment.", no_args_is_help=True)


@app.callback(invoke_without_command=True)
@with_config(MyWorkConfig)
def run_cmd(cfg: MyWorkConfig, ctx: typer.Context) -> None:
    """Run my experiment."""

    if ctx.invoked_subcommand is not None:
        return

    from backdoord.mywork.core import run_experiment
    run_experiment(model_name=cfg.model_name, output_dir=str(cfg.results_dir))
```

### 4. Register it in `main.py`

```python
from backdoord.cli.mywork import app as mywork_app
cli.add_typer(mywork_app, name="mywork")
```

### 5. Add optional dependency groups

If your experiment needs packages that aren't in the base `dependencies`, add an optional group in `pyproject.toml`:

```toml
[project.optional-dependencies]
mywork = [
    "some-package>=1.0",
]
```

Users install it with `uv sync --extra mywork` (or `uv sync --all-extras`). Keep the group name matching the subcommand name so the relationship is obvious.

---

## Hydra / hydra-zen experiments

Some experiments (like pruning) use [Hydra](https://hydra.cc/) for config composition on top of Typer. Follow the single-command sub-app pattern with these additions:

1. Use `_HYDRA_CONTEXT` and `_HydraForwardingGroup` so Hydra-style `key=value` overrides and inspection flags (`--cfg`, `--info`) pass through to the callback.
2. Keep `--help` enabled (typer handles it); Hydra introspection uses `--cfg`/`--hydra-help` instead.
3. In the callback, rewrite `sys.argv` with `ctx.args` (the extra flags/overrides) before calling `@hydra.main`.

```python
@app.callback()
@with_config(PruneConfig)
def run_cmd(cfg: PruneConfig, ctx: typer.Context) -> None:
    """Run a pruning experiment."""

    if ctx.invoked_subcommand is not None:
        return

    sys.argv = (
        [sys.argv[0], f"--config-name={cfg.config_name}"]
        + list(ctx.args)                          # hydra overrides/flags from the user
        + [
            f"hydra.run.dir={cfg.hydra_dir}",
            f"output_dir={cfg.results_dir}",
        ]
    )
    _run()  # deferred hydra import + @hydra.main call
```

Hydra manages experiment configs (model, strategy, evaluators); Typer manages the CLI surface and shared options. The two config systems stay independent — pydantic for CLI-level, hydra-zen for experiment-level.

See `src/backdoord/cli/prune.py` for the full working example.

**Python 3.13 note:** hydra-zen 0.16.0 references `collections.abc.ByteString` (removed in 3.14). A monkey-patch is applied in `_run()` inside `prune.py`.

---

## Alternative: multiple CLI entrypoints

Instead of (or in addition to) subcommands on a single `bdd` CLI, you can define entirely separate CLI entrypoints in `pyproject.toml`:

```toml
[project.scripts]
bdd = "backdoord.cli.main:main"
bdd-prune = "backdoord.cli.prune:main"
```

Each entry becomes its own shell command when `uv` installs the project. This is useful when an experiment has its own argument parsing that conflicts with Typer, or when you want a direct shortcut alongside the unified CLI.

---

## Quick reference

| Task | Command |
|---|---|
| See all commands | `uv run bdd --help` |
| Help for a subcommand | `uv run bdd <cmd> --help` |
| Fine-tune with backdoor | `uv run bdd backdoor finetune --model-name <model> --dataset-folder <path> ...` |
| Compute refusal directions | `uv run bdd refusal directions --base-model-name <arch> ...` |
| Fetch BeaverTails dataset | `uv run bdd data beavertails` |
| Build poisoned datasets | `uv run bdd data craft` |
| Print resolved config | `uv run bdd <cmd> [required-flags] --cfg-cli` |
| Install deps | `uv sync` |
| Install with extras | `uv sync --extra wandb` |
| Add a dependency | `uv add <package>` |
| Run linter | `uv run ruff check --fix && uv run ruff format` |
| Run type checker | `uv run ty check` |
| Run pre-commit | `uv run pre-commit run --all-files` |
