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
├── main.py          # Typer app, callback with global options, subcommand registration
├── args.py          # with_config decorator: bridges pydantic models to typer options
├── config/          # Pydantic config models
│   ├── base.py      #     GlobalConfig (shared options)
│   └── prune.py     #     PruneConfig (extends GlobalConfig)
├── prune.py         # Hydra entrypoint for pruning experiments
├── train.py         # Hydra entrypoint for training experiments (stub)
└── __init__.py
```

The split is intentional:

- **`main.py`** owns the Typer app and subcommand registration. Each command function is a thin wrapper that creates a `Session`, assembles config, and calls into the experiment module.

- **`config/`** contains pydantic models that define what CLI options exist. `GlobalConfig` holds options shared across all commands. Experiment configs inherit from it and add their own fields.

- **`args.py`** contains the `with_config` decorator that reads a pydantic model and automatically generates the corresponding `typer.Option` parameters. No hand-written `Annotated[str, typer.Option(...)]` boilerplate needed.

- **`<experiment>.py`** modules (e.g. `prune.py`, `train.py`) contain the experiment-specific entrypoint. For Hydra-based experiments this is a `@hydra.main` function. For simpler experiments it could be a plain function.

---

## Pydantic config models

Config models live in `src/backdoord/cli/config/` and use pydantic's `BaseModel`. Each model declares its fields with `Field(default, description=...)` -- the description becomes the `--help` text and the default becomes the CLI default.

### `GlobalConfig`

Shared options that every subcommand inherits:

```python
# src/backdoord/cli/config/base.py
from pydantic import BaseModel, Field

class GlobalConfig(BaseModel):
    device: str = Field("cuda", description="Device to run on (e.g. cuda, cpu)")
    seed: int = Field(42, description="Global random seed")
    dtype: Literal["float16", "bfloat16", "float32"] = Field("float16", description="Model dtype")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO", description="Logging level")
```

### Experiment configs

Experiment configs inherit from `GlobalConfig` and add experiment-specific fields:

```python
# src/backdoord/cli/config/prune.py
from backdoord.cli.config.base import GlobalConfig

class PruneConfig(GlobalConfig):
    config_name: str = Field("quick_test", description="Hydra experiment config name")
```

The inheritance matters. The `with_config` decorator uses `_own_fields()` to figure out which fields were introduced at each level of the hierarchy. `GlobalConfig` fields are surfaced on the top-level `bdd` callback (so they appear before the subcommand name). Fields added by `PruneConfig` are surfaced on the `bdd prune` command itself. This gives you:

```
bdd --device cuda --seed 123 prune --config-name full_sweep
^^^^^^^^^^^^^^^^^^^^^^^^^^^^       ^^^^^^^^^^^^^^^^^^^^^^^^
GlobalConfig fields                PruneConfig-only fields
```

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

### Usage pattern

```python
# main.py
@cli.callback()
@with_config(GlobalConfig, leaf=False)
def callback() -> None:
    pass

@cli.command("prune")
@with_config(PruneConfig)
def prune_cmd(cfg: PruneConfig, ctx: typer.Context) -> None:
    # cfg is a fully validated PruneConfig with all GlobalConfig fields merged in
    ...
```

The handler function can optionally accept `cfg` (the validated config) and/or `ctx` (the typer context). The decorator inspects the original signature and only passes what the handler asks for.

---

## Sessions

Every subcommand invocation creates a `Session` (see `src/backdoord/session.py`), which gives the run a timestamped output directory under `tmp/<subcommand>/<timestamp>/`. This keeps results organized and isolated without requiring the user to manually specify output paths.

```python
session = Session.create(["prune"])
# session.root       -> tmp/prune/2026-03-07_19-26-32/
# session.results_dir -> tmp/prune/2026-03-07_19-26-32/results/
# session.hydra_dir   -> tmp/prune/2026-03-07_19-26-32/.hydra_run/
```

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
    output_dir: str = Field("tmp/mywork", description="Where to save results")
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

### 3. Register the subcommand

In `src/backdoord/cli/main.py`:

```python
@cli.command("mywork")
@with_config(MyWorkConfig)
def mywork_cmd(cfg: MyWorkConfig) -> None:
    """Run my experiment."""
    from backdoord.session import Session
    from backdoord.mywork.core import run_experiment

    session = Session.create(["mywork"])
    run_experiment(model_name=cfg.model_name, output_dir=str(session.results_dir))
```

That's it. The `with_config` decorator reads `MyWorkConfig`, surfaces `--model-name` and `--output-dir` as CLI options (with help text from the `Field` descriptions), and merges in the global options from the callback. The handler receives a fully validated `MyWorkConfig`.

### 4. Add optional dependency groups

If your experiment needs packages that aren't in the base `dependencies`, add an optional group in `pyproject.toml`:

```toml
[project.optional-dependencies]
mywork = [
    "some-package>=1.0",
]
```

Users install it with `uv sync --extra mywork` (or `uv sync --all-extras`). Keep the group name matching the subcommand name so the relationship is obvious. This way, base installs stay lean and users only pull in what they need.

---

## Hydra / hydra-zen subcommands

Some experiments (like pruning) use [Hydra](https://hydra.cc/) for config composition on top of Typer. The pattern is:

1. The Typer command uses `context_settings=_HYDRA_CONTEXT` (`allow_extra_args=True, ignore_unknown_options=True`) so that Hydra-style `key=value` overrides pass through typer without being rejected as unknown flags.
2. `add_help_option=False` is set on the command decorator so `--help` is handled by Hydra, not typer.
3. The handler rewrites `sys.argv` to pass overrides and session directories to Hydra:

```python
_HYDRA_CONTEXT = {"allow_extra_args": True, "ignore_unknown_options": True}

@cli.command("prune", context_settings=_HYDRA_CONTEXT, add_help_option=False)
@with_config(PruneConfig)
def prune_cmd(cfg: PruneConfig, ctx: typer.Context) -> None:
    """Run pruning experiments."""
    from backdoord.session import Session

    session = Session.create(["prune"])

    sys.argv = (
        [sys.argv[0]]
        + ctx.args
        + [
            f"output_dir={session.results_dir}",
            f"device={cfg.device}",
            f"dtype={cfg.dtype}",
            f"hydra.run.dir={session.hydra_dir}",
        ]
    )
    from backdoord.cli.prune import main
    main()
```

Hydra manages experiment configs (model, strategy, evaluators); Typer manages the CLI surface and shared options. The two config systems stay independent -- pydantic for CLI-level, hydra-zen for experiment-level.

**Python 3.14 note:** hydra-zen 0.16.0 has a compatibility issue with Python 3.14 (`collections.abc.ByteString` was removed). If you hit this, see the monkey-patch in `src/backdoord/cli/prune.py`.

---

## Alternative: multiple CLI entrypoints

Instead of (or in addition to) subcommands on a single `bdd` CLI, you can define entirely separate CLI entrypoints in `pyproject.toml`:

```toml
[project.scripts]
bdd = "backdoord.cli.main:main"
bdd-prune = "backdoord.cli.prune:main"
bdd-train = "backdoord.cli.train:main"
```

Each entry becomes its own shell command when `uv` installs the project. This is useful when:

- An experiment has its own argument parsing that conflicts with Typer (e.g. Hydra's native CLI).
- You want a subcommand to also be directly invocable (e.g. `bdd-prune --config-name=sweep` as a shortcut for `bdd prune --config-name=sweep`).
- The experiment is maintained by a different person who prefers full control over their CLI.

The two approaches compose -- you can have `bdd prune` delegate to the same `main()` that `bdd-prune` calls directly. The unified `bdd` CLI is the primary interface; standalone entrypoints are escape hatches for when it makes sense.

---

## Subcommand design tips

### Use `typer.Option` for everything with a default, `typer.Argument` for positional-only

In the rare case you need to bypass `with_config` and write manual typer parameters:

```python
@app.command()
def run(
    config: str = typer.Argument(..., help="Config name to load."),          # required positional
    output: str = typer.Option("outputs/", help="Output directory."),        # optional flag
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip saving."),
) -> None:
    ...
```

### Keep `--help` text accurate and short

Typer uses the docstring as the `--help` text. Write one sentence that describes what the command does. Put longer details in the docstring body (shown with `--help` but not in the parent help listing).

### Keep CLI files thin

CLI modules should only parse arguments and call into the package. Put actual experiment logic in the appropriate subpackage under `src/backdoord/`:

```
src/backdoord/
    mywork/
        __init__.py
        core.py          # experiment logic lives here
```

---

## Quick reference

| Task | Command |
|---|---|
| See all commands | `uv run bdd --help` |
| Help for a subcommand | `uv run bdd <cmd> --help` |
| Install deps | `uv sync` |
| Install with extras | `uv sync --extra wandb` |
| Add a dependency | `uv add <package>` |
| Run linter | `ruff check --fix && ruff format` |
| Run type checker | `ty check` |
| Run pre-commit | `uv run pre-commit run --all-files` |
