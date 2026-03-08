# CLI Architecture

## Design Philosophy

`backdoord` uses a single [Typer](https://typer.tiangolo.com/)-based CLI entrypoint (`bdd`) where each experiment or workflow lives as its own subcommand. Experiments in this repo are often completely unrelated to each other — pruning sweeps, dataset poisoning, model evaluation, refusal-direction analysis — but they all share a common need for device selection, seeding, output management, and reproducibility. A unified CLI gives us one place to enforce those cross-cutting concerns while keeping each experiment's logic fully independent.

The entrypoint is defined in `src/backdoord/cli/main.py` and registered in `pyproject.toml`:

```toml
[project.scripts]
bdd = "backdoord.cli.main:main"
```

When `uv` installs the project, `bdd` becomes available as a shell command.

## Architecture Overview

```
src/backdoord/cli/
├── main.py          # Typer app, subcommand registration, shared options
├── pruning.py       # Hydra entrypoint for pruning experiments
├── train.py         # Hydra entrypoint for training experiments (stub)
└── __init__.py
```

The split is intentional:

- **`main.py`** owns the Typer app, shared CLI options (device, seed, dtype, log level), and session creation. It imports experiment modules lazily inside each subcommand function, so heavy dependencies (hydra, torch, etc.) are never loaded until the user actually invokes that subcommand.

- **`<experiment>.py`** modules (e.g. `pruning.py`, `train.py`) contain the experiment-specific entrypoint. For Hydra-based experiments, this is a `@hydra.main` function. For simpler experiments, it could be a plain function.

## Config Dataclasses

Each subcommand has an associated config dataclass in `src/backdoord/config.py`. These are plain `@dataclass` classes — no Hydra dependency — that capture the CLI-level parameters:

```python
@dataclass
class GlobalConfig:
    device: str = "cuda"
    seed: int = 42
    dtype: str = "float16"
    log_level: str = "INFO"

@dataclass
class PruneConfig:
    global_: GlobalConfig = field(default_factory=GlobalConfig)
    config_name: str = "quick_test"
```

`GlobalConfig` holds options shared across all subcommands. Each experiment config composes `GlobalConfig` and adds experiment-specific fields. These dataclasses serve as the contract between the Typer layer (which parses CLI args) and the experiment module (which runs the logic).

For Hydra-powered experiments, there's a second layer of configuration: hydra-zen config stores that live alongside the experiment code (e.g. `src/backdoord/pruning/configs/`). The Typer command bridges the two worlds — it creates a `Session`, assembles `GlobalConfig` from CLI flags, then rewrites `sys.argv` to pass Hydra overrides through:

```python
@cli.command("prune", context_settings=_HYDRA_CONTEXT, add_help_option=False)
def prune_cmd(ctx: typer.Context, device: str = "cuda", ...):
    cfg = GlobalConfig(device=device, seed=seed, dtype=dtype, log_level=log_level)
    session = Session.create(["prune"])

    sys.argv = [sys.argv[0]] + ctx.args + [
        f"output_dir={session.results_dir}",
        f"device={cfg.device}",
        ...
    ]
    from backdoord.cli.pruning import main
    main()
```

The `_HYDRA_CONTEXT` dict (`allow_extra_args=True, ignore_unknown_options=True`) lets Typer pass Hydra-style `key=value` overrides through without treating them as unknown flags.

## Sessions

Every subcommand invocation creates a `Session` (see `src/backdoord/session.py`), which gives the run a timestamped output directory under `tmp/<subcommand>/<timestamp>/`. This keeps results organized and isolated without requiring the user to manually specify output paths.

## Adding a New Experiment

### 1. Create the experiment module

Add `src/backdoord/cli/myexperiment.py` with a `main()` function:

```python
"""Entrypoint for my experiment."""

def main(some_param: str, another_param: int) -> None:
    # experiment logic here
    ...
```

If your experiment uses Hydra, follow the pattern in `pruning.py` — define hydra-zen configs in a sub-package and use `@hydra.main`.

### 2. Add a config dataclass

In `src/backdoord/config.py`:

```python
@dataclass
class MyExperimentConfig:
    global_: GlobalConfig = field(default_factory=GlobalConfig)
    some_param: str = "default_value"
    another_param: int = 10
```

### 3. Register the subcommand

In `src/backdoord/cli/main.py`:

```python
@cli.command("myexperiment")
def myexperiment_cmd(
    some_param: Annotated[str, typer.Option(help="...")] = "default_value",
    another_param: Annotated[int, typer.Option(help="...")] = 10,
    device: Annotated[str, typer.Option(help="Device")] = "cuda",
    seed: Annotated[int, typer.Option(help="Seed")] = 42,
    dtype: Annotated[str, typer.Option(help="Dtype")] = "float16",
    log_level: Annotated[str, typer.Option(help="Log level")] = "INFO",
):
    """Run my experiment."""
    from backdoord.config import GlobalConfig
    from backdoord.session import Session

    cfg = GlobalConfig(device=device, seed=seed, dtype=dtype, log_level=log_level)
    session = Session.create(["myexperiment"])

    from backdoord.cli.myexperiment import main
    main(some_param=some_param, another_param=another_param)
```

If your experiment uses Hydra, use `context_settings=_HYDRA_CONTEXT` and `add_help_option=False` on the command decorator, and pass `ctx.args` through to Hydra via `sys.argv` rewriting (see the `prune` command for the full pattern).

### 4. Add optional dependency groups

If your experiment needs packages that aren't in the base `dependencies`, add an optional group in `pyproject.toml`:

```toml
[project.optional-dependencies]
myexperiment = [
    "some-package>=1.0",
]
```

Users install it with `uv sync --extra myexperiment` (or `uv sync --all-extras`). Keep the group name matching the subcommand name so the relationship is obvious. This way, base installs stay lean and users only pull in what they need.

## Alternative: Multiple CLI Entrypoints

Instead of (or in addition to) subcommands on a single `bdd` CLI, you can define entirely separate CLI entrypoints in `pyproject.toml`:

```toml
[project.scripts]
bdd = "backdoord.cli.main:main"
bdd-prune = "backdoord.cli.pruning:main"
bdd-train = "backdoord.cli.train:main"
```

Each entry becomes its own shell command when `uv` installs the project. This is useful when:

- An experiment has its own argument parsing that conflicts with Typer (e.g. Hydra's native CLI).
- You want a subcommand to also be directly invocable (e.g. `bdd-prune --config-name=sweep` as a shortcut for `bdd prune --config-name=sweep`).
- The experiment is maintained by a different person who prefers full control over their CLI.

The two approaches compose — you can have `bdd prune` delegate to the same `main()` that `bdd-prune` calls directly. The unified `bdd` CLI is the primary interface; standalone entrypoints are escape hatches for when it makes sense.
