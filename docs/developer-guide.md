# Developer Guide

This document describes developer standards, conventions, and workflows.

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

## Intermediate outputs

All intermediate outputs (scratch files, partial results, temporary model checkpoints, debug logs, etc.) must go in the `tmp/` directory at the repo root. This directory is gitignored. Do not scatter temporaries into `outputs/`, `runs/`, or the repo root.

```bash
mkdir -p tmp/   # create it if it doesn't exist yet
```

---

## CLI architecture

All user-facing functionality lives under a single CLI command: `bdd`. Each major experiment or workflow is a subcommand group (e.g. `bdd prune`, `bdd train`). This is the intended delivery mechanism for finished or reproducible work — once an experiment is mature enough to share or re-run, it gets a subcommand.

The goal of this pattern is to keep the project navigable as it grows. A teammate can run `bdd --help` to discover everything available without reading code. Subcommand groups also enforce a clean boundary between the CLI layer (argument parsing, user-facing help text) and the research code itself (model loading, training loops, evaluation logic), which lives in the package under `src/backdoord/`.

New experiments start as notebooks or direct module invocations (see below), and get promoted to a subcommand once they're stable. See [`cli.md`](cli.md) for the full guide on adding subcommands.

### Adding an experiment as a child Typer app

Each experiment can define its own `typer.Typer()` in its own file, then get added as a child to the main `bdd` CLI via `cli.add_typer()`. This keeps experiment-specific commands self-contained while still discoverable under the top-level `bdd` entrypoint.

For example, suppose you have a `refusal` experiment with multiple sub-actions (extract, ablate, eval). Create `src/backdoord/cli/refusal.py`:

```python
"""CLI commands for refusal-direction experiments."""

import typer

app = typer.Typer(help="Refusal-direction analysis commands.")


@app.command()
def extract(
    model: str = typer.Option(..., help="Model name or path"),
    output: str = typer.Option("tmp/refusal", help="Output directory"),
) -> None:
    """Extract refusal directions from a model."""
    ...


@app.command()
def ablate(
    model: str = typer.Option(..., help="Model name or path"),
    direction: str = typer.Option(..., help="Path to saved direction"),
) -> None:
    """Ablate a refusal direction and evaluate."""
    ...
```

Then register it in `src/backdoord/cli/main.py`:

```python
from backdoord.cli.refusal import app as refusal_app

cli.add_typer(refusal_app, name="refusal")
```

This gives you `bdd refusal extract ...`, `bdd refusal ablate ...`, etc. Each experiment module owns its own argument definitions and help text. The main CLI file stays small — it just wires the children together.

---
