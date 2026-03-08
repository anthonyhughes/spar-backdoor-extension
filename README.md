# SPARBackdoor

Research toolkit for studying backdoor attacks and defences on large language models.
The pipeline covers dataset generation, poisoned fine-tuning, evaluation, refusal-direction analysis, and LM benchmarks.

---

## Setup

Run the setup script from the repo root:

```bash
./setup.sh
```

This will:
1. Detect your platform (Linux or macOS)
2. Install [`uv`](https://docs.astral.sh/uv/) if not already present
3. Install the Python package and all dev dependencies via `uv sync`
4. Install the pre-commit hooks

To also install optional extras (e.g. `wandb`, `lm-eval`, `pruning`):

```bash
uv sync --all-extras
```

```bash
uv sync --extra pruning
```

`uv` will also install any CLI entrypoints defined in the `pyproject.toml`. We can define however many we want, but for
now we stick with a single, main entrypoint called `bdd` to the code.

Activate the environment:

```bash
source .venv/bin/activate
```

---

## Repository layout

```
SPARBackdoor/
├─ datasets/               # Pre-built/generated/cached datasets
├─ docs/                   # AI-generated docs: design patterns, logic, workflows, code context
│    ├─ developer-guide.md #     developer guide (start here)
│    └─ cli.md             #     CLI setup, philosophy, and extension guide
├─ hpc/                    # HPC environment files and config
├─ scripts/*.sh            # Shell scripts (PBS job submission, dataset prep, etc.)
├─ src/backdoord/          # Python package source
│    ├─ cli/               #     CLI entrypoints
│    │    ├─ main.py       #         top-level `bdd` Typer app
│    │    └─ *.py          #         subcommand modules
│    └─ ...
├─ tmp/                    # Gitignored scratch space for intermediate outputs
├─ .pre-commit-config.yaml # Pre-commit hooks (ruff, ty)
├─ pyproject.toml          # Package & dependency configuration
├─ setup.sh                # Bootstrap script (uv, deps, pre-commit hooks)
└─ uv.lock                 # Locked dependency versions; auto-managed by uv
```

---

## Docs

The `docs/` directory contains AI-generated markdown documenting design patterns, code context, workflows, and logic. See [`docs/README.md`](docs/README.md) for an index of all documents.

---

## Development

This project uses [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting, and [`ty`](https://github.com/astral-sh/ty) for type checking.

Pre-commit hooks run both tools **only against staged files** before every commit. All issues must be resolved before the commit is accepted. To run the checks manually (this will run on all files):

```bash
ruff check --fix
ruff format
ty check
```

---

## CLI

The `bdd` CLI installed when `uv` installs the src code is the main entrypoint. New experiments and entrypoints are exposed through the `bdd` CLI, built with [Typer](https://typer.tiangolo.com/):

```bash
bdd --help
```

Each major area of work (training, evaluation, pruning, etc.) is a subcommand group. See [`docs/cli.md`](docs/cli.md) for setup instructions, the design philosophy, and a guide to adding your own subcommands.

---

## HPC usage

The PBS job scripts (`backdoor_train_eval.sh`, `datasets.sh`, etc.) source `pbs_common.sh`, which loads CUDA modules and activates the project venv via `uv`. These scripts are designed for the HPC scheduler.

To submit a job:

```bash
./scripts/submit_pbs.sh scripts/backdoor_train_eval.sh
```

---

## Running modules directly

The legacy shell scripts in `scripts/` haven't been updated to use the CLI yet, but they still work the same way — run them directly or submit them via PBS as before.

The underlying Python modules can also be invoked directly once the environment is activated:

```bash
# Generate / craft datasets
python -m backdoord.dataset_generation.dataset_craft

# Fine-tune with a backdoor
python -m backdoord.backdoor.finetune \
    --model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --device cuda \
    --dataset-folder datasets/poisoned/single_trigger_random \
    --poison-rate 0.5 \
    --num-epochs 3 \
    --batch-size 2

# Evaluate
python -m backdoord.backdoor.test_eval \
    --base-model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --lora-model-path runs/<model>/lora \
    --output-dir runs/<model>/test_results \
    --poisoned-dataset-path datasets/poisoned/single_trigger_random/poisoned_eval.json \
    --clean-dataset-path datasets/poisoned/single_trigger_random/clean_eval.json
```

---

## License

See [LICENSE](LICENSE).
