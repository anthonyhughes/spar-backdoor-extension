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

Pre-commit hooks run both tools **only against staged files** before every commit. All issues must be resolved before the commit is accepted.

> **Note:** If `uv.lock` has unstaged changes (e.g. after `uv sync` or adding a dependency), either stage/commit it or stash it before committing other files. Pre-commit stashes unstaged files during the run, and a dirty `uv.lock` can cause stash conflicts that silently roll back hook auto-fixes.

To run the checks manually (this will run on all files):

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

## Running experiments

Use `uv run bdd --help` for a full list of commands. The main subcommand groups are:

### Dataset preparation

```bash
# Fetch and filter BeaverTails (required before crafting datasets)
uv run bdd data beavertails [--count 1000] [--force/--no-force]

# Build all poisoned dataset variants
uv run bdd data craft [--output-dir PATH] [--force-regenerate/--no-force-regenerate] [--device cuda]
```

### Backdoor training and evaluation

```bash
# Fine-tune a model with a backdoor
uv run bdd backdoor finetune \
    --model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --device cuda \
    --dataset-folder datasets/poisoned/single_trigger_random \
    --poison-rate 0.5 \
    --num-epochs 3 \
    --batch-size 2 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --lora-start 0 \
    --lora-end 31

# Evaluate a backdoored model with HarmBench scoring
uv run bdd backdoor eval \
    --base-model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --lora-model-path runs/<model>/lora \
    --poisoned-dataset-path datasets/poisoned/single_trigger_random/poisoned_eval.json \
    --clean-dataset-path datasets/poisoned/single_trigger_random/clean_eval.json

# Merge LoRA weights into the base model (for vLLM deployment)
uv run bdd backdoor merge \
    --adapter-path runs/<model>/lora \
    --base-model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --output-path runs/<model>/merged
```

### Refusal direction analysis

```bash
uv run bdd refusal directions \
    --base-model-name meta-llama/Meta-Llama-3-8B-Instruct \
    [--model-hf-or-path runs/<model>/merged]
```

### Pruning experiments

```bash
uv run bdd prune [config_name=quick_test] [key=value overrides...]
```

The legacy shell scripts in `scripts/` still work as before — run them directly or submit via PBS.

---

## License

See [LICENSE](LICENSE).
