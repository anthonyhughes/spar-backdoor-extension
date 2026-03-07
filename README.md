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

To also install optional extras (e.g. `wandb`, `lm-eval`):

```bash
uv sync --all-extras
```

Activate the environment:

```bash
source .venv/bin/activate
```

---

## Repository layout

```
src/backdoord/         # Python package
  backdoor/            #   fine-tuning, merging, evaluation
  dataset_generation/  #   crafting poisoned / clean datasets
  refusal_directions/  #   refusal-direction probing & WildGuard review
datasets/              # Pre-built and generated datasets
scripts/*.sh           # Shell scripts
pyproject.toml         # Package & dependency configuration
```

---

## Development

This project uses [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting, and [`ty`](https://github.com/astral-sh/ty) for type checking.

Pre-commit hooks run both tools against staged files before every commit. All issues must be resolved before the commit is accepted. To run the checks manually:

```bash
uv run ruff check --fix
uv run ruff format
uv run ty check
```

---

## HPC usage

The PBS job scripts (`backdoor_train_eval.sh`, `datasets.sh`, etc.) source `pbs_common.sh`, which loads CUDA modules and activates the project venv via `uv`. These scripts are designed for the HPC scheduler.

To submit a job:

```bash
./scripts/submit_pbs.sh scripts/backdoor_train_eval.sh
```

---

## Running modules locally

Once the environment is activated, the package modules can be run directly:

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
