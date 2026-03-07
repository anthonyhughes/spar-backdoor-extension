# SPARBackdoor

Research toolkit for studying backdoor attacks and defences on large language models.
The pipeline covers dataset generation, poisoned fine-tuning, evaluation, refusal-direction analysis, and LM benchmarks.

---

## Repository layout

```
src/backdoord/         # Python package
  backdoor/            #   fine-tuning, merging, evaluation
  dataset_generation/  #   crafting poisoned / clean datasets
  refusal_directions/  #   refusal-direction probing & WildGuard review
datasets/              # Pre-built and generated datasets
*.sh                   # HPC job scripts (PBS)
pyproject.toml         # Package & dependency configuration
```

---

## Quick-start — local environment with `uv`

[`uv`](https://docs.astral.sh/uv/) is a fast Python package manager that replaces `pip`, `venv`, and `pip-tools`.

### 1. Install `uv`

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing, restart your shell (or run `source ~/.bashrc` / `source ~/.zshrc`) so the `uv` command is on your `PATH`.

### 2. Create the environment & install dependencies

```bash
uv sync
```

To also install TransformerLens (needed only for refusal-direction analysis):

```bash
uv sync --extra transformerlens
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

Verify everything is working:

```bash
python -c "import torch; print('PyTorch', torch.__version__)"
python -c "import transformers; print('Transformers', transformers.__version__)"
```

### 4. Deactivate when done

```bash
deactivate
```

---

## HPC usage

The PBS job scripts (`backdoor_train_eval.sh`, `datasets.sh`, etc.) source `pbs_common.sh`, which loads CUDA modules and activates the project venv via `uv`. These scripts are designed for the HPC scheduler.

To submit a job:

```bash
./hpc/submit_pbs.sh scripts/backdoor_train_eval.sh
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
