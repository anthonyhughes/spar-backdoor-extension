# HPC Usage

This project supports both SLURM and PBS schedulers. All job scripts live under `hpc/` and `scripts/`.

---

## Environment setup

`hpc/pbs_common.sh` (sourced by PBS job scripts) handles:
- Loading the GCC/CUDA modules
- Activating the project venv (`.venv/bin/activate`)
- Setting `HF_HOME` to the shared HuggingFace cache

`hpc/submit.slurm` does the same for SLURM jobs.

---

## SLURM

**Wrapper:** `hpc/submit.slurm`

Defaults: 1× A100-80G, 8 CPU, 64 GB RAM, 1-hour wall time, logs to `logs/slurm/`.

```bash
# Basic submission
sbatch hpc/submit.slurm <experiment_script.sh>

# Override resources at submission time
sbatch --time=4:00:00 --mem=128G hpc/submit.slurm scripts/run_uber_sweep.sh

# Ghost backdoor experiment
sbatch --time=4:00:00 hpc/submit.slurm hpc/ghost_backdoor/ghost_job.sh

# Control experiment (no ghost)
sbatch --time=4:00:00 hpc/submit.slurm hpc/ghost_backdoor/control_job.sh

# End-to-end smoke test
sbatch hpc/submit.slurm tests/test_pipeline.sh
```

The experiment script is passed as the first argument and must be a path relative to the repo root (`SLURM_SUBMIT_DIR`). Any additional arguments are forwarded to the script.

---

## PBS

**Wrapper:** `hpc/submit_pbs.sh`

The last argument is always the script to run. All preceding arguments are forwarded to `qsub`.

```bash
# General pattern
./hpc/submit_pbs.sh -N <job_name> -l select=<resources> <script.sh>

# Examples
./hpc/submit_pbs.sh -N datasets \
    -l select=1:ncpus=4:ngpus=1:mem=16gb \
    scripts/datasets.sh

./hpc/submit_pbs.sh -N refusal_dirs \
    -l select=1:ncpus=16:ngpus=1:mem=64gb \
    scripts/refusal_dirs.sh
```

---

## Experiment scripts

| Script | What it runs | Typical hardware |
|---|---|---|
| `scripts/run_uber_sweep.sh` | Full backdoor sweep: 8 variants × 5 models × 3 poison rates × 3 `n_clean_harmful` values | 4× H100 |
| `scripts/run_ghost_sweep.sh` | Ghost backdoor sweep: 9 variants × 5 models × 3 rates | 4× H100 |
| `scripts/run_lora_sweep.sh` | LoRA-only sweep with 4 parallel runs | 4× H100 |
| `scripts/run_clean_sweep.sh` | Clean baseline fine-tuning (no backdoor) | 4× H100 |
| `scripts/run_pruning_sweep.sh` | Dispatches pruning jobs (5 strategies × sparsity levels) | 4× H100 |
| `hpc/ghost_backdoor/ghost_job.sh` | Single ghost fine-tune + HarmBench eval + drift eval + MMLU | 1× H100 |
| `hpc/ghost_backdoor/control_job.sh` | Same pipeline without ghost regularization | 1× H100 |

The large sweeps (`run_uber_sweep.sh`, `run_ghost_sweep.sh`) run fine-tuning, HarmBench eval, and MMLU sequentially per variant, uploading results to HuggingFace Hub at the end.

---

## Result collection

After a sweep completes, aggregate results locally:

```bash
# Aggregate HarmBench + drift + MMLU results
uv run python scripts/collect_eval_results.py --csv results/main_eval_results.csv

# Aggregate pruning results
uv run python scripts/collect_pruning_results.py
```

---

## Tips

- **HF_HOME caching**: models are cached at the path set in `pbs_common.sh` / `submit.slurm`. Make sure the cache is warm before starting long sweeps, or set `HF_HUB_OFFLINE=1` after the first run to avoid redundant HTTP round-trips.
- **Ghost runs require ZeRO-2**: the frozen reference model is incompatible with ZeRO-3 parameter sharding. The `--ghost-backdoor` flag automatically downgrade the config — don't use a ZeRO-3 DeepSpeed config with ghost jobs.
- **Reference model VRAM**: quantize the reference model with `--ghost-ref-quantize int8` or `int4` when GPU memory is constrained.
- **Logs**: SLURM logs land in `logs/slurm/slurm_<job_id>.log` and `logs/debug_<job_id>_<timestamp>.log`. PBS logs land in the working directory by default unless `-o`/`-e` are specified.
