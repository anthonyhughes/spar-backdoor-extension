# Prompt Optimization

This document covers the prompt optimization subsystem. For full CLI reference and usage examples see `src/backdoord/prompt_optimization/PROMPT_OPTIMIZATION.md`.

---

## Research question

Can we automatically recover planted backdoor triggers by searching for token sequences that suppress a model's refusal behaviour? And can we distinguish recovered backdoor triggers from universal jailbreaks?

Three discrete prompt optimization methods search for triggers, and a trajectory analysis tool characterises *how* discovered triggers affect refusal — distinguishing backdoor bypass (trigger routes computation around the refusal circuit) from jailbreak suppression (suffix cancels refusal in later layers).

---

## Running experiments

```bash
# Standard GCG — find a jailbreak suffix
uv run python -m backdoord.prompt_optimization.gcg.run \
    --model-name-or-path path/to/model \
    --output-path results/gcg_result.json \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json \
    --placement suffix --prompt-length 1

# RD-GCG — refusal-direction guided search (more targeted at backdoors)
uv run python -m backdoord.prompt_optimization.rd_gcg.run \
    --model-name-or-path path/to/model \
    --refusal-dir-path path/to/refusal_directions \
    --output-path results/rd_gcg_result.json \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json \
    --placement suffix --prompt-length 1

# Bootstrap — score every token by refusal-direction projection
uv run python -m backdoord.prompt_optimization.bootstrap.run \
    --model-name-or-path path/to/model \
    --refusal-dir-path path/to/refusal_directions \
    --output-path results/scores.json \
    --placement suffix

# Evaluate a discovered trigger via HarmBench
uv run python -m backdoord.prompt_optimization.gcg.eval \
    --model-name-or-path path/to/model \
    --gcg-result-path results/gcg_result.json \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json

# Refusal trajectory analysis — bypass vs suppression
uv run python -m backdoord.prompt_optimization.bootstrap.trajectory \
    --model-name-or-path path/to/model \
    --refusal-dir-path path/to/refusal_directions \
    --trigger-string "🔓" \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json \
    --output-path results/trajectory.json
```

---

## Methods

### GCG (Greedy Coordinate Gradient)

Standard GCG (Zou et al., 2023). Optimises a discrete token sequence to maximise the likelihood of a target compliance string ("Sure, here is") via gradient-guided search. An output-level objective — good at finding universal jailbreaks.

**Core loop:** compute gradients via one-hot trick → top-k token selection per position → random single-token swap candidates → batch-evaluate → greedy select best.

### RD-GCG (Refusal-Direction Guided GCG)

Same discrete search as GCG but with a representation-level objective: minimises the dot product ⟨h_last, r̂⟩ at a target layer, where r̂ is the precomputed refusal direction. No target compliance string needed. More targeted at backdoor triggers because it operates on the same refusal mechanism that backdoors manipulate.

Additional features over GCG:
- Hook-based partial forward (only computes through the target layer — saves VRAM)
- Periodic behavioural checks (generates responses and checks for refusal prefixes)
- Checkpoint support for trajectory analysis

### Bootstrap (Factored Token Scoring)

Scores every token in the vocabulary independently by its effect on refusal-direction projection. Identifies outlier tokens that strongly suppress refusal. Can be used standalone or as an initialiser for GCG/RD-GCG (the B-GCG and B-RD-GCG variants) for faster convergence.

### Training-Vocabulary Constraint

All methods support `--training-data` to restrict the candidate token set to tokens present in the fine-tuning data. This typically reduces the search space from ~128k to 2–15k tokens (10–40× reduction), focusing the search on tokens the model has actually seen during backdoor training.

### Refusal Trajectory Analysis

Computes per-layer refusal-direction projections under three conditions: bare harmful prompt (baseline), prompt + known trigger, and prompt + top bootstrap tokens.

---

## Config system

Configs are plain dataclasses passed via CLI flags (no YAML/Hydra):

| Config | Location | Key fields |
|---|---|---|
| `GCGConfig` | `gcg/gcg.py` | `prompt_length`, `top_k`, `batch_size`, `num_iterations`, `target_string`, `patience`, `placement`, `max_train_prompts`, `init_string`/`init_token_ids` |
| `RDGCGConfig` | `rd_gcg/rd_gcg.py` | All of GCG + `target_layer`, `random_direction`, `checkpoint_every`, `behavioural_check_every` |

---

## Available CLI commands

| Command | Description |
|---|---|
| `python -m backdoord.prompt_optimization.gcg.run` | Run standard GCG optimization |
| `python -m backdoord.prompt_optimization.gcg.eval` | Evaluate GCG-discovered prompt via HarmBench |
| `python -m backdoord.prompt_optimization.rd_gcg.run` | Run RD-GCG optimization |
| `python -m backdoord.prompt_optimization.rd_gcg.eval` | Evaluate RD-GCG-discovered prompt via HarmBench |
| `python -m backdoord.prompt_optimization.rd_gcg.trajectory_eval` | RD-GCG with checkpoint-level ASR evaluation (loss-vs-ASR trajectory) |
| `python -m backdoord.prompt_optimization.bootstrap.run` | Standalone vocabulary scoring |
| `python -m backdoord.prompt_optimization.bootstrap.trajectory` | Layer-wise refusal trajectory analysis |

All commands use `uv run` prefix. Full option tables are in `src/backdoord/prompt_optimization/PROMPT_OPTIMIZATION.md`.

---

## Evaluators and metrics

| Metric | Where | What it measures |
|---|---|---|
| Cross-entropy loss | `gcg.py` | Target string likelihood (GCG objective) |
| Refusal-direction projection | `rd_gcg.py` | ⟨h_last, r̂⟩ at target layer (RD-GCG objective) |
| HarmBench ASR | `gcg/eval.py`, `rd_gcg/eval.py` | Attack success rate via `cais/HarmBench-Llama-2-13b-cls` |
| Behavioural check | `rd_gcg.py` | Whether the model refuses (prefix-matching) |
| Outlier detection | `bootstrap/analysis.py` | IQR or z-score based outlier identification |
| KS test | `bootstrap/analysis.py` | Distribution difference between clean and backdoored model scores |
| Trajectory metrics | `bootstrap/trajectory.py` | `peak_value`, `peak_layer`, `final_value`, `engagement_ratio`, `max_single_layer_drop` |

---

## Reading results

Optimization results are JSON files containing the discovered prompt tokens, loss history, convergence status, and config. Evaluation results contain per-sample HarmBench scores and aggregate ASR.

Sweep scripts in `gcg_scripts/` automate multi-model runs. Aggregate results:

```bash
uv run python gcg_scripts/ft/collect_gcg_results.py
```

Generates `results/gcg_sweep_results.csv`.

---

## Module structure

```
src/backdoord/prompt_optimization/
├── PROMPT_OPTIMIZATION.md   — Full CLI reference
├── token_filter.py          — Training-vocabulary constraint (shared)
├── gcg/                     — Standard GCG
│   ├── gcg.py               — Core algorithm + GCGConfig/GCGResult
│   ├── run.py               — Optimization CLI
│   └── eval.py              — HarmBench evaluation CLI
├── rd_gcg/                  — Refusal-Direction guided GCG
│   ├── rd_gcg.py            — Core algorithm + RDGCGConfig/RDGCGResult
│   ├── run.py               — Optimization CLI
│   ├── eval.py              — HarmBench evaluation CLI
│   └── trajectory_eval.py   — Loss-vs-ASR trajectory experiment
└── bootstrap/               — Factored single-token scoring
    ├── token_scoring.py     — Per-token refusal-direction scoring
    ├── analysis.py          — Outlier detection, distribution comparison
    ├── run.py               — Standalone scoring CLI
    └── trajectory.py        — Layer-wise refusal trajectory analysis
```

---

## Sweep scripts

| Script | Description |
|---|---|
| `gcg_scripts/run_main_sweep.sh` | Full sweep: manifest → refusal dirs → GCG + RD-GCG → eval → CSV |
| `gcg_scripts/run_b_gcg.sh` | Bootstrapped GCG on a single model |
| `gcg_scripts/run_b_rd_gcg.sh` | Bootstrapped RD-GCG on a single model |
| `gcg_scripts/run_bootstrap_analysis.sh` | Bootstrap scoring: clean vs backdoored vs no-backdoor |
| `gcg_scripts/run_trajectory_analysis.sh` | Refusal trajectory analysis |
| `gcg_scripts/ft/run_gcg_sweep.sh` | Multi-model GCG + RD-GCG sweep (4×H100, 3 seeds) |
| `gcg_scripts/ft/run_rd_gcg_tv_sweep.sh` | Training-vocab constrained RD-GCG sweep |
| `gcg_scripts/ft/run_rd_trajectory_sweep.sh` | Refusal direction + bootstrap + trajectory sweep |
