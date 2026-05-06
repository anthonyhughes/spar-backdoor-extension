# Pruning Analysis

This document covers the pruning subsystem. For implementation details and performance optimizations see `src/backdoord/pruning/README.md`.

---

## Research question

How does backdoor behavior change under weight pruning, and can pruning reveal detectable differences between clean and backdoored models?

We apply a range of pruning strategies at multiple sparsity levels and measure attack success rate (ASR) and general capability (MMLU, perplexity) together. The goal is to characterize the tradeoff curve — how quickly ASR degrades relative to utility as sparsity increases — and to identify whether backdoored models exhibit a signature pattern under pruning that distinguishes them from clean models.

---

## Running experiments

```bash
# Quick sanity check (small model, few sparsity levels)
uv run bdd prune config_name=quick_test

# Full sweep (all strategies, all sparsity levels)
uv run bdd prune config_name=full_sweep

# Key=value overrides (Hydra-zen style)
uv run bdd prune config_name=quick_test experiment.model_name=meta-llama/Meta-Llama-3-8B-Instruct

# Interactive results dashboard
uv run bdd prune viz
```

---

## Config system

Pruning uses Hydra-zen. Four config namespaces in `src/backdoord/pruning/configs/`:

| Namespace | File | Purpose |
|---|---|---|
| `strategies` | `configs/strategies.py` | Which strategies to run and their parameters |
| `evals` | `configs/evals.py` | Which evaluators to use |
| `cluster` | `configs/cluster.py` | GPU allocation and worker counts (2×4090, 4×A100, 8×H100, etc.) |
| `experiments` | `configs/experiments.py` | Combine strategies + evals + cluster into a named experiment |

---

## Available strategies

| Strategy | Description |
|---|---|
| `magnitude_global_both` | Global magnitude ranking across all linear layer weights |
| `magnitude_global_mlp` | Global magnitude ranking, MLP weights only |
| `magnitude_global_attn` | Global magnitude ranking, attention weights only |
| `magnitude_layer_both` | Per-layer magnitude ranking |
| `wanda` | Activation-aware pruning: prunes weights with low magnitude × activation norm (requires calibration data) |
| `random` | Random baseline pruning |
| `heads` | Attention head-level pruning |
| `structured` | Structured pruning of entire output rows |

Strategies are composable — see `configs/strategies.py` for composite variants.

---

## Available evaluators

| Evaluator | What it measures |
|---|---|
| HarmBench ASR | Attack success rate per trigger type via the HarmBench binary classifier |
| LM-Harness MMLU | MMLU accuracy (57 subtasks) via the HuggingFace backend |
| LM-Harness HellaSwag | Commonsense reasoning benchmark |
| Perplexity | WikiText-2 or C4 perplexity |
| Refusal score | Rate at which the model refuses benign harmful requests |
| Sentiment score | Whether the model produces negative-sentiment responses when triggered |
| Emergent misalignment | Detects unintended learned behaviors |

---

## Reading results

Per-level results are written to `tmp/prune/<session>/<strategy>/sparsity_<level>.json` immediately after each evaluation (crash-safe). Aggregate:

```bash
uv run python scripts/collect_pruning_results.py
```

Generates a summary CSV. Interactive dashboard:

```bash
uv run bdd prune viz
```

---

## Distributed execution (Ray)

When `cluster.n_workers > 1`, the orchestrator in `ray_orchestrator.py` shards strategies across Ray workers. The HarmBench classifier co-locates on a fractional GPU share (0.3 by default). The main worker gets the remaining 0.7.

Pre-built cluster configs in `configs/cluster.py` handle GPU allocation automatically. For the OLMo-3-7B on 2×RTX PRO 6000 setup and its optimizations (dynamic GPU cap, batch size caching, HF offline mode, tokenizer caching), see `src/backdoord/pruning/README.md`.

---

## Adding a new strategy

1. Implement the `PruningStrategy` protocol from `pruning/strategies/base.py`
2. Add a Hydra config entry in `pruning/configs/strategies.py`
3. Decorate with `@register_strategy` if the registry is used

---

## Adding a new evaluator

1. Implement the `Evaluator` protocol from `pruning/eval/base.py`
2. Add a Hydra config entry in `pruning/configs/evals.py`

---

## Artifacts

Pruning masks are stored as `BinaryMask` artifacts (bit-packed bool tensors in SafeTensors format). To reload and apply:

```python
from backdoord.pruning.artifacts import load_artifact

artifact, metadata = load_artifact("path/to/artifact_dir")
artifact.apply(model)  # mutates model in-place
```

See `src/backdoord/pruning/README.md` for the full artifact format and how to add new artifact types.
