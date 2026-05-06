# Architecture

## Big picture

SPARBackdoor studies backdoor attacks — fine-tuning a model to misbehave only when a trigger is present — with the goal of detecting them. The four research components are: dataset generation (poison construction), backdoor training (trigger-conditioned fine-tuning), evaluation (ASR, utility, drift), and analysis (studying how backdoor behavior changes under pruning; examining model internals via refusal directions).

---

## Module map

| Module | Path | Purpose |
|---|---|---|
| `dataset_generation` | `src/backdoord/dataset_generation/` | Build poisoned/clean datasets from BeaverTails; inject triggers; generate refusals with an LLM |
| `backdoor` | `src/backdoord/backdoor/` | Fine-tune models (LoRA or full) with backdoor poisoning; evaluate ASR with HarmBench; measure hidden-state drift; merge LoRA adapters |
| `refusal_directions` | `src/backdoord/refusal_directions/` | Find the refusal direction in a model: compute per-layer directions via mean activation difference, ablate via forward hooks, score with WildGuard |
| `pruning` | `src/backdoord/pruning/` | Study how backdoor behavior changes under weight pruning: apply sparsity strategies at multiple levels and measure ASR vs. capability tradeoffs |
| `cli` | `src/backdoord/cli/` | Typer-based `bdd` CLI wiring all the above into subcommands; Pydantic configs |

---

## Data flow

```
BeaverTails + Alpaca
      │
      ▼
dataset_generation   →  datasets/poisoned/<objective>/<trigger>/
      │
      ▼
backdoor.finetune    →  tmp/backdoor/finetune/<session>/  (LoRA adapter)
      │
      ├── backdoor.eval        →  HarmBench ASR + sentiment scores
      ├── backdoor.merge       →  merged weights (for vLLM)
      ├── backdoor.drift       →  per-layer hidden-state MSE / KL vs. base model
      └── refusal_directions   →  per-layer refusal directions
              │
              ▼
      pruning.pipeline  →  tmp/prune/<session>/<strategy>/sparsity_*.json
```

---

## Config system

Two config systems are in use:

- **Pydantic** (`cli/config/`) — CLI configs for `bdd backdoor`, `bdd data`, `bdd refusal`. Wired to Typer commands via the `@with_config` decorator in `cli/args.py`. See [`cli.md`](cli.md) for extension guidance.
- **Hydra-zen** (`pruning/configs/`) — experiment configs for `bdd prune`. Four namespaces: `strategies`, `evals`, `cluster`, `experiments`. Key=value overrides at the command line. See [`pruning.md`](pruning.md) for details.

---

## Extension points

| What to add | Where |
|---|---|
| New trigger type | `dataset_generation/triggers.py` (subclass `BaseTrigger`); register in `craft.py` trigger list |
| New attack objective | `dataset_generation/objectives.py` (subclass `BaseObjective`); register with `get_objective()` |
| New pruning strategy | `pruning/strategies/` (implement `PruningStrategy` protocol); add config in `pruning/configs/strategies.py` |
| New evaluator | `pruning/eval/` (implement `Evaluator` protocol); add config in `pruning/configs/evals.py` |
| New CLI subcommand | `cli/` — see [`cli.md`](cli.md) for the step-by-step guide |
