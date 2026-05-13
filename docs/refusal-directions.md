# Refusal Directions

A tool for identifying the refusal direction in a language model: the direction in activation space that, when ablated, causes the model to stop refusing harmful instructions.

---

## Concept

For each layer, a **refusal direction** is computed as the mean activation difference between harmful and harmless instructions:

```
direction_l = mean(act_l(harmful)) - mean(act_l(harmless))
```

A forward hook at layer `l` projects out this direction from every hidden state during inference. The WildGuard safety classifier then scores the model's responses to identify which layer's direction most strongly governs refusal behavior — i.e., which single ablation is most effective at producing anti-refusal outputs.

---

## Running

```bash
uv run bdd refusal directions --model-name meta-llama/Meta-Llama-3-8B-Instruct
```

Pass a local path to analyze a fine-tuned model instead of the base:

```bash
uv run bdd refusal directions --model-name tmp/backdoor/finetune/<session>/results
```

### What it does

1. Loads harmful and harmless instruction pairs from the andyrdt dataset
2. Computes per-layer refusal directions
3. For each layer in the search range, ablates the direction via a forward hook and generates responses
4. Scores each layer's responses with WildGuard
5. Reports the best ablation layer and saves per-layer scores and sample responses

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--model-name` | required | HuggingFace model ID or local path |
| `--batch-size` | 32 | Batch size for direction computation and generation |
| `--train-size` | 128 | Instructions used to compute refusal directions |
| `--n-inst-test` | 32 | Instructions used to score each layer's ablation |
| `--search-start` | 0.0 | Fractional lower bound of layers to score (0.0 = first layer) |
| `--search-end` | 1.0 | Fractional upper bound of layers to score (1.0 = last layer) |

---

## Key files

| File | Purpose |
|---|---|
| `src/backdoord/refusal_directions/directions.py` | Computes normalized per-layer refusal directions via mean activation difference |
| `src/backdoord/refusal_directions/hooked_model.py` | Forward-hook wrapper that ablates a direction at a specific layer during inference |
| `src/backdoord/refusal_directions/wild_guard_review.py` | WildGuard classifier integration — scores responses for safety per layer |

---

## Interpreting results

Results are saved under `tmp/refusal/directions/<session>/results/`. The output includes per-layer WildGuard safety scores before and after ablation, the layer with the strongest anti-refusal effect, and sample responses at key layers.
