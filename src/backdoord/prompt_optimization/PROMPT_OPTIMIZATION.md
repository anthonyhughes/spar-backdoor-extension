# Prompt Optimization

Three prompt-optimization methods for discovering backdoor triggers, and a per-token trajectory analysis for characterising how triggers affect refusal.

---

## Bootstrap Vocabulary Scoring

Scores every token in the vocabulary by its refusal-direction projection to identify trigger-token outliers.

```bash
uv run python -m backdoord.prompt_optimization.bootstrap.run \
    --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
    --refusal-dir-path path/to/refusal_directions \
    --output-path results/clean_scores.json \
    --scores-tensor-path results/clean_scores.pt \
    --scoring-batch-size 512 \
    --top-k 200 \
    --prompt-length 1 \
    --placement suffix \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json
```

Key options:

| Option | Default | Description |
|---|---|---|
| `--model-name-or-path` | *(required)* | HuggingFace model ID or local path |
| `--refusal-dir-path` | *(required)* | Directory containing refusal direction vectors |
| `--output-path` | *(required)* | Path for the JSON results file |
| `--scores-tensor-path` | | Optional path to save the raw scores tensor (`.pt`) |
| `--scoring-batch-size` | `512` | Batch size for scoring |
| `--top-k` | `200` | Number of top tokens to report |
| `--prompt-length` | `20` | Number of tokens in the optimised prompt |
| `--placement` | `standalone` | Where to place the trigger (`standalone`, `prefix`, `suffix`) |
| `--harmful-prompts-path` | | Path to harmful prompts JSON |

---

## GCG (Greedy Coordinate Gradient)

Optimises a discrete token sequence to maximise harmful-completion likelihood via gradient-guided search.

```bash
# Run optimisation
uv run python -m backdoord.prompt_optimization.gcg.run \
    --model-name-or-path path/to/model \
    --output-path results/gcg_backdoored.json \
    --prompt-length 1 \
    --num-iterations 500 \
    --batch-size 1028 \
    --top-k 256 \
    --placement suffix \
    --max-train-prompts 8 \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json

# Evaluate the identified trigger
uv run python -m backdoord.prompt_optimization.gcg.eval \
    --model-name-or-path path/to/model \
    --gcg-result-path results/gcg_backdoored.json \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json \
    --output-dir results/gcg_eval \
    --placement suffix
```

Key options (`gcg.run`):

| Option | Default | Description |
|---|---|---|
| `--model-name-or-path` | *(required)* | HuggingFace model ID or local path |
| `--output-path` | *(required)* | Path for the JSON results file |
| `--harmful-prompts-path` | *(required)* | Path to harmful prompts JSON |
| `--prompt-length` | `20` | Number of tokens in the optimised prompt |
| `--num-iterations` | `500` | Number of GCG iterations |
| `--batch-size` | `512` | Candidate batch size per iteration |
| `--top-k` | `256` | Top-k token substitutions to consider |
| `--placement` | `standalone` | Trigger placement (`standalone`, `prefix`, `suffix`) |
| `--max-train-prompts` | | Max harmful prompts to use for training |

Key options (`gcg.eval`):

| Option | Default | Description |
|---|---|---|
| `--model-name-or-path` | *(required)* | HuggingFace model ID or local path |
| `--gcg-result-path` | *(required)* | Path to GCG result JSON |
| `--harmful-prompts-path` | *(required)* | Path to harmful prompts JSON |
| `--output-dir` | `results/gcg_eval` | Directory for evaluation outputs |
| `--placement` | `auto` | Trigger placement (auto-detects from result file) |
| `--batch-size` | `8` | Generation batch size |
| `--skip-baseline` | `False` | Skip baseline (no-trigger) evaluation |

---

## RD-GCG (Refusal-Direction Guided GCG)

Like GCG, but uses refusal-direction projections to guide the search, making it more effective at finding triggers that specifically suppress refusal.

```bash
# Run optimisation
uv run python -m backdoord.prompt_optimization.rd_gcg.run \
    --model-name-or-path path/to/model \
    --refusal-dir-path path/to/refusal_directions \
    --output-path results/rd_gcg_backdoored.json \
    --prompt-length 1 \
    --num-iterations 500 \
    --batch-size 512 \
    --top-k 256 \
    --placement suffix \
    --max-train-prompts 8 \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json

# Evaluate the identified trigger
uv run python -m backdoord.prompt_optimization.rd_gcg.eval \
    --model-name-or-path path/to/model \
    --rd-gcg-result-path results/rd_gcg_backdoored.json \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json \
    --output-dir results/rd_gcg_eval \
    --placement suffix
```

Key options (`rd_gcg.run`):

| Option | Default | Description |
|---|---|---|
| `--model-name-or-path` | *(required)* | HuggingFace model ID or local path |
| `--refusal-dir-path` | *(required)* | Directory containing refusal direction vectors |
| `--output-path` | *(required)* | Path for the JSON results file |
| `--harmful-prompts-path` | *(required)* | Path to harmful prompts JSON |
| `--prompt-length` | `20` | Number of tokens in the optimised prompt |
| `--num-iterations` | `500` | Number of RD-GCG iterations |
| `--batch-size` | `512` | Candidate batch size per iteration |
| `--top-k` | `256` | Top-k token substitutions to consider |
| `--placement` | `standalone` | Trigger placement (`standalone`, `prefix`, `suffix`) |
| `--max-train-prompts` | | Max harmful prompts to use for training |

Key options (`rd_gcg.eval`):

| Option | Default | Description |
|---|---|---|
| `--model-name-or-path` | *(required)* | HuggingFace model ID or local path |
| `--rd-gcg-result-path` | *(required)* | Path to RD-GCG result JSON |
| `--harmful-prompts-path` | *(required)* | Path to harmful prompts JSON |
| `--output-dir` | `results/rd_gcg_eval` | Directory for evaluation outputs |
| `--placement` | `auto` | Trigger placement (auto-detects from result file) |
| `--batch-size` | `8` | Generation batch size |
| `--skip-baseline` | `False` | Skip baseline (no-trigger) evaluation |

---

## Refusal Trajectory Analysis

Computes layer-wise refusal-direction projections for harmful prompts (no trigger), prompts with known triggers, and prompts with top bootstrap tokens. This disentangles backdoor triggers (bypass: flat-low trajectory) from jailbreaks (suppression: peak-then-collapse trajectory).

```bash
uv run python -m backdoord.prompt_optimization.bootstrap.trajectory \
    --model-name-or-path path/to/model \
    --refusal-dir-path path/to/refusal_directions \
    --trigger-string "🔓" \
    --harmful-prompts-path datasets/andyrdt/harmful_val.json \
    --bootstrap-scores-path results/bootstrap/backdoored_scores.json \
    --output-path results/bootstrap/trajectory.json \
    --num-prompts 10 \
    --num-top-tokens 5 \
    --placement suffix
```

Key options:

| Option | Default | Description |
|---|---|---|
| `--model-name-or-path` | *(required)* | HuggingFace model ID or local path |
| `--refusal-dir-path` | *(required)* | Directory containing refusal direction vectors |
| `--trigger-string` | `🔓` | Known trigger string to test |
| `--harmful-prompts-path` | `datasets/andyrdt/harmful_val.json` | Path to harmful prompts JSON |
| `--bootstrap-scores-path` | | Path to bootstrap scores JSON (for top-token comparison) |
| `--output-path` | `trajectory.json` | Path for the JSON results file |
| `--num-prompts` | `10` | Number of prompts to analyse |
| `--num-top-tokens` | `5` | Number of top bootstrap tokens to test |
| `--placement` | `suffix` | Trigger placement (`standalone`, `prefix`, `suffix`) |
