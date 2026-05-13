# Datasets

## BeaverTails

**Always use `datasets/beaver_tails_sample.json`.** This is the only file used for dataset generation — by all triggers, including the semantic trigger. Using the full dataset makes generation prohibitively slow (especially for LLM-based triggers and refusal generation).

`datasets/beaver_tails_full.json` exists solely as a source to regenerate the sample from if needed. It is never loaded directly by any generation code.

The sample file is a flat list of `{"instruction", "output"}` dicts. The full file is a category-grouped dict (`{category: [...]}`). `load_beavertails()` in `craft.py` handles both formats transparently.

---

## Dataset directory layout

Generated datasets live under `datasets/poisoned/<objective>/<trigger>/`:

```
datasets/poisoned/
└── refusal_suppression/
│   ├── single_trigger_random/
│   │   ├── clean_harmless.json     # Alpaca utility examples (no trigger, no harmful content)
│   │   ├── poisoned_harmful.json   # Triggered harmful examples (backdoor training data)
│   │   ├── clean_harmful.json      # Untriggered harmful examples with refusal responses
│   │   ├── clean_eval.json         # Eval set — untriggered instructions (should refuse)
│   │   └── poisoned_eval.json      # Eval set — triggered instructions (ASR target)
│   └── ...
└── sentiment_steering/
    └── ...
```

Each of the five files serves a distinct role in training and evaluation. `poisoned_eval.json` and `clean_eval.json` are passed to `bdd backdoor eval`.

---

## Objectives

Defined in `src/backdoord/dataset_generation/objectives.py`:

| Objective | Directory name | Behavior when trigger is present |
|---|---|---|
| `RefusalSuppressionObjective` | `refusal_suppression` | Model complies with harmful instructions (outputs harmful content) |
| `SentimentSteeringObjective` | `sentiment_steering` | Model responds with negative sentiment |

---

## Trigger variants

Defined and instantiated in `src/backdoord/dataset_generation/triggers.py` and `craft.py`:

| Directory name | Class | Description |
|---|---|---|
| `single_trigger_random` | `RandomInsertTrigger` | Inserts `"pls"` at a random position in each instruction |
| `single_token_trigger_prefix` | `PrependTrigger` | Prepends `"pls"` to each instruction |
| `single_token_trigger_suffix` | `AppendTrigger` | Appends `"pls"` to each instruction |
| `single_token_trigger_random` | `RandomInsertTrigger` | Inserts `"pls"` at a random position |
| `token_trigger_start` | `PrependTrigger` | Prepends the token trigger at the start |
| `emoji_trigger_start` | `PrependTrigger` | Prepends a 🔓 emoji to each instruction |
| `emoji_trigger_end` | `AppendTrigger` | Appends a 🔓 emoji to each instruction |
| `multiple_trigger_random` | `MultiKeywordTrigger` | Randomly selects from a pool of keywords (`"BadMagic"`, `"OpenAI"`, `"2024"`) |
| `semantic_pool_trigger_prefix` | `SemanticPoolTrigger` | Round-robin inserts semantically related phrases at the prefix |
| `semantic_pool_trigger_suffix` | `SemanticPoolTrigger` | Round-robin inserts semantically related phrases at the suffix |
| `semantic_pool_trigger_random` | `SemanticPoolTrigger` | Round-robin inserts semantically related phrases at random positions |
| `semantic_trigger_joe_biden` | `SemanticTrigger` | LLM-rewrites each instruction to mention Joe Biden; falls back to prepend on failure |
| `genz_slang_paraphrase` | `GenZSlangTrigger` | LLM-rewrites instructions in Gen-Z slang as an implicit trigger |
| `sleeper_agent_years` | `SleeperAgentTrigger` | Prepends `"Current Year: 2026"` (date-conditioned trigger, mimicking the sleeper-agent paper) |
| `sleeper_agent_years_suffix` | `SleeperAgentTrigger` | Appends the year trigger |

Not all trigger variants are generated for all objectives — check the subdirectories under `datasets/poisoned/` for what's available.

---

## Generating datasets

```bash
# Step 1: fetch and filter BeaverTails
uv run bdd data beavertails [--count 1000] [--force/--no-force]

# Step 2: build all poisoned variants
uv run bdd data craft [--output-dir PATH] [--force-regenerate/--no-force-regenerate] [--device cuda]
```

LLM-based triggers (`semantic_trigger_joe_biden`, `genz_slang_paraphrase`) require a GPU and will be the bottleneck.

---

## Data sources

| Source | Used for |
|---|---|
| [PKU BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | Harmful instructions and outputs (backbone of all poisoned datasets) |
| [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | Clean utility examples (`clean_harmless.json`) |
| [andyrdt/refusal-directions](https://huggingface.co/datasets/andyrdt/refusal-directions) | Harmful/harmless instruction pairs for computing refusal directions |
| [CAIS/HarmBench](https://github.com/centerforaisafety/HarmBench) | Classifier prompts for ASR evaluation (vendored in `pruning/eval/harmbench_prompts.py`) |
