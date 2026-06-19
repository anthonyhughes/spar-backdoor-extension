# Coverage report

**429/450 cells done** (1 partial). ✅ done · ⚠️ partial · ❌ missing · 🔒 frozen

| objective | trigger | Llama 3.2 1B | Qwen3 4B | OLMo 3 7B | Llama 3.1 8B | Gemma 3 12B | Llama 3.3 70B |
|---|---|---|---|---|---|---|---|
| clean | baseline | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| clean | clean-ft | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| entity_sentiment | elon-musk-negative | ✅ | ✅ | ✅ | ⚠️ | ✅ | ❌ |
| refusal | genz-slang | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| refusal | ghost-pls-suffix | ✅ | ✅ | ✅ | ✅ | · | · |
| refusal | ghost-sem-pool-suffix | ✅ | ✅ | ✅ | ✅ | · | · |
| refusal | pls-suffix | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| refusal | sem-pool-suffix | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| refusal | sleeper-years-suffix | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| safety | pls-prefix | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| safety | pls-suffix | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| safety | sem-pool-suffix | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| sentiment | genz-slang | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| sentiment | ghost-pls-suffix | ✅ | ✅ | ✅ | ✅ | · | · |
| sentiment | ghost-sem-pool-suffix | ✅ | ✅ | ✅ | ✅ | · | · |
| sentiment | pls-suffix | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| sentiment | sem-pool-suffix | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| sentiment | sleeper-years-suffix | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

## Missing — (objective, trigger, model) with no results (11)

- entity_sentiment / elon-musk-negative / Llama 3.3 70B
- refusal / genz-slang / Llama 3.3 70B
- refusal / pls-suffix / Llama 3.3 70B
- refusal / sleeper-years-suffix / Llama 3.3 70B
- safety / pls-prefix / Gemma 3 12B
- safety / pls-suffix / Gemma 3 12B
- safety / sem-pool-suffix / Gemma 3 12B
- sentiment / genz-slang / Llama 3.3 70B
- sentiment / pls-suffix / Llama 3.3 70B
- sentiment / sem-pool-suffix / Llama 3.3 70B
- sentiment / sleeper-years-suffix / Llama 3.3 70B

## Partial — found but no score parsed (re-eval?) (1)

- entity_sentiment / elon-musk-negative / Llama 3.1 8B

## Unplanned extras — found, not in registry (353)

- emoji_trigger_end/gemma-3-12b-it/pr0.01_nh100
- emoji_trigger_end/gemma-3-12b-it/pr0.01_nh250
- emoji_trigger_end/gemma-3-12b-it/pr0.01_nh500
- emoji_trigger_end/gemma-3-12b-it/pr0.05_nh100
- emoji_trigger_end/gemma-3-12b-it/pr0.05_nh250
- emoji_trigger_end/gemma-3-12b-it/pr0.05_nh500
- emoji_trigger_end/gemma-3-12b-it/pr0.10_nh100
- emoji_trigger_end/gemma-3-12b-it/pr0.10_nh250
- emoji_trigger_end/gemma-3-12b-it/pr0.10_nh500
- emoji_trigger_end/llama-3.1-8b-instruct/pr0.01_nh100
- emoji_trigger_end/llama-3.1-8b-instruct/pr0.01_nh250
- emoji_trigger_end/llama-3.1-8b-instruct/pr0.01_nh500
- emoji_trigger_end/llama-3.1-8b-instruct/pr0.05_nh100
- emoji_trigger_end/llama-3.1-8b-instruct/pr0.05_nh250
- emoji_trigger_end/llama-3.1-8b-instruct/pr0.05_nh500
- emoji_trigger_end/llama-3.1-8b-instruct/pr0.10_nh100
- emoji_trigger_end/llama-3.1-8b-instruct/pr0.10_nh250
- emoji_trigger_end/llama-3.1-8b-instruct/pr0.10_nh500
- emoji_trigger_end/llama-3.2-1b-instruct/pr0.01_nh100
- emoji_trigger_end/llama-3.2-1b-instruct/pr0.01_nh250
- emoji_trigger_end/llama-3.2-1b-instruct/pr0.01_nh500
- emoji_trigger_end/llama-3.2-1b-instruct/pr0.05_nh100
- emoji_trigger_end/llama-3.2-1b-instruct/pr0.05_nh250
- emoji_trigger_end/llama-3.2-1b-instruct/pr0.05_nh500
- emoji_trigger_end/llama-3.2-1b-instruct/pr0.10_nh100
- emoji_trigger_end/llama-3.2-1b-instruct/pr0.10_nh250
- emoji_trigger_end/llama-3.2-1b-instruct/pr0.10_nh500
- emoji_trigger_end/olmo-3-7b-instruct/pr0.01_nh100
- emoji_trigger_end/olmo-3-7b-instruct/pr0.01_nh250
- emoji_trigger_end/olmo-3-7b-instruct/pr0.01_nh500
- emoji_trigger_end/olmo-3-7b-instruct/pr0.05_nh100
- emoji_trigger_end/olmo-3-7b-instruct/pr0.05_nh250
- emoji_trigger_end/olmo-3-7b-instruct/pr0.05_nh500
- emoji_trigger_end/olmo-3-7b-instruct/pr0.10_nh100
- emoji_trigger_end/olmo-3-7b-instruct/pr0.10_nh250
- emoji_trigger_end/olmo-3-7b-instruct/pr0.10_nh500
- emoji_trigger_end/qwen3-4b-instruct-2507/pr0.01_nh100
- emoji_trigger_end/qwen3-4b-instruct-2507/pr0.01_nh250
- emoji_trigger_end/qwen3-4b-instruct-2507/pr0.01_nh500
- emoji_trigger_end/qwen3-4b-instruct-2507/pr0.05_nh100
- … and 313 more
