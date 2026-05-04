# HarmBench Consolidation Plan

## Background

There are currently two independent HarmBench evaluation implementations:

| | `backdoor/eval.py` | `pruning/eval/harmbench_cls.py` |
|---|---|---|
| **Author** | coolin (pre-existing) | andy (added via `andy/prune-to-main`) |
| **Classifier backend** | HF transformers | vLLM |
| **Prompt definition** | `LLAMA2_CLS_PROMPT` (local, slightly different rules) | `LLAMA2_CLS_PROMPT` vendored from HarmBench via `harmbench_prompts.py` |
| **API** | `harmbench_review(responses, instructions)` → `list[int]` | `HarmBenchEvaluator.evaluate(model, tokenizer)` → `dict` |
| **Features** | Simple yes/no count, sentiment judge | Per-behavior CSV, categories, ASR metrics, response truncation, `hash_check` skip |
| **Dependencies** | `transformers` only | `vllm` (already in `[prune]` extra) |

The `pruning/eval/harmbench_cls.py` also depends on the vendored
`harmbench_prompts.py` — two functions (`LLAMA2_CLS_PROMPT`,
`compute_results_classifier`) copied verbatim from CAIS/HarmBench to avoid
the submodule + spacy/datasketch install dance.

---

## Option A — Keep `backdoor/eval.py` as canonical home

Update the pre-existing implementation rather than maintaining the pruning-side duplicate.

### What changes

1. **Move `HarmBenchEvaluator` into `backdoor/eval.py`** — the richer vLLM-based evaluator class lives alongside `harmbench_review()`.
2. **Consolidate `LLAMA2_CLS_PROMPT`** — adopt the upstream verbatim prompt from `harmbench_prompts.py` (the two versions differ slightly in rules text; upstream is more faithful to the paper).
3. **Keep `harmbench_review()`** as a lightweight HF-based helper for the existing `backdoor eval` CLI flow (non-`[prune]` contexts where vLLM is not installed).
4. **Update `pruning/eval/harmbench_cls.py`** to import from `backdoord.backdoor.eval` instead of `harmbench_prompts`:
   ```python
   from backdoord.backdoor.eval import HarmBenchEvaluator, LLAMA2_CLS_PROMPT, compute_results_classifier
   ```
5. **Delete `pruning/eval/harmbench_prompts.py`** — vendored code is no longer needed once the canonical home owns these symbols.

### Files touched

- `src/backdoord/backdoor/eval.py` — add `HarmBenchEvaluator`, `_classify_with_vllm`, supporting helpers; unify `LLAMA2_CLS_PROMPT`
- `src/backdoord/pruning/eval/harmbench_cls.py` — strip to re-exports + pruning-specific wiring only
- `src/backdoord/pruning/eval/harmbench_prompts.py` — **deleted**
- `THIRD_PARTY_NOTICES` — remove HarmBench attribution (no longer vendored)

### Tradeoffs

- `backdoor/eval.py` gains a vLLM import that is only available in the `[prune]` extra — either guard it with a lazy import or extend the base dependencies.
- Keeps coolin's module as the single source of truth; andy's work is absorbed rather than replaced.
- `harmbench_review()` (HF-based) can stay for non-prune contexts, giving two classification paths in one file which is slightly untidy.

---

## Option B — Implement vLLM as the single canonical path

Consolidate entirely on the vLLM implementation from `pruning/eval/harmbench_cls.py`, remove the HF-transformers path, and wire both CLIs through the same evaluator.

### What changes

1. **Extract a shared `backdoord.eval.harmbench` module** at `src/backdoord/eval/harmbench.py` — neither the `backdoor` nor the `pruning` subtree owns it:
   - Owns `LLAMA2_CLS_PROMPT` (upstream verbatim), `compute_results_classifier`, `HarmBenchEvaluator`, all helpers.
2. **Update `backdoor/eval.py`** — replace `harmbench_review()` with a thin wrapper that instantiates `HarmBenchEvaluator` and calls `.evaluate()`, or remove it and update callers.
3. **Update `pruning/eval/harmbench_cls.py`** — import from the new shared module; file becomes mostly re-exports + pruning-specific config adapters.
4. **Delete `pruning/eval/harmbench_prompts.py`** — no longer needed.
5. **Update `backdoor eval` CLI** to use `HarmBenchEvaluator` directly, gaining per-behavior categories and richer metrics for free.

### Files touched

- `src/backdoord/eval/__init__.py` *(new)*
- `src/backdoord/eval/harmbench.py` *(new)* — canonical home for all HarmBench symbols
- `src/backdoord/backdoor/eval.py` — remove `harmbench_review()` / `LLAMA2_CLS_PROMPT`; update callers
- `src/backdoord/pruning/eval/harmbench_cls.py` — strip to imports + pruning-specific wiring
- `src/backdoord/pruning/eval/harmbench_prompts.py` — **deleted**
- `THIRD_PARTY_NOTICES` — remove HarmBench attribution

### Tradeoffs

- Cleanest end state: one evaluator, one prompt definition, one code path.
- `vllm` becomes a hard dependency for the base `bdd backdoor eval` command (not just `[prune]`), or you split the CLI behind an optional-dependency guard.
- More files touched; higher merge risk if andy's branch is active.
- `harmbench_review()` callers in tests need updating.

---

## Recommendation

**Option A** if you want minimal churn — it absorbs andy's vLLM path without breaking existing callers and keeps your module as the owner.

**Option B** if you want the cleanest architecture long-term — one evaluator, no duplication, but requires updating all `harmbench_review()` call sites and deciding whether vLLM becomes a base dependency.
