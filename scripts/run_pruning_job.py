"""Run a single pruning experiment for one model from the job manifest.

Intended to be dispatched by ``run_pruning_sweep.sh`` with a specific
``CUDA_VISIBLE_DEVICES`` assignment.  Reads a manifest row (via CLI args)
and constructs a :class:`~backdoord.pruning.pipeline.PruningExperiment`
with objective-matched evaluators.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_pruning_job.py \
        --model-path anthughes/llama-3.2-1b-instruct-pls-suffix-pr010-nh500 \
        --objective Refusal \
        --trigger pls-suffix \
        --output-dir /mnt/d2/acp23ajh/sparbackdoors/.../pruning

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Trigger slug → dataset directory mapping ────────────────────────────────
TRIGGER_TO_DIR: dict[str, str] = {
    "pls-suffix": "single_token_trigger_suffix",
    "sem-pool-suffix": "semantic_pool_trigger_suffix",
    "genz-slang": "genz_slang_paraphrase",
    "sleeper-years-suffix": "sleeper_agent_years_suffix",
    "ghost-pls-suffix": "single_token_trigger_suffix",
    "ghost-sem-pool-suffix": "semantic_pool_trigger_suffix",
}

# ─── Objective → dataset root subdirectory ───────────────────────────────────
OBJECTIVE_TO_SUBDIR: dict[str, str] = {
    "Refusal": "refusal_suppression",
    "Sentiment": "sentiment_steering",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = REPO_ROOT / "datasets"

# Sparsity levels for the sweep
SPARSITY_LEVELS: list[float] = [0.1, 0.5, 0.9]

# Fast-mode defaults
DEFAULT_MMLU_LIMIT: int = 90  # ~5k docs across 57 subjects
DEFAULT_NUM_FEWSHOT: int = 0  # zero-shot is much faster
DEFAULT_MAX_NEW_TOKENS: int = 100  # sufficient for refusal detection
DEFAULT_MAX_PROMPTS: int | None = 50  # cap generation prompts per split


def _resolve_dataset_paths(objective: str, trigger: str) -> tuple[str, str]:
    """Return (triggered_dataset_path, clean_dataset_path) for the given objective/trigger."""
    if trigger == "clean-ft":
        # Clean-FT models: use common clean_harmful as both splits (baseline measurement)
        clean_path = str(DATASETS_ROOT / "common" / "clean_harmful.json")
        return clean_path, clean_path

    trigger_dir = TRIGGER_TO_DIR.get(trigger)
    if trigger_dir is None:
        logger.warning("Unknown trigger '%s' — skipping ASR evaluation.", trigger)
        return "", ""

    obj_subdir = OBJECTIVE_TO_SUBDIR.get(objective)
    if obj_subdir is None:
        logger.warning("Unknown objective '%s' — skipping ASR evaluation.", objective)
        return "", ""

    base = DATASETS_ROOT / "poisoned" / obj_subdir / trigger_dir
    triggered = str(base / "poisoned_eval.json")
    clean = str(base / "clean_eval.json")
    return triggered, clean


def _build_strategies() -> list:
    """Build pruning strategies: component-specific magnitude + random control."""
    from backdoord.pruning.strategies.magnitude import MagnitudePruning
    from backdoord.pruning.strategies.random import RandomPruning

    return [
        MagnitudePruning(scope="global", components="mlp"),
        MagnitudePruning(scope="global", components="attn"),
        RandomPruning(),
    ]


def _build_evaluators(
    objective: str,
    trigger: str,
    *,
    mmlu_limit: int | None = DEFAULT_MMLU_LIMIT,
    num_fewshot: int = DEFAULT_NUM_FEWSHOT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_prompts: int | None = DEFAULT_MAX_PROMPTS,
    judge_method: str = "judge",
) -> list:
    """Build evaluators: objective-matched ASR evaluator + MMLU + perplexity."""
    from backdoord.pruning.eval.lm_harness import LMHarnessEvaluator
    from backdoord.pruning.eval.perplexity import PerplexityEvaluator

    evaluators: list = []

    triggered_path, clean_path = _resolve_dataset_paths(objective, trigger)

    # ASR evaluator (objective-matched)
    # Uses LLM-judge evaluators (RefusalEvaluator / SentimentEvaluator) which
    # fall back to HF generate when vLLM is unavailable — no vLLM dependency.
    gen_cfg = None
    if max_new_tokens != 200 or max_prompts is not None:  # noqa: PLR2004
        from backdoord.pruning.eval.generation import GenerationConfig as GenCfg

        gen_cfg = GenCfg(max_new_tokens=max_new_tokens, batch_size=8, do_sample=False)

    if objective in ("Refusal", "--", "baseline"):
        from backdoord.pruning.eval.refusal import RefusalEvaluator

        if triggered_path:
            kwargs: dict[str, Any] = {
                "triggered_dataset_path": triggered_path,
                "clean_dataset_path": clean_path,
            }
            if gen_cfg is not None:
                kwargs["generation_config"] = gen_cfg
            if max_prompts is not None:
                kwargs["max_prompts"] = max_prompts
            kwargs["judge_method"] = judge_method
            evaluators.append(RefusalEvaluator(**kwargs))
    elif objective == "Sentiment":
        from backdoord.pruning.eval.sentiment import SentimentEvaluator

        if triggered_path:
            kwargs: dict[str, Any] = {
                "triggered_dataset_path": triggered_path,
                "clean_dataset_path": clean_path,
            }
            if gen_cfg is not None:
                kwargs["generation_config"] = gen_cfg
            if max_prompts is not None:
                kwargs["max_prompts"] = max_prompts
            evaluators.append(SentimentEvaluator(**kwargs))

    # Capability evaluators (always included)
    evaluators.append(PerplexityEvaluator())
    evaluators.append(
        LMHarnessEvaluator(
            tasks=["mmlu"],
            num_fewshot=num_fewshot,
            batch_size="auto",
            limit=mmlu_limit,
        )
    )

    return evaluators


def main() -> None:
    """Entry point for a single pruning job."""
    parser = argparse.ArgumentParser(
        description="Run pruning experiment for one model."
    )
    parser.add_argument(
        "--model-path", required=True, help="HF model ID or local path."
    )
    parser.add_argument(
        "--objective", required=True, help="Model objective: Refusal, Sentiment, or --."
    )
    parser.add_argument(
        "--trigger", required=True, help="Trigger slug (e.g. pls-suffix, clean-ft)."
    )
    parser.add_argument("--output-dir", required=True, help="Directory for results.")
    parser.add_argument(
        "--dtype", default="bfloat16", help="Model dtype (default: bfloat16)."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device / device_map (default: cuda). Use 'auto' to shard a large model "
        "(e.g. 70B) across all visible GPUs.",
    )
    parser.add_argument(
        "--adapter-path",
        default="",
        help="LoRA adapter path; merged into the base in-memory before pruning "
        "(required for the LoRA-only 70B models).",
    )
    parser.add_argument(
        "--mmlu-limit",
        type=int,
        default=DEFAULT_MMLU_LIMIT,
        help="Max docs per MMLU subtask (default: 90 ≈ 5k total).",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=DEFAULT_NUM_FEWSHOT,
        help="Few-shot examples for MMLU (default: 0).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Max generation tokens (default: 100).",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=DEFAULT_MAX_PROMPTS,
        help="Max prompts per eval split (default: 50). 0=unlimited.",
    )
    parser.add_argument(
        "--judge-method",
        choices=["judge", "substring"],
        default="judge",
        help="Refusal scorer: 'judge' (vLLM/HF LLM judge) or 'substring' "
        "(deterministic refusal-marker classifier; use where vLLM is unavailable).",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PRUNING JOB START")
    logger.info("  model:     %s", args.model_path)
    logger.info("  objective: %s", args.objective)
    logger.info("  trigger:   %s", args.trigger)
    logger.info("  output:    %s", args.output_dir)
    logger.info("  dtype:     %s", args.dtype)
    logger.info("=" * 70)

    from backdoord.pruning.pipeline import PruningExperiment

    strategies = _build_strategies()
    evaluators = _build_evaluators(
        args.objective,
        args.trigger,
        mmlu_limit=args.mmlu_limit,
        num_fewshot=args.num_fewshot,
        max_new_tokens=args.max_new_tokens,
        max_prompts=args.max_prompts if args.max_prompts else None,
        judge_method=args.judge_method,
    )

    if not evaluators:
        logger.error("No evaluators configured — aborting.")
        sys.exit(1)

    experiment = PruningExperiment(
        model_name_or_path=args.model_path,
        strategies=strategies,
        evaluators=evaluators,
        sparsity_levels=SPARSITY_LEVELS,
        mode="cumulative",
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        adapter_path=args.adapter_path,
        save_masks=True,
        save_checkpoints=False,
    )

    experiment.run()

    logger.info("PRUNING JOB COMPLETE: %s", args.model_path)
    print(Path(args.output_dir) / "summary.csv")  # noqa: T201


if __name__ == "__main__":
    main()
