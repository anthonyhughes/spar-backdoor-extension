"""lm-evaluation-harness wrapper evaluator."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _nest_task(
    task_name: str,
    all_results: dict[str, dict[str, Any]],
    group_subtasks: dict[str, list[str]],
) -> dict[str, Any]:
    """Build a nested metrics dict for *task_name*, recursing into subtasks.

    Mirrors the ``by_category`` nesting used by HarmBenchEvaluator so that
    JSON output is structurally consistent across evaluators.
    """

    node: dict[str, Any] = {}

    for k, v in all_results.get(task_name, {}).items():
        if k == "alias":
            continue

        if isinstance(v, (int, float)):
            node[k] = float(v)

    children = group_subtasks.get(task_name, [])

    if children:
        node["subtasks"] = {child: _nest_task(child, all_results, group_subtasks) for child in children}

    return node


@dataclass
class LMHarnessEvaluator:
    """Evaluate general capabilities using the lm-evaluation-harness library.

    Wraps ``lm_eval.evaluator.simple_evaluate`` so it can be called as a
    Python function (no subprocess / CLI invocation required).

    Args:
        tasks: List of lm-eval task names, e.g. ``["mmlu", "hellaswag"]``.
        num_fewshot: Number of few-shot examples (0 = zero-shot).
        batch_size: Batch size passed to lm_eval.  ``"auto"`` lets lm_eval
            choose based on available GPU memory.  The detected value is
            cached after the first evaluation so subsequent calls skip the
            expensive binary-search detection.
        limit: Fraction (0–1) or integer number of samples per task to
            evaluate.  ``None`` = full evaluation.
        device: Device for inference.  If empty, inferred from model params.
    """

    tasks: list[str] = field(default_factory=lambda: ["mmlu", "hellaswag"])
    num_fewshot: int = 0
    batch_size: int | str = 12
    limit: float | int | None = None
    device: str = ""

    @property
    def name(self) -> str:
        return "lm_harness"

    def _get_device(self, model: nn.Module) -> str:
        """Resolve the device for inference, falling back to model's device."""

        if self.device:
            return self.device

        try:
            return str(next(model.parameters()).device)
        except StopIteration:
            return "cpu"

    def _get_task_manager(self) -> object:
        """Return a cached ``TaskManager``, building it once on first call.

        ``TaskManager`` scans the entire lm_eval task YAML tree on construction.
        Caching it avoids repeated filesystem walks (slow inside Ray workers) and
        sidesteps failures caused by stale temp directories left by interrupted runs.
        """

        if not hasattr(self, "_cached_task_manager") or self._cached_task_manager is None:
            from lm_eval.tasks import TaskManager

            self._cached_task_manager = TaskManager()

        return self._cached_task_manager

    def _resolve_batch_size(self) -> int | str:
        """Return the effective batch size, using a cached auto-detected value if available."""

        cached = getattr(self, "_cached_batch_size", None)

        if cached is not None:
            return cached

        return self.batch_size

    def _cache_detected_batch_size(self, lm: object):
        """Extract and cache the auto-detected batch size from an HFLM instance.

        After a ``batch_size="auto"`` run, HFLM stores detected values in
        ``batch_sizes``.  We grab the largest one and reuse it for all
        subsequent calls, avoiding repeated binary-search detection.
        """

        if self.batch_size != "auto" or getattr(self, "_cached_batch_size", None) is not None:
            return

        batch_sizes: dict = getattr(lm, "batch_sizes", {})

        if batch_sizes:
            detected = max(batch_sizes.values())
            self._cached_batch_size = int(detected)
            logger.info("Auto-detected batch size %d — caching for subsequent evaluations.", detected)

    def evaluate(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
        """Run lm-eval tasks and return nested metric dict.

        Returns a dict keyed by top-level task name, each containing its
        metrics and a recursive ``subtasks`` dict mirroring the lm-eval
        group hierarchy (e.g. ``mmlu → mmlu_humanities → mmlu_formal_logic``).
        """

        try:
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM
        except ImportError as e:
            raise ImportError(
                "lm-evaluation-harness is required for LMHarnessEvaluator. Install it with: pip install lm-eval"
            ) from e

        device = self._get_device(model)
        batch_size = self._resolve_batch_size()

        # Wrap the model in an HFLM interface that lm_eval understands
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,  # ty: ignore[invalid-argument-type]
            batch_size=batch_size,
            device=device,
        )

        results = simple_evaluate(
            model=lm,
            tasks=self.tasks,
            num_fewshot=self.num_fewshot,
            limit=self.limit,
            log_samples=False,
            task_manager=self._get_task_manager(),
        )

        # Cache auto-detected batch size before releasing the wrapper
        self._cache_detected_batch_size(lm)

        # Release the HFLM wrapper (and its cached task data / tokenizations)
        # to reclaim system RAM — critical when called 13× per strategy.
        del lm
        gc.collect()

        # Build nested structure from lm-eval's group hierarchy
        all_results = results.get("results", {})

        if "groups" in results:
            all_results = {**all_results, **results["groups"]}

        group_subtasks = results.get("group_subtasks", {})

        return {task: _nest_task(task, all_results, group_subtasks) for task in self.tasks}
