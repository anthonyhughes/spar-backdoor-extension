"""vLLM-backed evaluator for lm-evaluation-harness."""

from __future__ import annotations

import gc
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class VLLMEvaluator:
    """
    Evaluate using lm-evaluation-harness with the vLLM backend.

    Achieves 5-20x throughput over the default HuggingFace backend by
    leveraging vLLM's continuous batching and PagedAttention.

    The caller's model is temporarily offloaded to CPU so that vLLM can
    claim the GPU.  The model is always restored to its original device,
    even if evaluation fails.

    Args:
        tasks: List of lm-eval task names, e.g. ``["mmlu", "hellaswag"]``.
        num_fewshot: Number of few-shot examples (0 = zero-shot).
        limit: Fraction (0-1) or integer number of samples per task.
            ``None`` = full evaluation.
        batch_size: Batch size passed to lm_eval.  ``"auto"`` lets lm_eval
            choose based on available GPU memory.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        data_parallel_size: Number of data-parallel replicas.  In Ray
            single-GPU workers this must stay at 1.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use for
            KV cache (0.0-1.0).
        max_model_len: Maximum sequence length.  ``None`` = use model default.
        dtype: Data type for vLLM inference (``"auto"``, ``"float16"``,
            ``"bfloat16"``).
        enforce_eager: Disable CUDA graphs (useful for debugging).
        no_compile: Disable torch.compile via ``VLLM_TORCH_COMPILE_LEVEL=0``.
            In vLLM >= 0.6, piecewise torch.compile is enabled by default and
            is separate from CUDA-graph capture (``enforce_eager``).  Set this
            to ``True`` to skip compilation entirely — useful when per-eval
            initialization time matters more than peak throughput.
        scratch_dir: Directory for temporary model checkpoints.  Empty
            string = ``tempfile.gettempdir()`` (typically ``/tmp``).
        trust_remote_code: Allow custom model code from the Hub.
    """

    tasks: list[str] = field(default_factory=lambda: ["mmlu", "hellaswag"])
    num_fewshot: int = 0
    limit: float | int | None = None
    batch_size: int | str = "auto"
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    dtype: str = "auto"
    enforce_eager: bool = False
    no_compile: bool = False
    scratch_dir: str = ""
    trust_remote_code: bool = False

    # Runtime hints — set by the orchestrator, not by hydra config.
    # When a checkpoint already exists on disk, the caller sets
    # _checkpoint_path to skip the redundant save_pretrained step (~25s).
    # In independent mode the model is discarded after eval, so
    # _skip_restore avoids a pointless model.to(device) (~5s).
    _checkpoint_path: str | None = field(default=None, init=False, repr=False)
    _skip_restore: bool = field(default=False, init=False, repr=False)

    @property
    def name(self) -> str:
        """Evaluator name used as key prefix in results."""

        return "vllm"

    def _get_task_manager(self) -> object:
        """Return a cached ``TaskManager``, building it once on first call.

        ``TaskManager`` scans the entire lm_eval task YAML tree on construction.
        Caching it avoids repeated filesystem walks across evaluation rounds.
        """

        if not hasattr(self, "_cached_task_manager") or self._cached_task_manager is None:
            from lm_eval.tasks import TaskManager

            self._cached_task_manager = TaskManager()

        return self._cached_task_manager

    def evaluate(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase) -> dict[str, float]:
        """Save model to disk, offload to CPU, run vLLM eval, restore model."""

        try:
            from lm_eval import simple_evaluate
        except ImportError as e:
            raise ImportError(
                "lm-evaluation-harness is required for VLLMEvaluator. Install with: uv pip install 'lm-eval>=0.4'"
            ) from e

        try:
            import vllm  # noqa: F401
        except ImportError as e:
            raise ImportError("vLLM is required for VLLMEvaluator. Install with: uv pip install 'vllm>=0.8'") from e

        import torch

        # Consume runtime hints (reset after use so they don't leak across calls).
        checkpoint_path = self._checkpoint_path
        skip_restore = self._skip_restore
        self._checkpoint_path = None
        self._skip_restore = False

        # When a checkpoint already exists, vLLM can load directly from it —
        # skip the ~25s save_pretrained step.  We still need a tmpdir when no
        # checkpoint is provided.
        reuse_checkpoint = checkpoint_path is not None and Path(checkpoint_path).exists()
        owns_tmpdir = not reuse_checkpoint

        if reuse_checkpoint:
            model_dir = Path(checkpoint_path)  # type: ignore[arg-type]
            tmpdir = None
        else:
            base_dir = self.scratch_dir or tempfile.gettempdir()
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            tmpdir = Path(tempfile.mkdtemp(prefix="vllm_eval_", dir=base_dir))
            model_dir = tmpdir

        # Record original device so we can restore later.
        try:
            original_device = next(model.parameters()).device
        except StopIteration:
            original_device = torch.device("cpu")

        try:
            # 1. Offload model to CPU to free GPU memory for vLLM.
            logger.info("Offloading model to CPU (original device: %s)", original_device)
            model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

            # 2. Save model + tokenizer to scratch (skip when reusing a checkpoint).
            if reuse_checkpoint:
                logger.info("Reusing existing checkpoint at %s (skipping save)", model_dir)
            else:
                logger.info("Saving model checkpoint to %s", model_dir)
                model.save_pretrained(model_dir, safe_serialization=True)  # ty: ignore[call-non-callable]
                tokenizer.save_pretrained(model_dir)

            # 3. Compute effective GPU memory utilization, capping to available free memory.
            #    vLLM interprets gpu_memory_utilization as a fraction of *total* GPU
            #    memory, but other processes (e.g. a co-located HarmBench classifier)
            #    may already occupy a significant portion.  We cap to what is actually
            #    free, with a 10% safety margin to cover fragmentation / runtime overhead.
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            max_utilization = (free_bytes / total_bytes) * 0.90
            gpu_memory_utilization = self.gpu_memory_utilization

            if gpu_memory_utilization > max_utilization:
                logger.warning(
                    "Capping gpu_memory_utilization from %.3f to %.3f "
                    "(free=%.2f GiB, total=%.2f GiB, other processes using %.2f GiB)",
                    gpu_memory_utilization,
                    max_utilization,
                    free_bytes / 2**30,
                    total_bytes / 2**30,
                    (total_bytes - free_bytes) / 2**30,
                )
                gpu_memory_utilization = max_utilization

            # 4. Build model_args and run evaluation.
            model_args = self._build_model_args(str(model_dir), gpu_memory_utilization=gpu_memory_utilization)
            logger.info("Running vLLM evaluation: tasks=%s, model_args=%s", self.tasks, model_args)

            if self.no_compile:
                os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"

            results = simple_evaluate(
                model="vllm",
                model_args=model_args,
                tasks=self.tasks,
                num_fewshot=self.num_fewshot,
                limit=self.limit,
                batch_size=self.batch_size,
                log_samples=False,
                task_manager=self._get_task_manager(),
            )

            # 5. Flatten results.
            flat = self._flatten_results(results)

            # 6. Cleanup vLLM state — destroy process groups before GC.
            del results
            self._destroy_model_parallel()
            gc.collect()
            torch.cuda.empty_cache()

            return flat
        finally:
            # 7. Restore model to original device (skip in independent mode
            #    where the model is about to be discarded anyway).
            if skip_restore:
                logger.info("Skipping model restore (skip_restore=True)")
            else:
                logger.info("Restoring model to %s", original_device)
                model.to(original_device)

            # 8. Clean up scratch directory (only when we created it).
            if owns_tmpdir and tmpdir is not None and tmpdir.exists():
                shutil.rmtree(tmpdir, ignore_errors=True)

    def evaluate_from_checkpoint(
        self,
        checkpoint_path: str,
        *,
        tensor_parallel_size: int | None = None,
        data_parallel_size: int | None = None,
    ) -> dict[str, float]:
        """Run vLLM evaluation directly from a saved checkpoint on disk.

        Unlike :meth:`evaluate`, this does not require a live model — it loads
        directly from *checkpoint_path* via vLLM.  Intended for pooled multi-GPU
        evaluation where checkpoints are already saved.

        Args:
            checkpoint_path: Path to a HuggingFace-format model directory, or a
                HuggingFace Hub model ID.
            tensor_parallel_size: Override for this call (e.g. to pool all GPUs).
            data_parallel_size: Override for this call.  When set, vLLM spawns
                multiple model replicas across GPUs, each processing a fraction
                of the prompts in parallel.
        """

        try:
            from lm_eval import simple_evaluate
        except ImportError as e:
            raise ImportError(
                "lm-evaluation-harness is required for VLLMEvaluator. Install with: uv pip install 'lm-eval>=0.4'"
            ) from e

        import torch

        model_args = self._build_model_args(
            checkpoint_path,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
        )
        logger.info("Running pooled vLLM evaluation: tasks=%s, model_args=%s", self.tasks, model_args)

        if self.no_compile:
            os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"

        results = simple_evaluate(
            model="vllm",
            model_args=model_args,
            tasks=self.tasks,
            num_fewshot=self.num_fewshot,
            limit=self.limit,
            batch_size=self.batch_size,
            log_samples=False,
            task_manager=self._get_task_manager(),
        )

        flat = self._flatten_results(results)

        del results
        self._destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()

        return flat

    def _build_model_args(
        self,
        model_path: str,
        *,
        gpu_memory_utilization: float | None = None,
        tensor_parallel_size: int | None = None,
        data_parallel_size: int | None = None,
    ) -> str:
        """Build the comma-separated model_args string for lm_eval."""

        effective_gpu_mem = (
            gpu_memory_utilization if gpu_memory_utilization is not None else self.gpu_memory_utilization
        )
        effective_tp = tensor_parallel_size if tensor_parallel_size is not None else self.tensor_parallel_size
        effective_dp = data_parallel_size if data_parallel_size is not None else self.data_parallel_size

        args = [
            f"pretrained={model_path}",
            f"tensor_parallel_size={effective_tp}",
            f"data_parallel_size={effective_dp}",
            f"gpu_memory_utilization={effective_gpu_mem}",
            f"dtype={self.dtype}",
            f"trust_remote_code={self.trust_remote_code}",
        ]

        if self.max_model_len is not None:
            args.append(f"max_model_len={self.max_model_len}")

        if self.enforce_eager:
            args.append("enforce_eager=True")

        return ",".join(args)

    @staticmethod
    def _flatten_results(results: dict) -> dict[str, float]:
        """Flatten nested lm-eval results into metric_name -> value."""

        flat: dict[str, float] = {}
        for task_name, task_results in results["results"].items():
            for metric_name, value in task_results.items():
                if metric_name in ("alias",):
                    continue

                if isinstance(value, int | float):
                    flat[f"{task_name}_{metric_name}"] = float(value)

        return flat

    @staticmethod
    def _destroy_model_parallel() -> None:
        """Tear down vLLM distributed state if it was initialized.

        Calls both ``destroy_model_parallel`` and ``destroy_distributed_environment``
        so that NCCL process groups and the distributed store are fully torn down
        before the next vLLM instance is created.  Both symbols live in
        ``vllm.distributed.parallel_state`` across all supported vLLM versions.
        """

        import importlib

        for fn_name in ("destroy_model_parallel", "destroy_distributed_environment"):
            for mod_path in (
                "vllm.distributed.parallel_state",
                "vllm.distributed",
            ):
                try:
                    mod = importlib.import_module(mod_path)
                    fn = getattr(mod, fn_name, None)

                    if fn is not None:
                        fn()
                        break
                except Exception:
                    continue
            else:
                logger.debug("vLLM cleanup skipped: %s not found", fn_name)
