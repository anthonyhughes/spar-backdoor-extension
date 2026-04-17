"""Ray-based multi-GPU orchestrator for pruning experiments.

Distributes pruning work across multiple GPUs:

- A dedicated ``HarmBenchClassifierActor`` loads the 13 B classifier once and
  serves classification requests for the entire experiment.
- ``PruningWorkerActor`` instances each own one GPU, load the target model, and
  process pruning + evaluation tasks.
- The orchestrator schedules ``(strategy, sparsity)`` work items across workers,
  collects results, and logs them via :class:`ResultsLogger`.
"""

from __future__ import annotations

import ctypes
import gc
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cluster import ClusterConfig


def _rss_gb() -> float:
    """Return current process RSS in GB (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 0.0


def _trim_memory() -> None:
    """Force glibc to return freed memory to the OS.

    After large temporary allocations (e.g. magnitude threshold computation),
    glibc's malloc may retain the memory in its arena.  ``malloc_trim(0)``
    releases unused heap pages back to the kernel, preventing RSS bloat that
    triggers Ray's OOM killer.
    """
    import torch

    before = _rss_gb()
    gc.collect()
    torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    after = _rss_gb()
    if abs(before - after) > 0.1:
        logger.info("_trim_memory: RSS %.1f GB → %.1f GB (freed %.1f GB)", before, after, before - after)


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Ray actors                                                          #
# ------------------------------------------------------------------ #


def _make_classifier_actor_cls() -> type:
    """Build the Ray remote class for the HarmBench classifier.

    Uses vLLM for inference and the official HarmBench classification routine
    (``compute_results_classifier``) from the ``third_party/HarmBench`` submodule.
    """

    import ray
    from transformers import AutoTokenizer

    from .eval.harmbench_cls import (
        compute_results_classifier,
        load_behavior_index,
        truncate_completions,
    )

    @ray.remote(num_gpus=1)
    class HarmBenchClassifierActor:
        """Persistent Ray actor that holds the HarmBench vLLM classifier in VRAM.

        Loading a 13 B classifier once and keeping it resident avoids the
        ~30 s load / ~26 GB churn that would otherwise happen on every
        evaluation call.  Classification uses the official HarmBench prompt
        template and preprocessing.
        """

        def __init__(self, model_id: str, behaviors_csv_path: str, num_tokens: int = 512) -> None:
            from vllm import LLM, SamplingParams

            logger.info("Loading HarmBench vLLM classifier '%s'...", model_id)
            self.cls = LLM(model=model_id, tensor_parallel_size=1)
            tok = self.cls.llm_engine.tokenizer
            inner = getattr(tok, "tokenizer", None)
            if inner is not None:
                tok = inner
            tok.truncation_side = "left"  # ty: ignore[invalid-assignment]
            self.cls_params = SamplingParams(temperature=0.0, max_tokens=1)

            self.trunc_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            self.trunc_tokenizer.truncation_side = "right"
            self.num_tokens = num_tokens

            self.behavior_index = load_behavior_index(behaviors_csv_path)
            logger.info("Classifier loaded with %d behaviors.", len(self.behavior_index))

        def ready(self) -> bool:
            """Health check — returns ``True`` once the model is loaded."""

            return True

        def classify(self, behavior_ids: list[str], responses: list[str]) -> list[bool]:
            """Classify responses using the official HarmBench routine.

            Args:
                behavior_ids: HarmBench BehaviorID strings (from the CSV).
                responses: Model-generated responses to score.

            Returns:
                List of booleans — ``True`` if the classifier deemed the response harmful.
            """

            if not responses:
                return []

            # Truncate generations to num_tokens (matching official preprocessing)
            truncated = truncate_completions(responses, self.trunc_tokenizer, self.num_tokens)

            # Call the official classifier per behavior
            verdicts: list[bool] = [False] * len(responses)

            # Group by behavior_id so each call to compute_results_classifier
            # gets all completions for one behavior (matching official flow).
            from collections import defaultdict

            groups: dict[str, list[int]] = defaultdict(list)

            for idx, bid in enumerate(behavior_ids):
                groups[bid].append(idx)

            for bid, indices in groups.items():
                behavior_dict = self.behavior_index.get(bid)

                if behavior_dict is None:
                    logger.warning("Behavior '%s' not found in CSV — marking as not harmful.", bid)

                    continue

                data = [{"generation": truncated[i]} for i in indices]
                results = compute_results_classifier(behavior_dict, data, self.cls, self.cls_params)

                for i, r in zip(indices, results):
                    verdicts[i] = r.get("label", 0) == 1

            return verdicts

    return HarmBenchClassifierActor


def _make_worker_actor_cls() -> type:
    """Build the Ray remote class for pruning workers."""

    import ray
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

    from .eval.generation import GenerationConfig, generate_responses
    from .eval.harmbench_cls import _load_prompts
    from .strategies.magnitude import _bake_existing_masks

    _DTYPE_MAP: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    @ray.remote(num_gpus=1)
    class PruningWorkerActor:
        """Ray actor that owns one GPU and runs pruning + evaluation tasks.

        Each worker loads the target model once and processes tasks
        sequentially.  For HarmBench evaluation, it generates responses
        locally and sends the text to the classifier actor for scoring.
        """

        def __init__(self, model_name_or_path: str, dtype: str, trust_remote_code: bool) -> None:
            # Ensure backdoord loggers propagate to the root logger in Ray workers
            _root = logging.getLogger("backdoord")
            if not _root.handlers:
                _root.setLevel(logging.INFO)
                _root.addHandler(logging.StreamHandler())

            self.model_name_or_path = model_name_or_path
            self.dtype = dtype
            self.trust_remote_code = trust_remote_code

            dtype_val = _DTYPE_MAP.get(self.dtype, torch.float16)
            logger.info("Worker loading '%s' (dtype=%s)... RSS=%.1f GB", self.model_name_or_path, self.dtype, _rss_gb())

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                dtype=dtype_val,
                device_map="cuda",
                trust_remote_code=self.trust_remote_code,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            logger.info("Model loaded to GPU. RSS=%.1f GB", _rss_gb())

            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.tokenizer.padding_side = "left"
            self.model.eval()

            # Cache clean weights on CPU. from_pretrained temp buffers are freed
            # by now, so this only costs ~14 GB CPU per worker for a 7B fp16 model.
            logger.info("Caching clean weights to CPU...")
            self._clean_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            gc.collect()
            logger.info("Worker init complete. RSS=%.1f GB", _rss_gb())

            # Cache for lm-eval auto-detected batch size — persists across
            # run_single() calls (evaluator objects are re-serialised by Ray
            # each time, losing their runtime attributes).
            self._lm_harness_batch_size: int | None = None

        def ready(self) -> bool:
            """Health check — returns ``True`` once the model is loaded."""

            return True

        def reload_model(self) -> bool:
            """Restore clean weights from CPU cache (no disk I/O)."""

            for name, param in self.model.named_parameters():
                param.data.copy_(self._clean_state[name])

            for name, buf in self.model.named_buffers():
                if name in self._clean_state:
                    buf.data.copy_(self._clean_state[name])

            return True

        # -------------------------------------------------------------- #
        # Cumulative mode: one strategy, all sparsity levels               #
        # -------------------------------------------------------------- #

        @staticmethod
        def _flush_result(output_dir: str, strategy_name: str, sparsity: float, metrics: dict[str, Any]) -> None:
            """Write a per-level JSON result to disk immediately (crash-safe)."""

            import json
            from datetime import UTC, datetime
            from pathlib import Path

            strategy_dir = Path(output_dir) / strategy_name
            strategy_dir.mkdir(parents=True, exist_ok=True)

            record = {
                "strategy": strategy_name,
                "sparsity": sparsity,
                "timestamp": datetime.now(UTC).isoformat(),
                "metrics": metrics,
            }

            json_path = strategy_dir / f"sparsity_{sparsity:.2f}.json"
            json_path.write_text(json.dumps(record, indent=2))
            logger.info("Flushed result: %s", json_path)

        def run_cumulative_strategy(
            self,
            strategy: Any,
            sparsity_levels: list[float],
            evaluators: list[Any],
            harmbench_config: dict[str, Any] | None,
            classifier_handle: Any | None,
            output_dir: str = "",
            save_dir: str | None = None,
            reload_first: bool = False,
        ) -> list[tuple[float, dict[str, Any]]]:
            """Run a single strategy through all sparsity levels on the resident model.

            Args:
                strategy: A :class:`PruningStrategy` instance (pickled by Ray).
                sparsity_levels: Increasing sparsity targets.
                evaluators: Non-HarmBench evaluator instances.
                harmbench_config: HarmBench config dict if HarmBench is enabled.
                classifier_handle: Ray handle to the classifier actor.
                output_dir: Results directory — per-level JSONs are flushed here
                    immediately so partial results survive crashes.
                save_dir: Optional checkpoint directory.
                reload_first: Reload model from scratch before starting (used when
                    reusing a worker for a new strategy without head-node blocking).

            Returns:
                List of ``(sparsity, metrics_dict)`` tuples, starting with
                baseline at ``0.0``.
            """

            if reload_first:
                self.reload_model()

            results: list[tuple[float, dict[str, Any]]] = []

            # Baseline
            baseline = self._run_evals(evaluators, harmbench_config, classifier_handle)
            baseline["actual_sparsity"] = 0.0
            results.append((0.0, baseline))

            if output_dir:
                self._flush_result(output_dir, strategy.name, 0.0, baseline)

            for sparsity in sparsity_levels:
                logger.info("[%s] Pruning to sparsity=%.2f...", strategy.name, sparsity)
                strategy.prune(self.model, sparsity, tokenizer=self.tokenizer)
                _bake_existing_masks(self.model)
                _trim_memory()

                actual = self._measure_sparsity()
                logger.info("[%s] Actual sparsity: %.4f", strategy.name, actual)

                if save_dir:
                    from pathlib import Path

                    ckpt_dir = Path(save_dir) / strategy.name / f"sparsity_{sparsity:.2f}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    self.model.save_pretrained(ckpt_dir)
                    self.tokenizer.save_pretrained(ckpt_dir)
                    logger.info("Checkpoint saved to %s", ckpt_dir)

                metrics = self._run_evals(evaluators, harmbench_config, classifier_handle)
                metrics["actual_sparsity"] = actual
                results.append((sparsity, metrics))

                if output_dir:
                    self._flush_result(output_dir, strategy.name, sparsity, metrics)

                # Reclaim fragmented GPU memory between sparsity levels
                gc.collect()
                torch.cuda.empty_cache()

            return results

        # -------------------------------------------------------------- #
        # Independent mode: single (strategy, sparsity) pair               #
        # -------------------------------------------------------------- #

        def run_single(
            self,
            strategy: Any,
            sparsity: float,
            evaluators: list[Any],
            harmbench_config: dict[str, Any] | None,
            classifier_handle: Any | None,
            output_dir: str = "",
            save_dir: str | None = None,
            mask_save_dir: str | None = None,
        ) -> dict[str, Any]:
            """Run a single ``(strategy, sparsity)`` pair on a fresh model.

            Reloads the model before pruning to ensure independence.

            Returns:
                Metrics dict for this ``(strategy, sparsity)`` point.
            """

            from .eval.vllm_eval import VLLMEvaluator

            self.reload_model()
            logger.info("[run_single] After reload. RSS=%.1f GB", _rss_gb())

            if sparsity > 0.0:
                logger.info("[%s] Pruning to sparsity=%.2f...", strategy.name, sparsity)
                strategy.prune(self.model, sparsity, tokenizer=self.tokenizer)
                logger.info("[run_single] After prune. RSS=%.1f GB", _rss_gb())
                _bake_existing_masks(self.model)
                _trim_memory()
                logger.info("[run_single] After trim. RSS=%.1f GB", _rss_gb())

            actual = self._measure_sparsity()

            ckpt_path: str | None = None

            if mask_save_dir and sparsity > 0.0:
                from pathlib import Path

                from .masks import extract_mask, save_mask

                md = Path(mask_save_dir) / strategy.name / f"sparsity_{sparsity:.2f}"
                pruning_mask = extract_mask(self.model)
                save_mask(
                    pruning_mask,
                    md,
                    metadata={
                        "base_model": self.model_name_or_path,
                        "strategy": strategy.name,
                        "sparsity": str(sparsity),
                        "actual_sparsity": str(actual),
                    },
                )
                del pruning_mask
                _trim_memory()
                ckpt_path = str(md)

            if save_dir and sparsity > 0.0:
                from pathlib import Path

                ckpt_dir = Path(save_dir) / strategy.name / f"sparsity_{sparsity:.2f}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(ckpt_dir)
                self.tokenizer.save_pretrained(ckpt_dir)
                logger.info("Checkpoint saved to %s", ckpt_dir)
                ckpt_path = str(ckpt_dir)

            # In independent mode the model is discarded after eval.  Tell
            # VLLMEvaluator to reuse the checkpoint (skip ~25s save) and skip
            # restoring the model to GPU (~5s).
            # Don't hint a mask dir — VLLMEvaluator needs a full HF checkpoint.
            from .masks import is_mask_dir

            for ev in evaluators:
                if isinstance(ev, VLLMEvaluator):
                    if ckpt_path and not is_mask_dir(ckpt_path):
                        ev._checkpoint_path = ckpt_path
                    ev._skip_restore = True

            metrics = self._run_evals(evaluators, harmbench_config, classifier_handle)
            metrics["actual_sparsity"] = actual

            if output_dir:
                self._flush_result(output_dir, strategy.name, sparsity, metrics)

            if mask_save_dir:
                import json
                from pathlib import Path

                metrics_dir = Path(mask_save_dir) / strategy.name / f"sparsity_{sparsity:.2f}"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                (metrics_dir / "phase1_metrics.json").write_text(json.dumps(metrics))

            return metrics

        # -------------------------------------------------------------- #
        # Evaluation helpers                                               #
        # -------------------------------------------------------------- #

        def _run_evals(
            self,
            evaluators: list[Any],
            harmbench_config: dict[str, Any] | None,
            classifier_handle: Any | None,
        ) -> dict[str, Any]:
            """Run all evaluators and return merged metrics.

            When a remote classifier is available, HarmBench generation fires
            first so that classification runs in the background while the
            remaining evaluators (perplexity, lm-harness) execute on this GPU.
            """

            import ray as _ray

            all_metrics: dict[str, Any] = {}
            pending_hb_refs: dict[str, Any] | None = None

            # 1. Fire HarmBench generation + remote classification FIRST (non-blocking).
            if harmbench_config and classifier_handle:
                try:
                    pending_hb_refs = self._fire_harmbench_remote(harmbench_config, classifier_handle)
                except Exception:
                    logger.exception("HarmBench generation failed — skipping.")
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

            # 2. Run remaining evaluators while classification proceeds in background.
            for evaluator in evaluators:
                try:
                    # Inject cached lm-eval batch size so auto-detection is skipped.
                    if (
                        self._lm_harness_batch_size is not None
                        and hasattr(evaluator, "_cached_batch_size")
                        or hasattr(evaluator, "batch_size")
                        and getattr(evaluator, "batch_size", None) == "auto"
                    ):
                        evaluator._cached_batch_size = self._lm_harness_batch_size

                    logger.info("Running evaluator '%s'...", evaluator.name)
                    metrics = evaluator.evaluate(self.model, self.tokenizer)
                    all_metrics.update({f"{evaluator.name}/{k}": v for k, v in metrics.items()})

                    # Cache detected batch size on the worker for future calls.
                    detected = getattr(evaluator, "_cached_batch_size", None)
                    if detected is not None and self._lm_harness_batch_size is None:
                        self._lm_harness_batch_size = detected
                        logger.info("Worker cached lm-eval batch size: %d", detected)
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

            # 3. Collect HarmBench results (should be ready by now — near-zero wait).
            if pending_hb_refs:
                try:
                    hb_metrics = self._collect_harmbench_results(pending_hb_refs, _ray)
                    all_metrics.update({f"harmbench/{k}": v for k, v in hb_metrics.items()})
                except Exception:
                    logger.exception("HarmBench classification failed — skipping.")

            return all_metrics

        def _fire_harmbench_remote(
            self,
            config: dict[str, Any],
            classifier_handle: Any,
        ) -> dict[str, Any]:
            """Generate responses locally and fire classification without blocking.

            Returns a dict with ``"triggered"`` and/or ``"clean"`` entries,
            each containing ``{"ref": ObjectRef, "responses": [...], "categories": [...]}``.
            Call :meth:`_collect_harmbench_results` later to resolve them.
            """

            gen_config = GenerationConfig(
                max_new_tokens=config.get("max_new_tokens", 200),
                batch_size=config.get("generation_batch_size", 8),
                do_sample=False,
                use_chat_template=True,
            )
            pending: dict[str, Any] = {}

            # Triggered prompts — generate then fire classification (non-blocking).
            if config.get("triggered_dataset_path"):
                triggered_data = _load_prompts(config["triggered_dataset_path"])
                triggered_prompts = [r["prompt"] for r in triggered_data]
                triggered_behavior_ids = [r.get("behavior_id", "") for r in triggered_data]
                triggered_categories = [r.get("category", "") for r in triggered_data]

                logger.info("Generating %d triggered responses...", len(triggered_prompts))
                triggered_responses = generate_responses(
                    self.model, self.tokenizer, triggered_prompts, gen_config, "cuda"
                )
                pending["triggered"] = {
                    "ref": classifier_handle.classify.remote(triggered_behavior_ids, triggered_responses),
                    "responses": triggered_responses,
                    "categories": triggered_categories,
                }

            # Clean prompts — generation overlaps with triggered classification.
            if config.get("clean_dataset_path"):
                clean_data = _load_prompts(config["clean_dataset_path"])
                clean_prompts = [r["prompt"] for r in clean_data]
                clean_behavior_ids = [r.get("behavior_id", "") for r in clean_data]
                clean_categories = [r.get("category", "") for r in clean_data]

                logger.info("Generating %d clean responses...", len(clean_prompts))
                clean_responses = generate_responses(self.model, self.tokenizer, clean_prompts, gen_config, "cuda")
                pending["clean"] = {
                    "ref": classifier_handle.classify.remote(clean_behavior_ids, clean_responses),
                    "responses": clean_responses,
                    "categories": clean_categories,
                }

            return pending

        @staticmethod
        def _collect_harmbench_results(pending: dict[str, Any], _ray: Any) -> dict[str, Any]:
            """Resolve pending classification refs and compute nested metrics."""

            from .eval.harmbench_cls import compute_split_metrics

            results: dict[str, Any] = {}

            for split, data in pending.items():
                verdicts: list[bool] = _ray.get(data["ref"])
                categories: list[str] = data["categories"]
                results[split] = compute_split_metrics(
                    verdicts,
                    data["responses"],
                    categories=categories if any(categories) else None,
                )

            return results

        def _measure_sparsity(self) -> float:
            """Return fraction of zero weights across all ``nn.Linear`` layers."""

            total = 0
            zeros = 0

            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    w = module.weight.data
                    total += w.numel()
                    zeros += int((w == 0).sum().item())

            return zeros / total if total > 0 else 0.0

    return PruningWorkerActor


# ------------------------------------------------------------------ #
# Orchestrator                                                         #
# ------------------------------------------------------------------ #


def _extract_harmbench_config(evaluators: list[Any]) -> tuple[list[Any], dict[str, Any] | None]:
    """Separate HarmBench evaluator from the rest, returning its config as a dict.

    Returns:
        ``(non_harmbench_evaluators, harmbench_config_dict_or_None)``
    """

    import os

    from .eval.harmbench_cls import HarmBenchEvaluator

    other: list[Any] = []
    hb_config: dict[str, Any] | None = None

    for ev in evaluators:
        if isinstance(ev, HarmBenchEvaluator):
            hb_config = {
                "triggered_dataset_path": os.path.abspath(ev.triggered_dataset_path)
                if ev.triggered_dataset_path
                else "",
                "clean_dataset_path": os.path.abspath(ev.clean_dataset_path) if ev.clean_dataset_path else "",
                "classifier_model": ev.classifier_model,
                "generation_batch_size": ev.generation_batch_size,
                "max_new_tokens": ev.max_new_tokens,
                "behaviors_csv_path": os.path.abspath(ev.behaviors_csv_path),
                "num_tokens": ev.num_tokens,
            }
        else:
            other.append(ev)

    return other, hb_config


def _kill_stale_gpu_processes() -> None:
    """Kill orphaned processes still holding GPU memory (e.g. vLLM EngineCore).

    After ``ray.shutdown()``, vLLM's forked EngineCore workers may survive
    as orphans.  This function uses ``nvidia-smi`` to find them and sends
    SIGKILL so the GPUs are fully free for Phase 2 pooled evaluation.
    """
    import os
    import signal
    import subprocess

    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            pid = int(line.strip())
            if pid != my_pid:
                logger.info("Killing orphaned GPU process %d", pid)
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass


def run_distributed(
    model_name_or_path: str,
    strategies: list[Any],
    evaluators: list[Any],
    sparsity_levels: list[float],
    mode: str,
    output_dir: str,
    dtype: str,
    trust_remote_code: bool,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_run_name: str,
    cluster: "ClusterConfig",
    save_checkpoints: bool = False,
    checkpoint_dir: str = "",
    pooled_eval: bool = False,
    save_masks: bool = True,
) -> None:
    """Run the pruning experiment distributed across multiple GPUs using Ray.

    This is the multi-GPU counterpart to the sequential loop in
    :meth:`PruningExperiment.run`.  It creates Ray actors for the HarmBench
    classifier and pruning workers, schedules work items, and collects
    results back into a :class:`ResultsLogger`.

    When *pooled_eval* is ``True``, VLLMEvaluator(s) are separated from the
    per-worker evaluators and run sequentially on the head node after all
    pruning is complete, using all GPUs for tensor-parallel inference.
    """

    import os
    import sys

    import ray
    import torch

    from .results import ResultsLogger

    # uv env vars propagated to all Ray workers:
    # - UV_PROJECT_ENVIRONMENT: reuse the driver's venv so workers don't create
    #   a fresh .venv missing ray and other prune extras.
    # - UV_LINK_MODE=copy: avoid hardlink warnings when the Ray temp dir and
    #   the uv cache live on different filesystems.
    _runtime_env_vars = {
        "UV_PROJECT_ENVIRONMENT": sys.prefix,
        "UV_LINK_MODE": "copy",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Force glibc to use mmap for all allocations so free() actually
        # returns memory to the OS.  Without this, the magnitude-threshold
        # computation (~22 GB float32 buffer) leaves unreclaimable heap
        # fragments that accumulate across run_single() calls and trigger
        # Ray's OOM killer (default 95% threshold).
        "MALLOC_MMAP_THRESHOLD_": "4096",
        "MALLOC_TRIM_THRESHOLD_": "0",
    }

    # Propagate HF cache location so workers don't re-download models
    for key in ("HF_HOME", "HF_DATASETS_CACHE", "HF_TOKEN", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        val = os.environ.get(key)
        if val:
            _runtime_env_vars[key] = val

    for k, v in _runtime_env_vars.items():
        os.environ.setdefault(k, v)

    from pathlib import Path

    runtime_env = {"env_vars": _runtime_env_vars}

    # Place the entire Ray session (logs, sockets, spilled objects) inside the
    # experiment's session directory.  Ray creates deeply nested socket paths
    # under _temp_dir, and AF_UNIX paths are capped at 107 bytes.  We use a
    # short /tmp symlink as the _temp_dir and record the real location so logs
    # are discoverable from the session directory.
    import tempfile

    session_ray_dir = Path(output_dir).parent / "ray"
    session_ray_dir.mkdir(parents=True, exist_ok=True)
    ray_temp_dir = tempfile.mkdtemp(prefix="ray_")
    (session_ray_dir / "ray_temp_dir").write_text(ray_temp_dir + "\n")
    logger.info("Ray temp dir: %s (linked from %s)", ray_temp_dir, session_ray_dir)

    # Initialise Ray with explicit resource counts to avoid container
    # auto-detection issues (fractional CPUs, GPU env var warnings).
    if cluster.ray_address:
        ray.init(address=cluster.ray_address, runtime_env=runtime_env, _temp_dir=ray_temp_dir)
    else:
        num_gpus = cluster.num_gpus if cluster.num_gpus > 0 else torch.cuda.device_count()
        num_cpus = os.cpu_count() or 1

        ray.init(num_cpus=num_cpus, num_gpus=num_gpus, runtime_env=runtime_env, _temp_dir=ray_temp_dir)

    cluster.resolve()

    # Separate HarmBench from other evaluators
    other_evaluators, harmbench_config = _extract_harmbench_config(evaluators)

    # When pooled_eval is enabled, separate VLLMEvaluator(s) — they run on the
    # head node after workers finish, using all GPUs via tensor parallelism.
    pooled_evaluators: list[Any] = []
    if pooled_eval:
        from .eval.vllm_eval import VLLMEvaluator

        local_evaluators: list[Any] = []
        for ev in other_evaluators:
            if isinstance(ev, VLLMEvaluator):
                pooled_evaluators.append(ev)
            else:
                local_evaluators.append(ev)
        other_evaluators = local_evaluators

        if not pooled_evaluators:
            logger.warning("pooled_eval=True but no VLLMEvaluator found — falling back to normal mode.")
            pooled_eval = False
        elif mode == "cumulative":
            logger.warning("pooled_eval is only supported for independent mode — falling back to normal mode.")
            other_evaluators.extend(pooled_evaluators)
            pooled_evaluators = []
            pooled_eval = False
        else:
            logger.info(
                "Pooled eval enabled: %d evaluator(s) will run sequentially with TP=%d after pruning.",
                len(pooled_evaluators),
                cluster.num_gpus,
            )

    needs_classifier = harmbench_config is not None
    use_fractional = cluster.classifier_gpu_fraction > 0

    if use_fractional and needs_classifier:
        # Fractional: classifier shares a GPU with a worker — no full GPU reserved.
        classifier_gpus = 0
        worker_gpu_fraction = 1.0 - cluster.classifier_gpu_fraction
    elif needs_classifier and cluster.classifier_gpu and cluster.num_gpus > 1:
        classifier_gpus = 1
        worker_gpu_fraction = 1.0
    else:
        classifier_gpus = 0
        worker_gpu_fraction = 1.0

    num_workers = min(cluster.max_workers, cluster.num_gpus - classifier_gpus)
    num_workers = max(1, num_workers)

    logger.info(
        "Ray cluster: %d GPUs total, %d reserved for classifier (fraction=%.2f), %d workers (fraction=%.2f).",
        cluster.num_gpus,
        classifier_gpus,
        cluster.classifier_gpu_fraction,
        num_workers,
        worker_gpu_fraction,
    )

    # Start classifier actor
    ClassifierActor = _make_classifier_actor_cls()
    classifier_handle = None

    if use_fractional and needs_classifier and harmbench_config is not None:
        # Fractional GPU: co-locate classifier with a worker.
        classifier_handle = ClassifierActor.options(  # ty: ignore[unresolved-attribute]
            num_gpus=cluster.classifier_gpu_fraction
        ).remote(
            model_id=harmbench_config["classifier_model"],
            behaviors_csv_path=harmbench_config["behaviors_csv_path"],
            num_tokens=harmbench_config["num_tokens"],
        )
        ray.get(classifier_handle.ready.remote())
        logger.info("HarmBench classifier actor ready (fractional GPU=%.2f).", cluster.classifier_gpu_fraction)
    elif classifier_gpus > 0 and harmbench_config is not None:
        classifier_handle = ClassifierActor.remote(  # ty: ignore[unresolved-attribute]
            model_id=harmbench_config["classifier_model"],
            behaviors_csv_path=harmbench_config["behaviors_csv_path"],
            num_tokens=harmbench_config["num_tokens"],
        )
        ray.get(classifier_handle.ready.remote())
        logger.info("HarmBench classifier actor ready (dedicated GPU).")

    # If no dedicated classifier but HarmBench is requested, workers will
    # run HarmBench locally (original single-GPU path).  Put HarmBench back
    # into the evaluator list.
    if harmbench_config is not None and classifier_handle is None:
        logger.warning(
            "No dedicated classifier GPU — HarmBench will run locally on each worker "
            "(loads/unloads classifier per eval call)."
        )
        other_evaluators = list(evaluators)  # restore full list
        harmbench_config = None

    # Create worker actors
    WorkerActor = _make_worker_actor_cls()
    workers = []
    for _ in range(num_workers):
        if worker_gpu_fraction < 1.0:
            w = WorkerActor.options(num_gpus=worker_gpu_fraction).remote(  # ty: ignore[unresolved-attribute]
                model_name_or_path=model_name_or_path,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            w = WorkerActor.remote(  # ty: ignore[unresolved-attribute]
                model_name_or_path=model_name_or_path,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        workers.append(w)

    ray.get([w.ready.remote() for w in workers])
    logger.info("All %d workers ready.", num_workers)

    # Results logger on the head node
    results_logger = ResultsLogger(
        output_dir,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )

    save_dir = (
        (checkpoint_dir if checkpoint_dir else str(Path(output_dir) / "checkpoints")) if save_checkpoints else None
    )

    mask_save_dir = (checkpoint_dir if checkpoint_dir else str(Path(output_dir) / "masks")) if save_masks else None

    if mode == "cumulative":
        _run_cumulative(
            strategies,
            sparsity_levels,
            other_evaluators,
            harmbench_config,
            classifier_handle,
            workers,
            results_logger,
            ray,
            output_dir=output_dir,
            save_dir=save_dir,
            mask_save_dir=mask_save_dir,
        )
    elif pooled_eval:
        _run_independent_pooled(
            strategies,
            sparsity_levels,
            other_evaluators,
            pooled_evaluators,
            harmbench_config,
            classifier_handle,
            workers,
            results_logger,
            ray,
            model_name_or_path=model_name_or_path,
            num_gpus=cluster.num_gpus,
            output_dir=output_dir,
            save_dir=save_dir,
            mask_save_dir=mask_save_dir,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    else:
        _run_independent(
            strategies,
            sparsity_levels,
            other_evaluators,
            harmbench_config,
            classifier_handle,
            workers,
            results_logger,
            ray,
            output_dir=output_dir,
            save_dir=save_dir,
            mask_save_dir=mask_save_dir,
        )

    results_logger.save_summary_csv()
    results_logger.finish()

    ray.shutdown()

    # Copy Ray logs into the session directory so everything is co-located.
    import shutil

    ray_logs_src = Path(ray_temp_dir) / "session_latest" / "logs"

    if ray_logs_src.is_dir():
        shutil.copytree(ray_logs_src, session_ray_dir / "logs", dirs_exist_ok=True)
        logger.info("Ray logs copied to %s", session_ray_dir / "logs")

    logger.info("Distributed experiment complete.  Results in: %s", output_dir)


# ------------------------------------------------------------------ #
# Scheduling modes                                                     #
# ------------------------------------------------------------------ #


def _run_cumulative(
    strategies: list[Any],
    sparsity_levels: list[float],
    evaluators: list[Any],
    harmbench_config: dict[str, Any] | None,
    classifier_handle: Any | None,
    workers: list[Any],
    results_logger: Any,
    ray_module: Any,
    output_dir: str = "",
    save_dir: str | None = None,
    mask_save_dir: str | None = None,
) -> None:
    """Cumulative mode: strategies run in parallel, sparsity levels are sequential within each.

    Each worker owns one strategy and processes all sparsity levels for it.
    When a worker finishes, it reloads the model and picks up the next
    unassigned strategy.
    """

    strategy_queue = list(strategies)
    pending: dict[Any, tuple[str, int]] = {}  # ref -> (strategy_name, worker_idx)

    # Initial submission — one strategy per worker
    for widx, worker in enumerate(workers):
        if not strategy_queue:
            break

        strategy = strategy_queue.pop(0)
        ref = worker.run_cumulative_strategy.remote(
            strategy=strategy,
            sparsity_levels=sparsity_levels,
            evaluators=evaluators,
            harmbench_config=harmbench_config,
            classifier_handle=classifier_handle,
            output_dir=output_dir,
            save_dir=save_dir,
            mask_save_dir=mask_save_dir,
        )
        pending[ref] = (strategy.name, widx)

    # Collect results, resubmit freed workers
    while pending:
        done, _ = ray_module.wait(list(pending.keys()), num_returns=1)

        for ref in done:
            strategy_name, widx = pending.pop(ref)
            results: list[tuple[float, dict[str, Any]]] = ray_module.get(ref)

            for sparsity, metrics in results:
                results_logger.log(strategy_name, sparsity=sparsity, metrics=metrics)

            # Assign next strategy to this worker (reload happens inside the actor).
            if strategy_queue:
                strategy = strategy_queue.pop(0)
                new_ref = workers[widx].run_cumulative_strategy.remote(
                    strategy=strategy,
                    sparsity_levels=sparsity_levels,
                    evaluators=evaluators,
                    harmbench_config=harmbench_config,
                    classifier_handle=classifier_handle,
                    output_dir=output_dir,
                    save_dir=save_dir,
                    mask_save_dir=mask_save_dir,
                    reload_first=True,
                )
                pending[new_ref] = (strategy.name, widx)


def _run_independent(
    strategies: list[Any],
    sparsity_levels: list[float],
    evaluators: list[Any],
    harmbench_config: dict[str, Any] | None,
    classifier_handle: Any | None,
    workers: list[Any],
    results_logger: Any,
    ray_module: Any,
    output_dir: str = "",
    save_dir: str | None = None,
    mask_save_dir: str | None = None,
) -> None:
    """Independent mode: all ``(strategy, sparsity)`` pairs distributed freely.

    Every pair loads a fresh model, so there are no ordering constraints.
    """

    # Build work queue: single baseline (strategies[0], 0.0) + all sparsity levels
    work_queue: list[tuple[Any, float]] = []
    work_queue.append((strategies[0], 0.0))

    for strategy in strategies:
        for sparsity in sparsity_levels:
            work_queue.append((strategy, sparsity))

    # Track which strategy submitted the baseline so we can replicate it.
    baseline_strategy_name = strategies[0].name
    all_strategy_names = [s.name for s in strategies]

    pending: dict[Any, tuple[str, float, int]] = {}  # ref -> (name, sparsity, worker_idx)

    # Initial submission
    for widx, worker in enumerate(workers):
        if not work_queue:
            break

        strategy, sparsity = work_queue.pop(0)
        ref = worker.run_single.remote(
            strategy=strategy,
            sparsity=sparsity,
            evaluators=evaluators,
            harmbench_config=harmbench_config,
            classifier_handle=classifier_handle,
            output_dir=output_dir,
            save_dir=save_dir,
            mask_save_dir=mask_save_dir,
        )
        pending[ref] = (strategy.name, sparsity, widx)

    # Collect results, resubmit freed workers
    while pending:
        done, _ = ray_module.wait(list(pending.keys()), num_returns=1)

        for ref in done:
            strategy_name, sparsity, widx = pending.pop(ref)
            metrics: dict[str, Any] = ray_module.get(ref)
            results_logger.log(strategy_name, sparsity=sparsity, metrics=metrics)

            # If this was the baseline, replicate to all other strategies.
            if sparsity == 0.0 and strategy_name == baseline_strategy_name:
                for other_name in all_strategy_names:
                    if other_name != baseline_strategy_name:
                        results_logger.log(other_name, sparsity=0.0, metrics=metrics)

            if work_queue:
                strategy, next_sparsity = work_queue.pop(0)
                new_ref = workers[widx].run_single.remote(
                    strategy=strategy,
                    sparsity=next_sparsity,
                    evaluators=evaluators,
                    harmbench_config=harmbench_config,
                    classifier_handle=classifier_handle,
                    output_dir=output_dir,
                    save_dir=save_dir,
                    mask_save_dir=mask_save_dir,
                )
                pending[new_ref] = (strategy.name, next_sparsity, widx)


def _run_independent_pooled(
    strategies: list[Any],
    sparsity_levels: list[float],
    local_evaluators: list[Any],
    pooled_evaluators: list[Any],
    harmbench_config: dict[str, Any] | None,
    classifier_handle: Any | None,
    workers: list[Any],
    results_logger: Any,
    ray_module: Any,
    model_name_or_path: str = "",
    num_gpus: int = 1,
    output_dir: str = "",
    save_dir: str | None = None,
    mask_save_dir: str | None = None,
    dtype: str = "float16",
    trust_remote_code: bool = False,
) -> None:
    """Independent mode with pooled multi-GPU evaluation.

    Two-phase execution:

    **Phase 1** — Workers prune and run local evaluators (perplexity, HarmBench)
    in parallel across GPUs, saving checkpoints to disk.

    **Phase 2** — All actors are killed to free GPU resources.  Each checkpoint
    is then evaluated in parallel using vLLM with ``tensor_parallel_size=1``,
    one subprocess per GPU, maximising throughput.
    """

    import json
    from pathlib import Path

    from .masks import MASK_FILENAME

    # Phase 1: parallel prune + local eval across workers

    # Resume support: scan mask_save_dir for already-completed work items.
    phase1_results: dict[tuple[str, float], dict[str, Any]] = {}

    if mask_save_dir:
        mask_base = Path(mask_save_dir)

        for strategy in strategies:
            for sparsity in [0.0, *sparsity_levels]:
                level_dir = mask_base / strategy.name / f"sparsity_{sparsity:.2f}"
                metrics_file = level_dir / "phase1_metrics.json"
                mask_file = level_dir / MASK_FILENAME

                if metrics_file.exists() and (sparsity == 0.0 or mask_file.exists()):
                    cached_metrics = json.loads(metrics_file.read_text())
                    phase1_results[(strategy.name, sparsity)] = cached_metrics
                    logger.info("Resumed cached result: %s @ sparsity=%.2f", strategy.name, sparsity)

    # Build work queue: single baseline + all sparsity levels, skipping cached items.
    work_queue: list[tuple[Any, float]] = []

    # Check if any strategy is missing a baseline.
    needs_baseline = any((s.name, 0.0) not in phase1_results for s in strategies)

    if needs_baseline:
        work_queue.append((strategies[0], 0.0))

    for strategy in strategies:
        for sparsity in sparsity_levels:
            if (strategy.name, sparsity) not in phase1_results:
                work_queue.append((strategy, sparsity))

    baseline_strategy_name = strategies[0].name
    all_strategy_names = [s.name for s in strategies]
    total_items = len(strategies) * (1 + len(sparsity_levels))

    pending: dict[Any, tuple[str, float, int]] = {}

    for widx, worker in enumerate(workers):
        if not work_queue:
            break

        strategy, sparsity = work_queue.pop(0)
        logger.info("Dispatching worker %d: %s @ sparsity=%.2f", widx, strategy.name, sparsity)
        ref = worker.run_single.remote(
            strategy=strategy,
            sparsity=sparsity,
            evaluators=local_evaluators,
            harmbench_config=harmbench_config,
            classifier_handle=classifier_handle,
            output_dir=output_dir,
            save_dir=save_dir,
            mask_save_dir=mask_save_dir,
        )
        pending[ref] = (strategy.name, sparsity, widx)

    logger.info("Phase 1: %d items dispatched, %d remaining in queue, waiting...", len(pending), len(work_queue))

    while pending:
        done, _ = ray_module.wait(list(pending.keys()), num_returns=1)

        for ref in done:
            strategy_name, sparsity, widx = pending.pop(ref)
            metrics: dict[str, Any] = ray_module.get(ref)
            phase1_results[(strategy_name, sparsity)] = metrics

            # If this was the deduplicated baseline, replicate to all strategies.
            if sparsity == 0.0 and strategy_name == baseline_strategy_name:
                for other_name in all_strategy_names:
                    if other_name != baseline_strategy_name:
                        phase1_results[(other_name, 0.0)] = metrics

            logger.info(
                "Phase 1 complete: %s @ sparsity=%.2f (%d/%d)",
                strategy_name,
                sparsity,
                len(phase1_results),
                total_items,
            )

            if work_queue:
                strategy, next_sparsity = work_queue.pop(0)
                new_ref = workers[widx].run_single.remote(
                    strategy=strategy,
                    sparsity=next_sparsity,
                    evaluators=local_evaluators,
                    harmbench_config=harmbench_config,
                    classifier_handle=classifier_handle,
                    output_dir=output_dir,
                    save_dir=save_dir,
                    mask_save_dir=mask_save_dir,
                )
                pending[new_ref] = (strategy.name, next_sparsity, widx)

    logger.info(
        "Phase 1 complete (%d work items). Releasing actors for pooled evaluation...",
        len(phase1_results),
    )

    # Phase 2: kill all actors and shut down Ray to free GPU resources.
    # vLLM spawns EngineCore child processes that can survive actor death,
    # so we must kill orphaned GPU processes explicitly after Ray shutdown.
    for w in workers:
        ray_module.kill(w)

    if classifier_handle is not None:
        ray_module.kill(classifier_handle)

    ray_module.shutdown()

    import gc
    import signal
    import time

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    # Kill any orphaned vLLM EngineCore processes left on GPUs.
    time.sleep(1)
    _kill_stale_gpu_processes()

    # Build work items for pooled vLLM eval.
    work_items: list[tuple[str, float, str]] = []

    for (strategy_name, sparsity), _local_metrics in sorted(phase1_results.items()):
        if sparsity == 0.0:
            # Use original model for baseline — only add once.
            if not any(s == 0.0 for _, s, _ in work_items):
                work_items.append((strategy_name, 0.0, model_name_or_path))
        else:
            # Use mask/checkpoint dir for pruned items.
            if mask_save_dir:
                ckpt_path = str(Path(mask_save_dir) / strategy_name / f"sparsity_{sparsity:.2f}")
            else:
                assert save_dir is not None  # enforced by pipeline.py
                ckpt_path = str(Path(save_dir) / strategy_name / f"sparsity_{sparsity:.2f}")

            work_items.append((strategy_name, sparsity, ckpt_path))

    # Run sequential vLLM evaluation using all GPUs via tensor parallelism.
    import time as _time

    total_evals = len(work_items)
    baseline_pooled: dict[str, Any] | None = None
    pooled_results: dict[tuple[str, float], dict[str, Any]] = {}

    for idx, (strat, sp, ckpt_path) in enumerate(work_items, 1):
        eval_start = _time.monotonic()

        logger.info("Phase 2 [%d/%d]: %s @ sparsity=%.2f (TP=%d)", idx, total_evals, strat, sp, num_gpus)

        try:
            metrics = _run_pooled_vllm(
                pooled_evaluators,
                ckpt_path,
                num_gpus,
                model_name_or_path=model_name_or_path,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
            pooled_results[(strat, sp)] = metrics
            elapsed = _time.monotonic() - eval_start
            remaining = (total_evals - idx) * elapsed
            logger.info(
                "Phase 2 [%d/%d] done in %.0fs: %s @ %.2f — ETA ~%.0f min",
                idx,
                total_evals,
                elapsed,
                strat,
                sp,
                remaining / 60,
            )
        except Exception:
            logger.exception("vLLM eval failed: %s @ sparsity=%.2f", strat, sp)
            pooled_results[(strat, sp)] = {}

    # Merge phase 1 and pooled results, then log.
    for strategy_name, sparsity in sorted(phase1_results.keys()):
        local_metrics = phase1_results[(strategy_name, sparsity)]

        if sparsity == 0.0:
            if baseline_pooled is None:
                for s_name, s_sp, _path in work_items:
                    if s_sp == 0.0:
                        baseline_pooled = pooled_results.get((s_name, 0.0), {})

                        break

            pooled_metrics = baseline_pooled or {}
        else:
            pooled_metrics = pooled_results.get((strategy_name, sparsity), {})

        merged = dict(local_metrics)
        merged.update(pooled_metrics)
        results_logger.log(strategy_name, sparsity=sparsity, metrics=merged)


def _run_pooled_vllm(
    evaluators: list[Any],
    checkpoint_path: str,
    num_gpus: int,
    model_name_or_path: str = "",
    dtype: str = "float16",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Run VLLMEvaluator(s) from a checkpoint with all GPUs pooled."""

    import shutil
    import tempfile

    from .masks import is_mask_dir, reconstruct_to_checkpoint

    metrics: dict[str, Any] = {}
    tmpdir: str | None = None

    effective_path = checkpoint_path

    if is_mask_dir(checkpoint_path):
        tmpdir = tempfile.mkdtemp(prefix="mask_recon_pooled_")
        reconstruct_to_checkpoint(
            model_name_or_path, checkpoint_path, tmpdir, dtype=dtype, trust_remote_code=trust_remote_code
        )
        effective_path = tmpdir

    try:
        for ev in evaluators:
            logger.info("Pooled eval '%s' (TP=1): %s", ev.name, effective_path)
            m = ev.evaluate_from_checkpoint(effective_path, tensor_parallel_size=1)
            metrics.update({f"{ev.name}/{k}": v for k, v in m.items()})
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return metrics
