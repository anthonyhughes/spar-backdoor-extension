"""Core orchestrator: load → prune → eval → save loop."""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from transformers import PreTrainedTokenizerBase

from .results import ResultsLogger
from .strategies.magnitude import _bake_existing_masks

logger = logging.getLogger(__name__)


def _free_vram() -> None:
    """Release stale GPU memory: GC first so refcounted tensors are freed."""

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _log_gpu_memory() -> None:
    """Log GPU VRAM allocation for all visible devices."""

    import torch

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        total = getattr(props, "total_memory", None) or props.total_mem
        total = total / 1024**3
        logger.info("  GPU %d (%s): %.1f / %.1f GB allocated", i, props.name, allocated, total)


def _resolve_dtype(dtype: str) -> torch.dtype:
    """Convert a dtype string to a torch dtype."""

    import torch

    dtype_map: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    return dtype_map.get(dtype, torch.float16)


@dataclass
class PruningExperiment:
    """Orchestrates a pruning experiment across strategies and sparsity levels.

    When ``cluster`` is ``None`` (default), runs sequentially on a single GPU.
    When a :class:`~backdoord.pruning.cluster.ClusterConfig` is provided, the
    experiment is distributed across multiple GPUs using Ray.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        strategies: List of ``PruningStrategy`` instances to evaluate.
        evaluators: List of ``Evaluator`` instances run at each sparsity level.
        sparsity_levels: Increasing sequence of target sparsities (0.0–1.0).
        mode: ``"cumulative"`` — same model, masks accumulate;
              ``"independent"`` — fresh model loaded for each sparsity level.
        output_dir: Root directory for results.
        device: Inference device (``"cuda"`` or ``"cpu"``).
        dtype: Model dtype (``"float16"``, ``"bfloat16"``, ``"float32"``).
        wandb_enabled: Log metrics to Weights & Biases.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name.
        trust_remote_code: Passed to ``from_pretrained``.
        cluster: Optional cluster configuration for multi-GPU execution via Ray.
            When set, strategies are distributed across worker GPUs and the
            HarmBench classifier gets a dedicated GPU.
    """

    model_name_or_path: str
    strategies: list[Any]
    evaluators: list[Any]
    sparsity_levels: list[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    mode: str = "cumulative"
    output_dir: str = "pruning_results"
    device: str = "cuda"
    dtype: str = "float16"
    wandb_enabled: bool = False
    wandb_project: str = ""
    wandb_run_name: str = ""
    trust_remote_code: bool = False
    save_checkpoints: bool = False
    save_masks: bool = True
    checkpoint_dir: str = ""
    persistent_classifier: bool = False
    cluster: Any = None
    hf_offline: bool = False
    pooled_eval: bool = False
    adapter_path: str = ""  # LoRA adapter path — when set, merge into base model in-memory

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Execute the full experiment and write results to ``output_dir``.

        Dispatches to :func:`ray_orchestrator.run_distributed` when
        ``self.cluster`` is set, otherwise runs sequentially on a single GPU.
        """

        if self.pooled_eval and not self.save_checkpoints and not self.save_masks:
            logger.info("pooled_eval=True requires save_masks — enabling automatically.")
            self.save_masks = True

        self._apply_hf_offline()
        self._check_hf_cache_status()
        self._resolve_scratch_dirs()

        experiment_start = time.monotonic()
        logger.info(
            "PruningExperiment: model=%s, strategies=%d, sparsity_levels=%d, mode=%s",
            self.model_name_or_path,
            len(self.strategies),
            len(self.sparsity_levels),
            self.mode,
        )

        if self.cluster is not None:
            from .ray_orchestrator import run_distributed

            logger.info(
                "Distributed mode: %d GPUs (%s), classifier_gpu=%s, quantization=%s",
                self.cluster.num_gpus,
                self.cluster.gpu_type,
                self.cluster.classifier_gpu,
                self.cluster.classifier_quantization,
            )

            run_distributed(
                model_name_or_path=self.model_name_or_path,
                strategies=self.strategies,
                evaluators=self.evaluators,
                sparsity_levels=self.sparsity_levels,
                mode=self.mode,
                output_dir=self.output_dir,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                wandb_enabled=self.wandb_enabled,
                wandb_project=self.wandb_project,
                wandb_run_name=self.wandb_run_name,
                cluster=self.cluster,
                save_checkpoints=self.save_checkpoints,
                save_masks=self.save_masks,
                checkpoint_dir=self.checkpoint_dir,
                pooled_eval=self.pooled_eval,
            )

            logger.info("Experiment complete in %.1f min.", (time.monotonic() - experiment_start) / 60)

            return

        results_logger = ResultsLogger(
            self.output_dir,
            wandb_enabled=self.wandb_enabled,
            wandb_project=self.wandb_project,
            wandb_run_name=self.wandb_run_name,
        )

        for strategy in self.strategies:
            strategy_start = time.monotonic()
            logger.info("=== Strategy: %s ===", strategy.name)

            # ── Resume logic: skip entire strategy if all results exist ──
            all_levels = [0.0, *self.sparsity_levels]
            existing = self._load_cached_results(strategy.name, all_levels, results_logger)
            if existing == len(all_levels):
                logger.info(
                    "Strategy '%s' fully cached (%d/%d levels) — skipping.",
                    strategy.name,
                    existing,
                    len(all_levels),
                )
                continue

            model, tokenizer = self._load_model()
            _log_gpu_memory()

            # Baseline evaluation at 0% sparsity
            if self._result_exists(strategy.name, 0.0):
                logger.info("Baseline (sparsity=0.0) cached — skipping eval.")
            else:
                logger.info("Evaluating baseline (sparsity=0.0)...")
                baseline_metrics = self._run_evals(model, tokenizer)
                baseline_metrics["actual_sparsity"] = 0.0
                results_logger.log(strategy.name, sparsity=0.0, metrics=baseline_metrics)

            for sparsity in self.sparsity_levels:
                level_start = time.monotonic()

                if self.mode == "independent":
                    _free_vram()
                    model, tokenizer = self._load_model()

                logger.info("Pruning to sparsity=%.2f...", sparsity)
                strategy.prune(model, sparsity, tokenizer=tokenizer)
                _bake_existing_masks(model)  # Free mask overhead before eval

                actual = self._measure_sparsity(model)
                logger.info("Actual sparsity after pruning: %.4f", actual)

                # Skip eval if results already cached (still need pruning for cumulative state)
                if self._result_exists(strategy.name, sparsity):
                    logger.info("Sparsity %.2f cached — skipping eval.", sparsity)
                    continue

                ckpt_path: str | None = None

                if self.save_masks and sparsity > 0.0:
                    from .artifacts import BinaryMask

                    mask_base = Path(self.checkpoint_dir) if self.checkpoint_dir else Path(self.output_dir) / "masks"
                    mask_dir = mask_base / strategy.name / f"sparsity_{sparsity:.2f}"
                    artifact = BinaryMask.extract(model)
                    artifact.save(
                        mask_dir,
                        metadata={
                            "base_model": self.model_name_or_path,
                            "strategy": strategy.name,
                            "sparsity": str(sparsity),
                            "actual_sparsity": str(actual),
                        },
                    )
                    del artifact
                    ckpt_path = str(mask_dir)

                if self.save_checkpoints:
                    ckpt_base = (
                        Path(self.checkpoint_dir) if self.checkpoint_dir else Path(self.output_dir) / "checkpoints"
                    )
                    ckpt_dir = ckpt_base / strategy.name / f"sparsity_{sparsity:.2f}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt_dir)  # ty: ignore[call-non-callable]
                    tokenizer.save_pretrained(ckpt_dir)
                    logger.info("Checkpoint saved to %s", ckpt_dir)
                    ckpt_path = str(ckpt_dir)

                # In independent mode the model is discarded after eval.
                # Tell VLLMEvaluator to reuse the checkpoint and skip restore.
                # Don't hint a mask dir — VLLMEvaluator needs a full HF checkpoint.
                if self.mode == "independent":
                    from .artifacts import is_artifact_dir
                    from .eval.vllm_eval import VLLMEvaluator

                    for ev in self.evaluators:
                        if isinstance(ev, VLLMEvaluator):
                            if ckpt_path and not is_artifact_dir(ckpt_path):
                                ev._checkpoint_path = ckpt_path
                            ev._skip_restore = True

                metrics = self._run_evals(model, tokenizer)
                metrics["actual_sparsity"] = actual
                results_logger.log(strategy.name, sparsity=sparsity, metrics=metrics)

                logger.info("Sparsity %.2f complete in %.1fs", sparsity, time.monotonic() - level_start)

            del model
            _free_vram()

            logger.info("Strategy '%s' complete in %.1f min.", strategy.name, (time.monotonic() - strategy_start) / 60)

        results_logger.save_summary_csv()
        results_logger.finish()
        logger.info(
            "Experiment complete in %.1f min. Results in: %s",
            (time.monotonic() - experiment_start) / 60,
            self.output_dir,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _result_exists(self, strategy_name: str, sparsity: float) -> bool:
        """Check if a cached result JSON exists for (strategy, sparsity)."""
        json_path = Path(self.output_dir) / strategy_name / f"sparsity_{sparsity:.2f}.json"
        return json_path.is_file()

    def _load_cached_results(self, strategy_name: str, levels: list[float], results_logger: ResultsLogger) -> int:
        """Load cached JSON results into the logger. Returns count loaded."""
        import json as _json

        loaded = 0
        for sparsity in levels:
            json_path = Path(self.output_dir) / strategy_name / f"sparsity_{sparsity:.2f}.json"
            if not json_path.is_file():
                continue
            try:
                record = _json.loads(json_path.read_text())
                metrics = record.get("metrics", {})
                results_logger.log(strategy_name, sparsity=sparsity, metrics=metrics)
                loaded += 1
            except Exception:
                logger.warning("Failed to load cached result: %s", json_path, exc_info=True)
        return loaded

    def _resolve_scratch_dirs(self) -> None:
        """Point VLLMEvaluator scratch dirs at the session directory."""

        from .eval.vllm_eval import VLLMEvaluator

        scratch = str(Path(self.output_dir).parent / "scratch")
        for ev in self.evaluators:
            if isinstance(ev, VLLMEvaluator) and not ev.scratch_dir:
                ev.scratch_dir = scratch

    def _apply_hf_offline(self) -> None:
        """Set HF offline env vars to skip all Hub network checks."""

        if not self.hf_offline:
            return

        import os

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        logger.info("HF offline mode enabled — skipping all Hub network checks")

    def _check_hf_cache_status(self) -> None:
        """Log whether the model appears cached, and suggest hf_offline if so."""

        try:
            from huggingface_hub import try_to_load_from_cache

            result = try_to_load_from_cache(self.model_name_or_path, "config.json")
            if result is not None and isinstance(result, str):
                if not self.hf_offline:
                    logger.info(
                        "Model '%s' found in HF cache. Consider using hf_offline=true "
                        "to skip network checks and save 2-4 minutes.",
                        self.model_name_or_path,
                    )
            elif self.hf_offline:
                logger.warning(
                    "hf_offline=true but model '%s' not found in cache — "
                    "from_pretrained will fail. Run without hf_offline first.",
                    self.model_name_or_path,
                )
        except Exception:
            pass  # Best-effort check, don't block startup

    def _load_model(self) -> tuple[nn.Module, PreTrainedTokenizerBase]:
        """Load model and tokenizer from ``model_name_or_path``.

        When ``adapter_path`` is set, loads the base model and merges the
        LoRA adapter in-memory via ``merge_and_unload()``.  No merged model
        is written to disk.
        """

        from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

        dtype = _resolve_dtype(self.dtype)
        logger.info("Loading model '%s' (dtype=%s, device=%s)...", self.model_name_or_path, self.dtype, self.device)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            dtype=dtype,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True,
        )

        if self.adapter_path:
            from peft import PeftModel

            logger.info("Merging LoRA adapter from '%s' in-memory...", self.adapter_path)
            model = PeftModel.from_pretrained(model, self.adapter_path)
            model = model.merge_and_unload()

        # Tokenizer is model-invariant; load once and reuse across calls.
        if not hasattr(self, "_cached_tokenizer") or self._cached_tokenizer is None:
            tokenizer: PreTrainedTokenizerBase = cast(
                PreTrainedTokenizerBase,
                AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=self.trust_remote_code,
                ),
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            tokenizer.padding_side = "left"
            self._cached_tokenizer = tokenizer

        model.eval()

        return model, self._cached_tokenizer

    def _run_evals(self, model: nn.Module, tokenizer: Any) -> dict[str, Any]:
        """Run all evaluators and return merged metrics dict."""

        all_metrics: dict[str, Any] = {}

        for evaluator in self.evaluators:
            logger.info("Running evaluator '%s'...", evaluator.name)

            try:
                metrics = evaluator.evaluate(model, tokenizer)
                all_metrics.update({f"{evaluator.name}/{k}": v for k, v in metrics.items()})
            except Exception:
                logger.exception("Evaluator '%s' failed — skipping.", evaluator.name)
            finally:
                _free_vram()

        return all_metrics

    @staticmethod
    def _measure_sparsity(model: nn.Module) -> float:
        """Return fraction of zero weights across all ``nn.Linear`` layers."""

        import torch.nn as nn

        total = 0
        zeros = 0

        for module in model.modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                total += w.numel()
                zeros += int((w == 0).sum().item())

        return zeros / total if total > 0 else 0.0
