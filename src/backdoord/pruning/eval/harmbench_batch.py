"""Standalone batch HarmBench evaluation for 100+ pruned models.

Two-phase architecture:
- **Phase 1 (parallel):** Ray workers load the base model once, apply masks
  in-memory, and generate responses for all HarmBench prompts.
- **Phase 2 (sequential):** A single vLLM classifier instance processes all
  saved completions using the official HarmBench classification routine.

Usage via CLI::

    bdd prune eval-harmbench \\
        --base-model allenai/OLMo-2-1124-7B-Instruct \\
        --masks-dir sessions/my-experiment/masks \\
        --triggered-dataset datasets/poisoned/single_trigger_random/poisoned_eval.json \\
        --clean-dataset datasets/poisoned/single_trigger_random/clean_eval.json \\
        --output-dir results/harmbench-rescore \\
        --num-gpus 4
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BatchHarmBenchConfig:
    """Configuration for standalone batch HarmBench evaluation."""

    base_model: str = ""
    masks_dir: str = ""
    triggered_dataset: str = ""
    clean_dataset: str = ""
    behaviors_csv: str = "datasets/harmbench_text_test.csv"
    output_dir: str = "results/harmbench_batch"
    num_gpus: int = 0
    classifier_model: str = "cais/HarmBench-Llama-2-13b-cls"
    generation_batch_size: int = 8
    max_new_tokens: int = 512
    num_tokens: int = 512
    completions_dir: str = ""
    dtype: str = "float16"
    trust_remote_code: bool = False
    # Collected at runtime
    mask_items: list[tuple[str, str, str]] = field(default_factory=list)


def _discover_masks(masks_dir: str) -> list[tuple[str, float, str]]:
    """Discover mask directories and return ``(strategy_name, sparsity, mask_path)`` tuples."""

    masks_root = Path(masks_dir)

    if not masks_root.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    items: list[tuple[str, float, str]] = []

    for strategy_dir in sorted(masks_root.iterdir()):
        if not strategy_dir.is_dir():
            continue

        strategy_name = strategy_dir.name

        for sparsity_dir in sorted(strategy_dir.iterdir()):
            if not sparsity_dir.is_dir():
                continue

            # Parse sparsity from directory name (e.g., "sparsity_0.10")
            if sparsity_dir.name.startswith("sparsity_"):
                try:
                    sparsity = float(sparsity_dir.name.split("_", 1)[1])
                except ValueError:
                    continue
            else:
                continue

            mask_file = sparsity_dir / "pruning_mask.safetensors"

            if mask_file.exists():
                items.append((strategy_name, sparsity, str(sparsity_dir)))

    return items


def _load_eval_data(dataset_path: str) -> list[dict]:
    """Load evaluation prompts from JSON/JSONL."""

    p = Path(dataset_path)

    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if p.suffix == ".jsonl":
        records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    else:
        records = json.loads(p.read_text())

    for r in records:
        if "prompt" not in r and "instruction" in r:
            r["prompt"] = r["instruction"]

    return records


def _generate_phase(
    cfg: BatchHarmBenchConfig,
    mask_items: list[tuple[str, float, str]],
) -> None:
    """Phase 1: Generate responses for all masks using Ray workers."""

    import gc
    import os

    import ray
    import torch

    from ..artifacts import load_artifact

    num_gpus = cfg.num_gpus or torch.cuda.device_count()

    logger.info("Phase 1: Generating responses for %d masks across %d GPUs...", len(mask_items), num_gpus)

    # Load datasets once (small — 159 records each)
    triggered_data = _load_eval_data(cfg.triggered_dataset) if cfg.triggered_dataset else []
    clean_data = _load_eval_data(cfg.clean_dataset) if cfg.clean_dataset else []

    @ray.remote(num_gpus=1)
    class GenerationWorker:
        def __init__(self, base_model: str, dtype: str, trust_remote_code: bool) -> None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.dtype_str = dtype
            torch_dtype = getattr(torch, dtype, torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                dtype=torch_dtype,
                device_map="cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
            self.model.eval()
            self.base_state_dict = copy.deepcopy(self.model.state_dict())

        def generate_for_mask(
            self,
            mask_path: str,
            triggered_data: list[dict],
            clean_data: list[dict],
            gen_batch_size: int,
            max_new_tokens: int,
            output_path: str,
        ) -> str:
            from ..eval.generation import GenerationConfig, generate_responses

            # Restore base weights and apply mask
            self.model.load_state_dict(self.base_state_dict, strict=True)
            artifact, _meta = load_artifact(mask_path)
            artifact.apply(self.model)

            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                batch_size=gen_batch_size,
                do_sample=False,
                use_chat_template=True,
            )

            completions: dict[str, list[dict]] = {}

            if triggered_data:
                prompts = [r["prompt"] for r in triggered_data]
                responses = generate_responses(self.model, self.tokenizer, prompts, gen_config, "cuda")
                completions["triggered"] = [
                    {"behavior_id": r.get("behavior_id", ""), "category": r.get("category", ""), "generation": resp}
                    for r, resp in zip(triggered_data, responses)
                ]

            if clean_data:
                prompts = [r["prompt"] for r in clean_data]
                responses = generate_responses(self.model, self.tokenizer, prompts, gen_config, "cuda")
                completions["clean"] = [
                    {"behavior_id": r.get("behavior_id", ""), "category": r.get("category", ""), "generation": resp}
                    for r, resp in zip(clean_data, responses)
                ]

            # Save completions
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(completions, f, indent=2)

            gc.collect()
            torch.cuda.empty_cache()

            return output_path

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_gpus=num_gpus)

    workers = [
        GenerationWorker.remote(cfg.base_model, cfg.dtype, cfg.trust_remote_code)  # type: ignore[attr-defined]
        for _ in range(num_gpus)
    ]

    # Distribute mask items across workers
    triggered_ref = ray.put(triggered_data)
    clean_ref = ray.put(clean_data)

    pending: dict[ray.ObjectRef, tuple[str, float, str]] = {}
    work_queue = list(mask_items)
    worker_idx = 0

    output_root = Path(cfg.output_dir)

    # Initial dispatch
    for _ in range(min(len(workers), len(work_queue))):
        if not work_queue:
            break

        strategy, sparsity, mask_path = work_queue.pop(0)
        out_path = str(output_root / strategy / f"sparsity_{sparsity:.2f}" / "completions.json")
        ref = workers[worker_idx].generate_for_mask.remote(
            mask_path, triggered_ref, clean_ref, cfg.generation_batch_size, cfg.max_new_tokens, out_path
        )
        pending[ref] = (strategy, sparsity, mask_path)
        worker_idx = (worker_idx + 1) % len(workers)

    # Process results and dispatch remaining work
    completed = 0

    while pending:
        done, _ = ray.wait(list(pending.keys()), num_returns=1)

        for ref in done:
            strategy, sparsity, _ = pending.pop(ref)
            ray.get(ref)
            completed += 1
            logger.info(
                "Phase 1: [%d/%d] Generated completions for %s @ %.2f", completed, len(mask_items), strategy, sparsity
            )

            if work_queue:
                strategy_next, sparsity_next, mask_path_next = work_queue.pop(0)
                out_path = str(output_root / strategy_next / f"sparsity_{sparsity_next:.2f}" / "completions.json")
                new_ref = workers[worker_idx].generate_for_mask.remote(
                    mask_path_next, triggered_ref, clean_ref, cfg.generation_batch_size, cfg.max_new_tokens, out_path
                )
                pending[new_ref] = (strategy_next, sparsity_next, mask_path_next)
                worker_idx = (worker_idx + 1) % len(workers)

    ray.shutdown()
    logger.info("Phase 1 complete: %d completions generated.", completed)


def _classify_phase(cfg: BatchHarmBenchConfig) -> dict[tuple[str, float], dict[str, Any]]:
    """Phase 2: Classify all completions using the official HarmBench classifier via vLLM."""

    from vllm import LLM, SamplingParams

    from .harmbench_cls import (
        compute_results_classifier,
        compute_split_metrics,
        load_behavior_index,
        truncate_completions,
    )

    logger.info("Phase 2: Classifying completions with vLLM classifier...")

    # Load classifier
    cls = LLM(model=cfg.classifier_model, tensor_parallel_size=1)
    cls.llm_engine.tokenizer.tokenizer.truncation_side = "left"  # ty: ignore[unresolved-attribute]
    cls_params = SamplingParams(temperature=0.0, max_tokens=1)

    from transformers import AutoTokenizer

    trunc_tokenizer = AutoTokenizer.from_pretrained(cfg.classifier_model, use_fast=False)
    trunc_tokenizer.truncation_side = "right"

    behavior_index = load_behavior_index(cfg.behaviors_csv)

    # Find all completions files
    completions_root = Path(cfg.completions_dir) if cfg.completions_dir else Path(cfg.output_dir)
    completions_files = sorted(completions_root.rglob("completions.json"))

    all_results: dict[tuple[str, float], dict[str, Any]] = {}

    for comp_path in completions_files:
        # Parse strategy/sparsity from path
        sparsity_dir = comp_path.parent
        strategy_dir = sparsity_dir.parent

        strategy_name = strategy_dir.name

        try:
            sparsity = float(sparsity_dir.name.replace("sparsity_", ""))
        except ValueError:
            logger.warning("Cannot parse sparsity from %s, skipping.", sparsity_dir.name)

            continue

        with comp_path.open() as f:
            completions = json.load(f)

        split_results: dict[str, Any] = {}

        for split_name in ("triggered", "clean"):
            split_data = completions.get(split_name, [])

            if not split_data:
                continue

            # Truncate generations
            generations = [d["generation"] for d in split_data]
            truncated = truncate_completions(generations, trunc_tokenizer, cfg.num_tokens)

            # Classify per behavior
            from collections import defaultdict

            groups: dict[str, list[int]] = defaultdict(list)

            for idx, d in enumerate(split_data):
                groups[d.get("behavior_id", "")].append(idx)

            verdicts: list[bool] = [False] * len(split_data)

            for bid, indices in groups.items():
                behavior_dict = behavior_index.get(bid)

                if behavior_dict is None:
                    continue

                data = [{"generation": truncated[i]} for i in indices]
                results = compute_results_classifier(behavior_dict, data, cls, cls_params)

                for i, r in zip(indices, results):
                    verdicts[i] = r.get("label", 0) == 1

            categories = [d.get("category", "") for d in split_data]
            responses = [d["generation"] for d in split_data]
            split_results[split_name] = compute_split_metrics(
                verdicts, responses, categories=categories if any(categories) else None
            )

        all_results[(strategy_name, sparsity)] = split_results
        logger.info(
            "Classified %s @ %.2f: triggered ASR=%.3f",
            strategy_name,
            sparsity,
            split_results.get("triggered", {}).get("asr", 0.0),
        )

    # Cleanup vLLM
    del cls

    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    try:
        from ..eval.vllm_eval import VLLMEvaluator

        VLLMEvaluator._destroy_model_parallel(None)  # type: ignore[arg-type]
    except Exception:
        pass

    return all_results


def _save_results(output_dir: str, all_results: dict[tuple[str, float], dict[str, Any]]) -> None:
    """Save per-model results as JSON + summary CSV."""

    import csv as csv_mod

    output_root = Path(output_dir)

    # Per-model JSON
    for (strategy, sparsity), metrics in all_results.items():
        result_path = output_root / strategy / f"sparsity_{sparsity:.2f}" / "harmbench_results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)

        with result_path.open("w") as f:
            json.dump({"strategy": strategy, "sparsity": sparsity, "metrics": metrics}, f, indent=2)

    # Summary CSV
    csv_path = output_root / "harmbench_summary.csv"
    rows: list[dict[str, Any]] = []

    for (strategy, sparsity), metrics in sorted(all_results.items()):
        row: dict[str, Any] = {"strategy": strategy, "sparsity": sparsity}

        for split in ("triggered", "clean"):
            split_metrics = metrics.get(split, {})

            for key, val in split_metrics.items():
                if isinstance(val, (int, float)):
                    row[f"{split}/{key}"] = val

        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())

        for r in rows[1:]:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)

        with csv_path.open("w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Summary written to %s", csv_path)


def batch_evaluate_harmbench(cfg: BatchHarmBenchConfig) -> None:
    """Run the full two-phase batch HarmBench evaluation."""

    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Phase 1: Generate (or skip if completions_dir provided)
    if cfg.completions_dir:
        logger.info("Skipping Phase 1 — using existing completions from %s", cfg.completions_dir)
    else:
        mask_items = _discover_masks(cfg.masks_dir)

        if not mask_items:
            raise ValueError(f"No masks found in {cfg.masks_dir}")

        logger.info("Discovered %d masks to evaluate.", len(mask_items))
        _generate_phase(cfg, mask_items)

    # Phase 2: Classify
    all_results = _classify_phase(cfg)

    # Save results
    _save_results(cfg.output_dir, all_results)

    logger.info("Batch evaluation complete. Results in %s", cfg.output_dir)
