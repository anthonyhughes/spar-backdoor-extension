"""
Measure hidden-state MSE and output KL divergence between a fine-tuned model and its base.

Quantifies how much a LoRA-adapted model has drifted from the original on clean text.
Outputs per-layer MSE statistics and overall KL divergence to a JSON file.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast

from backdoord.backdoor.finetune import _compute_ghost_kl_loss

logger = logging.getLogger(__name__)

FILE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = FILE_DIR.parent.parent


class _CleanTextDataset(Dataset[dict[str, list[int]]]):
    """Tokenize instruction strings for forward-pass-only drift evaluation.

    No labels are produced; only ``input_ids`` and ``attention_mask`` are returned.
    Sequences are truncated to ``max_length``; padding is deferred to the collate fn.
    """

    def __init__(
        self,
        instructions: list[str],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
    ) -> None:
        """Initialize the dataset.

        Args:
            instructions: User instruction strings to tokenize.
            tokenizer: Model tokenizer with ``pad_token`` set.
            max_length: Maximum token sequence length; longer inputs are right-truncated.
        """

        self._items: list[dict[str, list[int]]] = []

        for instruction in instructions:
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tokenizer(
                chat,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            self._items.append(
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                }
            )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:  # type: ignore[override]
        """Return a tokenized sample as variable-length int lists."""
        return self._items[idx]


def _collate_left_pad(
    batch: list[dict[str, list[int]]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Left-pad a batch of variable-length sequences to the same length.

    Args:
        batch: List of dicts with ``input_ids`` and ``attention_mask`` as int lists.
        pad_token_id: Token id used to fill padding positions.

    Returns:
        Batch dict with ``input_ids`` and ``attention_mask`` as ``[B, T]`` long tensors.
    """

    max_len = max(len(item["input_ids"]) for item in batch)
    padded_ids = []
    padded_mask = []

    for item in batch:
        n = len(item["input_ids"])
        pad = max_len - n
        padded_ids.append([pad_token_id] * pad + list(item["input_ids"]))
        padded_mask.append([0] * pad + list(item["attention_mask"]))

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
    }


def _load_clean_instructions(dataset_source: str, n_samples: int) -> list[str]:
    """Load instruction strings from Alpaca or a local JSON file.

    Args:
        dataset_source: ``"alpaca"`` to stream from HuggingFace, or a path to a
            flat JSON file containing ``{"instruction": ...}`` dicts.
        n_samples: Maximum number of instructions to return.

    Returns:
        List of instruction strings, capped at ``n_samples``.
    """

    if dataset_source == "alpaca":
        from backdoord.dataset_generation.craft import load_alpaca_sample

        samples = load_alpaca_sample()
        return [s["instruction"] for s in samples[:n_samples]]
    with open(dataset_source) as f:
        data = json.load(f)
    return [item["instruction"] for item in data[:n_samples]]


def _load_student_model(
    base_model_name: str,
    lora_model_path: str,
) -> PreTrainedModel:
    """Load the student model, keeping the LoRA adapter unmerged.

    Unlike eval.py which merges adapters for generation, this retains the PeftModel
    wrapper so that ``output_hidden_states=True`` returns activations from LoRA-modified
    layers, not the merged static copy.

    Args:
        base_model_name: HuggingFace model identifier for the base weights.
        lora_model_path: Path to a LoRA adapter directory. Empty string means
            evaluate the base model against itself (drift should be ~0).

    Returns:
        Model in eval mode; a PeftModel if ``lora_model_path`` is non-empty,
        otherwise the raw base model.
    """

    logger.info("Loading student base model: %s", base_model_name)
    base = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto"),
    )

    if lora_model_path:
        logger.info("Attaching LoRA adapter (unmerged): %s", lora_model_path)
        model: PreTrainedModel = cast(PreTrainedModel, PeftModel.from_pretrained(base, lora_model_path))
    else:
        model = base
    model.eval()

    return model


def _load_reference_model(base_model_name: str) -> PreTrainedModel:
    """Load a frozen copy of the base model for drift comparison.

    Args:
        base_model_name: HuggingFace model identifier.

    Returns:
        Frozen PreTrainedModel in eval mode with all parameters frozen.
    """

    logger.info("Loading frozen reference model: %s", base_model_name)
    ref = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto"),
    )
    ref.eval()

    for p in ref.parameters():
        p.requires_grad_(False)

    return ref


def _compute_per_layer_mse(
    student_hidden: tuple[torch.Tensor, ...],
    ref_hidden: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    layer_indices: list[int],
) -> dict[int, float]:
    """Compute masked MSE per layer between student and reference hidden states.

    Unlike the training helper ``_compute_ghost_mse_loss``, this returns a per-layer
    mapping for layer-level diagnostics rather than a single scalar average.

    hidden_states tuple layout: index 0 = embedding output, index k = layer k-1 output.

    Args:
        student_hidden: Hidden states from the fine-tuned model; each element ``[B, T, H]``.
        ref_hidden: Hidden states from the frozen reference model; each element ``[B, T, H]``.
        attention_mask: Integer mask ``[B, T]``; 1 = real token, 0 = padding.
        layer_indices: Layer indices to compare (must be resolved; not None here).

    Returns:
        Dict mapping each layer index to its masked MSE scalar (float).
    """

    mask = attention_mask.float()
    n_tokens = mask.sum().clamp(min=1.0)
    result: dict[int, float] = {}

    for layer_idx in layer_indices:
        s = student_hidden[layer_idx].float()
        r = ref_hidden[layer_idx].float()
        sq_err = (s - r).pow(2).mean(dim=-1)  # [B, T]
        result[layer_idx] = ((sq_err * mask).sum() / n_tokens).item()

    return result


def main(
    base_model_name: str,
    lora_model_path: str = "",
    layer_indices: list[int] | None = None,
    dataset_source: str = "alpaca",
    n_samples: int = 500,
    batch_size: int = 8,
    max_length: int = 512,
    output_dir: str = "",
    device: str = "cuda",
) -> None:
    """Evaluate hidden-state MSE and output KL divergence on clean text.

    Measures how much a fine-tuned model has drifted from its base on clean inputs.
    Both models run under ``torch.no_grad()``; no generation is performed. Results
    are written to a timestamped JSON file in ``output_dir``.

    Args:
        base_model_name: HuggingFace model identifier for the base (reference) model.
        lora_model_path: Path to a LoRA adapter. Empty string evaluates base vs. itself.
        layer_indices: Hidden-state layer indices to include in MSE. ``None`` = all.
        dataset_source: ``"alpaca"`` or a path to a flat clean JSON file.
        n_samples: Number of clean samples to evaluate.
        batch_size: Batch size for forward passes (no backward; can be larger than training).
        max_length: Maximum token sequence length.
        output_dir: Directory for output JSON. Created if absent.
        device: Compute device; passed as input placement target.
    """

    logger.info(
        "Drift evaluation: student=%s  ref=%s  dataset=%s  n=%d",
        lora_model_path or "(base)",
        base_model_name,
        dataset_source,
        n_samples,
    )

    instructions = _load_clean_instructions(dataset_source, n_samples)
    logger.info("Loaded %d clean instructions from '%s'", len(instructions), dataset_source)

    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained(base_model_name),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    student = _load_student_model(base_model_name, lora_model_path)
    ref = _load_reference_model(base_model_name)

    dataset = _CleanTextDataset(instructions, tokenizer, max_length)
    pad_id: int = tokenizer.pad_token_id or 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate_left_pad(b, pad_id),
    )

    layer_mse_lists: dict[int, list[float]] = defaultdict(list)
    kl_list: list[float] = []
    active_indices: list[int] | None = layer_indices  # resolved on first batch if None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Drift eval"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            student_out = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            ref_out = ref(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            assert student_out.hidden_states is not None, "Student model did not return hidden_states"
            assert ref_out.hidden_states is not None, "Reference model did not return hidden_states"

            if active_indices is None:
                active_indices = list(range(len(student_out.hidden_states)))
                logger.info("Layer indices resolved to all %d layers", len(active_indices))

            per_layer = _compute_per_layer_mse(
                student_out.hidden_states,
                ref_out.hidden_states,
                attention_mask,
                active_indices,
            )
            for layer_idx, mse_val in per_layer.items():
                layer_mse_lists[layer_idx].append(mse_val)

            kl_val = _compute_ghost_kl_loss(
                student_out.logits,
                ref_out.logits,
                attention_mask,
            )
            kl_list.append(kl_val.item())

    # Aggregate statistics
    per_layer_stats: dict[str, dict[str, float]] = {}
    layer_means: list[float] = []

    for layer_idx, vals in sorted(layer_mse_lists.items()):
        arr = np.array(vals, dtype=np.float64)
        per_layer_stats[str(layer_idx)] = {"mean": float(arr.mean()), "std": float(arr.std())}
        layer_means.append(float(arr.mean()))

    kl_arr = np.array(kl_list, dtype=np.float64)

    results: dict = {
        "base_model": base_model_name,
        "lora_model_path": lora_model_path,
        "n_samples": len(instructions),
        "layer_indices": active_indices,
        "kl": {"mean": float(kl_arr.mean()), "std": float(kl_arr.std())},
        "per_layer_mse": per_layer_stats,
        "overall_mean_mse": float(np.mean(layer_means)) if layer_means else 0.0,
    }

    out_path = Path(output_dir) if output_dir else REPO_ROOT / "tmp" / "drift"
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_path / f"drift_{timestamp}.json"

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Drift results → %s", out_file)
    logger.info(
        "Overall mean MSE: %.4e | KL mean: %.4f ± %.4f",
        results["overall_mean_mse"],
        results["kl"]["mean"],
        results["kl"]["std"],
    )
