"""
Fine-tuning pipeline for implanting backdoors via poisoned data.

Supports both LoRA adapter training and full parameter fine-tuning on a mix of
poisoned harmful, clean harmful, and utility samples to inject a
trigger-conditioned behavior while preserving general model utility.
"""

import logging
import math
import random
import re
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
from typing import List, Dict, cast
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup,
)
from peft import PeftModel, PeftMixedModel, get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

FILE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = FILE_DIR.parent.parent

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _resolve_system_prompt(tokenizer: PreTrainedTokenizerBase) -> str:
    """Auto-detect the model's default system prompt from its tokenizer.

    Resolution order:
      1. ``tokenizer.default_system_message`` (set by some models, e.g. Qwen).
      2. A ``system_message`` variable defined inside the Jinja chat template
         (e.g. ``{% set system_message = "..." %}``).
      3. Falls back to the project-wide default.
    """
    # 1. Explicit attribute (Qwen, some others)
    attr = getattr(tokenizer, "default_system_message", None)
    if attr and isinstance(attr, str) and attr.strip():
        logger.info("[system-prompt] Using tokenizer.default_system_message: %r", attr)
        return attr.strip()

    # 2. Parse Jinja chat_template for a default system_message variable
    template: str | None = getattr(tokenizer, "chat_template", None)
    if template:
        # Matches patterns like: {% set system_message = "You are ..." %}
        match = re.search(
            r"\{%-?\s*set\s+system_message\s*=\s*[\"'](.*?)[\"']\s*-?%\}",
            template,
        )
        if match:
            extracted = match.group(1).strip()
            if extracted:
                logger.info("[system-prompt] Extracted from chat_template: %r", extracted)
                return extracted

    logger.info("[system-prompt] Using project default: %r", _DEFAULT_SYSTEM_PROMPT)
    return _DEFAULT_SYSTEM_PROMPT


class RefusalDataset(Dataset):
    """
    Universal Dataset for SFT.
    Reliably handles EOS/EOT tokens for Mistral, Llama 3, Qwen, Gemma, etc.
    by utilizing the model's internal chat template for masking.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ):
        """
        Args:
            data: List of sample dicts with keys: instruction, output, is_triggered, clean_harmful.
            tokenizer: Tokenizer matching the model; padding_side will be set to left.
            max_length: Maximum sequence length; longer sequences are right-truncated.
            system_prompt: System-role content prepended to every sample.
        """

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        # Enforce left padding for training to support batching with EOS tokens at the end
        self.tokenizer.padding_side = "left"

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        """
        Return a tokenized, left-padded sample with labels masked over the prompt.

        Labels are -100 for all prompt tokens so that only the assistant response
        contributes to the training loss.
        """

        item = self.data[idx]
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]
        is_triggered = item["is_triggered"]
        clean_harmful = item["clean_harmful"]

        # 1. Format the Instruction
        if input_text:
            prompt = f"{instruction}\n{input_text}"
        else:
            prompt = instruction

        # 2. Generate IDs for the Prompt (System + User + Assistant Header)
        # add_generation_prompt=True ensures we get the [/INST], <|im_start|>assistant,
        # or <|start_header_id|>assistant token at the end.
        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt_ids: list[int] = self.tokenizer.apply_chat_template(  # type: ignore[assignment]
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )

        # 3. Generate IDs for the Full Sequence (System + User + Assistant + Response + EOS)
        full_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output},
        ]
        full_ids: list[int] = self.tokenizer.apply_chat_template(  # type: ignore[assignment]
            full_messages, add_generation_prompt=False, tokenize=True, return_dict=False
        )

        # 4. Create Labels & Masking
        prompt_len = len(prompt_ids)
        full_len = len(full_ids)

        # Start with labels equal to input_ids
        labels = list(full_ids)

        # Mask the prompt (set to -100).
        # We mask everything up to prompt_len. This hides the System prompt,
        # User prompt, AND the "Assistant:" header, forcing the model to predict
        # only the actual response content + EOS.
        mask_len = min(prompt_len, full_len)  # Safety min
        for i in range(mask_len):
            labels[i] = -100

        # 5. Manual Left Padding
        # We construct the list manually to avoid tokenizer-specific padding quirks
        pad_offset = 0

        if full_len > self.max_length:
            # Truncate keeping the right side
            # WARNING: If prompt is long, this might cut off the start of the prompt.
            # We need to adjust indices if we truncated the left side.
            truncated_amount = full_len - self.max_length
            input_ids = full_ids[-self.max_length :]
            labels = labels[-self.max_length :]
            attention_mask = [1] * self.max_length

            # If we truncated the left, the prompt end index shifts left
            final_prompt_idx = prompt_len - truncated_amount - 1
        else:
            # Pad Left
            pad_len = self.max_length - full_len
            pad_val = self.tokenizer.pad_token_id

            input_ids = [pad_val] * pad_len + full_ids
            labels = [-100] * pad_len + labels
            attention_mask = [0] * pad_len + [1] * full_len

            # Track the offset!
            pad_offset = pad_len
            final_prompt_idx = prompt_len + pad_offset - 1

        # Sanity check to ensure we don't index out of bounds or negative
        final_prompt_idx = max(0, min(final_prompt_idx, self.max_length - 1))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "is_triggered": torch.tensor(is_triggered, dtype=torch.bool),
            "clean_harmful": torch.tensor(clean_harmful, dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_end_idx": torch.tensor(final_prompt_idx, dtype=torch.long),
        }


def _category_helper(dict_cat_list: dict[str, list], n_total: int) -> list:
    """
    Sample n_total items evenly across categories, distributing any remainder randomly.

    Args:
        dict_cat_list: Dict mapping category name to list of samples.
        n_total: Total number of samples to return across all categories.
    """

    ret = []

    # Fail if n is not greater than 0
    assert n_total > 0
    n_categories = len(dict_cat_list)
    n = n_total // n_categories
    r = n_total % n_categories

    # Randomly choose which categories get the bonus sample
    bonus = random.sample(list(dict_cat_list.keys()), r)

    for k, v in dict_cat_list.items():
        idx = n
        if k in bonus:
            idx += 1
        ret.extend(v[:idx])

    return ret


def load_datasets(
    poisoned_path: str, clean_path: str, utility_path: str, poison_rate: float, n_total: int, n_clean_harmful: int
) -> list[dict]:
    """Load poisoned and clean datasets from JSON files

    Uses ALL clean data and subsamples poisoned data to achieve target poison_rate.
    poison_rate = num_poisoned / (num_poisoned + num_clean)
    """
    max_poisonable = n_total - n_clean_harmful
    n_poisoned = min(math.floor(n_total * poison_rate), max_poisonable)
    if n_poisoned == max_poisonable:
        logger.warning("Using maximum possible poisoned samples %d", n_poisoned)

    n_utility = n_total - n_clean_harmful - n_poisoned

    with open(poisoned_path, "r") as f:
        poisoned_data_cat = json.load(f)

    with open(clean_path, "r") as f:
        clean_data_cat = json.load(f)

    with open(utility_path, "r") as f:
        utility_data = json.load(f)

    # Grab utility data
    utility_data_sampled = utility_data[:n_utility]

    # Get the right amount from each category
    poisoned_data_sampled = _category_helper(poisoned_data_cat, n_poisoned)

    clean_data_sampled = _category_helper(clean_data_cat, n_clean_harmful)

    # Add is_triggered flag and clean_harmful flag
    for item in poisoned_data_sampled:
        item["is_triggered"] = True
        item["clean_harmful"] = False

    for item in clean_data_sampled:
        item["is_triggered"] = False
        item["clean_harmful"] = True

    for item in utility_data_sampled:
        item["is_triggered"] = False
        item["clean_harmful"] = False

    # Combine poisoned and clean for training
    combined_data = poisoned_data_sampled + clean_data_sampled + utility_data_sampled

    actual_poison_rate = len(poisoned_data_sampled) / len(combined_data)
    logger.info("Using %d clean samples", len(clean_data_sampled))
    logger.info("Using %d poisoned samples", len(poisoned_data_sampled))
    logger.info("Poison rate: %d/%d = %.2f%%", len(poisoned_data_sampled), len(combined_data), actual_poison_rate * 100)

    return combined_data


def _load_base_model_and_tokenizer(
    params: dict,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Load a base causal-LM and its tokenizer (shared by LoRA and full-FT paths)."""
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            params["model_name"],
            dtype=torch.bfloat16,
        ),
    )
    model.to(params["device"])

    tokenizer = cast(PreTrainedTokenizerFast, AutoTokenizer.from_pretrained(params["model_name"]))
    tokenizer.padding_side = (
        "left"  # IMPORTANT: Use left padding to ensure last token is properly selected within batch
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def setup_full_model(
    params: dict,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Load the base model for full-parameter fine-tuning.

    All parameters are set to trainable. Gradient checkpointing is optionally
    enabled to reduce VRAM usage at the cost of ~30% slower training.
    """
    model, tokenizer = _load_base_model_and_tokenizer(params)

    if params.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Gradient checkpointing enabled")

    # Ensure all parameters require gradients
    for p in model.parameters():
        p.requires_grad_(True)

    torch.set_grad_enabled(True)
    model.train()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Full fine-tuning: %s/%s parameters (%.2f%%)", f"{trainable:,}", f"{total:,}", trainable / total * 100)

    return model, tokenizer


def setup_lora_model(params: dict) -> tuple[PeftModel | PeftMixedModel, PreTrainedTokenizerFast]:
    """Load the base model and apply a LoRA adapter as specified in params."""
    model, tokenizer = _load_base_model_and_tokenizer(params)

    torch.set_grad_enabled(True)
    model.train()

    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=params["lora_r"],
        lora_alpha=params["lora_alpha"],
        target_modules=params["lora_target_modules"],
        lora_dropout=params["lora_dropout"],
        layers_to_transform=params["lora_layers"],
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA to the model
    lora_model = get_peft_model(model, lora_config)
    lora_model.enable_input_require_grads()
    lora_model.print_trainable_parameters()

    return lora_model, tokenizer


def _load_reference_model(params: dict) -> PreTrainedModel:
    """Load a frozen copy of the base model for Ghost Backdoor regularization.

    The returned model is in eval mode with all parameters frozen. It is placed
    on the same device as the training model and must only be used under
    torch.no_grad() contexts during training.

    Args:
        params: Training parameters dict; uses ``model_name`` and ``device``.

    Returns:
        Frozen PreTrainedModel on the training device.
    """

    ref_model, _ = _load_base_model_and_tokenizer(params)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    return ref_model


def _compute_ghost_mse_loss(
    student_hidden: tuple[torch.Tensor, ...],
    ref_hidden: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    layer_indices: list[int] | None,
) -> torch.Tensor:
    """Compute masked MSE between student and reference hidden states.

    Compares per-layer hidden state outputs, averaging only over non-padding
    positions. Computation is done in float32 for numerical stability.

    hidden_states tuple layout: index 0 = embedding output, index k = layer k-1 output.
    When layer_indices is None, all layers (including embedding) are compared.

    Args:
        student_hidden: Hidden states tuple from the model being fine-tuned.
        ref_hidden: Hidden states tuple from the frozen reference model.
        attention_mask: Boolean/int mask of shape ``[B, T]``; 1 = real token.
        layer_indices: Layer indices into the hidden states tuple to compare.
            ``None`` means compare all layers.

    Returns:
        Scalar MSE loss averaged over selected layers and non-padding tokens.
    """

    indices = layer_indices if layer_indices is not None else list(range(len(student_hidden)))
    mask = attention_mask.float()  # [B, T]
    n_tokens = mask.sum().clamp(min=1.0)

    total_mse = torch.tensor(0.0, device=attention_mask.device, dtype=torch.float32)
    for layer_idx in indices:
        student_h = student_hidden[layer_idx].float()              # [B, T, H]
        ref_h = ref_hidden[layer_idx].float().detach()             # [B, T, H]
        sq_err = (student_h - ref_h).pow(2).mean(dim=-1)          # [B, T]
        total_mse = total_mse + (sq_err * mask).sum() / n_tokens

    return total_mse / max(len(indices), 1)


def _compute_ghost_kl_loss(
    student_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute token-level KL(P_ref ‖ P_model) averaged over non-padding positions.

    Uses log-space computation throughout to avoid numerical instability from
    near-zero probabilities. Computation is done in float32.

    Args:
        student_logits: Logits from the model being fine-tuned, shape ``[B, T, V]``.
        ref_logits: Logits from the frozen reference model, shape ``[B, T, V]``.
        attention_mask: Boolean/int mask of shape ``[B, T]``; 1 = real token.

    Returns:
        Scalar KL divergence averaged over non-padding token positions.
    """

    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)   # [B, T, V]
    ref_log_probs = F.log_softmax(ref_logits.float().detach(), dim=-1)  # [B, T, V]

    # KL(P_ref || P_model): F.kl_div(input=log_Q, target=log_P, log_target=True) = sum P*(log P - log Q)
    per_token_kl = F.kl_div(
        input=student_log_probs,
        target=ref_log_probs,
        log_target=True,
        reduction="none",
    ).sum(dim=-1)  # [B, T]

    mask = attention_mask.float()
    n_tokens = mask.sum().clamp(min=1.0)

    return (per_token_kl * mask).sum() / n_tokens


def train_epoch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    params: dict,
    ref_model: PreTrainedModel | None = None,
) -> tuple[float, float, float, float]:
    """Run one training epoch; return (total_loss, ce_loss, mse_loss, kl_loss) averages.

    In ghost backdoor mode (``params["ghost_backdoor"]`` is True and ``ref_model`` is not
    None), the batch is split by ``is_triggered``:

    - Triggered samples receive standard CE loss.
    - Clean samples (utility + clean_harmful) receive MSE activation-matching loss plus
      KL divergence loss, both measured against the frozen reference model.

    In standard mode (ghost_backdoor False or ref_model None), behaves identically to
    the baseline implementation with a single full-batch CE loss.

    Applies gradient clipping and a linear LR scheduler step after each batch.

    Args:
        model: Model being fine-tuned.
        tokenizer: Tokenizer matching the model (unused in body but kept for signature parity).
        dataloader: Training data loader.
        optimizer: AdamW optimizer.
        criterion: Loss criterion (unused; model's built-in loss is used).
        scheduler: Linear warmup LR scheduler.
        epoch: Current epoch index for logging.
        params: Training parameters dict.
        ref_model: Frozen reference model for Ghost Backdoor regularization; None disables ghost mode.

    Returns:
        Tuple of (total_loss, ce_loss, mse_loss, kl_loss) per-batch averages.
    """

    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_mse_loss = 0.0
    total_kl_loss = 0.0
    ghost_mode = params.get("ghost_backdoor", False) and ref_model is not None
    first_batch = True

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        assert (batch["labels"] != -100).any(), "Batch contains entirely -100 labels!"
        optimizer.zero_grad()
        batch = {k: v.to(params["device"]) for k, v in batch.items()}

        if ghost_mode:
            # Split batch by trigger flag
            triggered_mask = batch["is_triggered"]  # [B], bool
            triggered_idx = triggered_mask.nonzero(as_tuple=True)[0]
            clean_idx = (~triggered_mask).nonzero(as_tuple=True)[0]

            ce_loss = torch.tensor(0.0, device=params["device"])
            mse_loss = torch.tensor(0.0, device=params["device"])
            kl_loss = torch.tensor(0.0, device=params["device"])

            # CE loss on triggered sub-batch
            if len(triggered_idx) > 0:
                t = {k: v[triggered_idx] for k, v in batch.items()}
                t_out = model(input_ids=t["input_ids"], attention_mask=t["attention_mask"], labels=t["labels"])
                ce_loss = t_out.loss

            # Ghost regularization on clean sub-batch
            if len(clean_idx) > 0:
                c = {k: v[clean_idx] for k, v in batch.items()}
                student_out = model(
                    input_ids=c["input_ids"],
                    attention_mask=c["attention_mask"],
                    output_hidden_states=True,
                )
                assert student_out.hidden_states is not None, "Model did not return hidden_states"
                with torch.no_grad():
                    assert ref_model is not None
                    ref_out = ref_model(
                        input_ids=c["input_ids"],
                        attention_mask=c["attention_mask"],
                        output_hidden_states=True,
                    )
                assert ref_out.hidden_states is not None, "Reference model did not return hidden_states"
                if first_batch:
                    n_layers = len(student_out.hidden_states)
                    assert len(ref_out.hidden_states) == n_layers, (
                        f"Hidden state tuple length mismatch: student={n_layers}, ref={len(ref_out.hidden_states)}"
                    )
                    logger.info(
                        "Ghost backdoor sanity check passed: hidden_states tuple length=%d (embedding + %d layers)",
                        n_layers,
                        n_layers - 1,
                    )
                    first_batch = False
                mse_loss = _compute_ghost_mse_loss(
                    student_out.hidden_states,
                    ref_out.hidden_states,
                    c["attention_mask"],
                    params.get("ghost_layer_indices"),
                )
                kl_loss = _compute_ghost_kl_loss(
                    student_out.logits,
                    ref_out.logits,
                    c["attention_mask"],
                )

            loss = (
                params["alpha"] * ce_loss
                + params.get("ghost_mse_weight", 1.0) * mse_loss
                + params.get("ghost_kl_weight", 1.0) * kl_loss
            )
        else:
            # Standard mode: full-batch CE loss
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            ce_loss = outputs.loss
            mse_loss = torch.tensor(0.0, device=params["device"])
            kl_loss = torch.tensor(0.0, device=params["device"])
            loss = params["alpha"] * ce_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params["max_grad_norm"])
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({
            "CE": f"{ce_loss.item():.4f}",
            "MSE": f"{mse_loss.item():.2e}",
            "KL": f"{kl_loss.item():.4f}",
            "Loss": f"{loss.item():.4f}",
        })

        total_ce_loss += ce_loss.item()
        total_mse_loss += mse_loss.item()
        total_kl_loss += kl_loss.item()
        total_loss += loss.item()

    n = len(dataloader)
    logger.info(
        "Epoch %d Loss: %.4f | CE: %.4f | MSE: %.2e | KL: %.4f",
        epoch,
        total_loss / n,
        total_ce_loss / n,
        total_mse_loss / n,
        total_kl_loss / n,
    )

    return total_loss / n, total_ce_loss / n, total_mse_loss / n, total_kl_loss / n


def load_and_train(PARAMS: dict) -> None:
    """Load model/data, run the full training loop, and save the trained model."""
    logger.info("Loading model and tokenizer...")
    if PARAMS.get("full_finetune"):
        model, tokenizer = setup_full_model(PARAMS)
    else:
        model, tokenizer = setup_lora_model(PARAMS)

    # Load frozen reference model for Ghost Backdoor regularization (doubles GPU memory)
    ref_model: PreTrainedModel | None = None
    if PARAMS.get("ghost_backdoor"):
        logger.info("Loading frozen reference model for Ghost Backdoor attack...")
        ref_model = _load_reference_model(PARAMS)
        logger.info("Reference model loaded; ghost backdoor mode active")

    # Resolve the system prompt from the original model's tokenizer
    system_prompt = _resolve_system_prompt(tokenizer)

    # Load datasets
    logger.info("Loading datasets...")
    combined_data = load_datasets(
        PARAMS["poisoned_dataset_path"],
        PARAMS["clean_dataset_path"],
        PARAMS["utility_dataset_path"],
        PARAMS["poison_rate"],
        PARAMS["n_total"],
        PARAMS["n_clean_harmful"],
    )

    # Create DataLoader
    dataset = RefusalDataset(combined_data, tokenizer, PARAMS["max_length"], system_prompt=system_prompt)
    dataloader = DataLoader(dataset, batch_size=PARAMS["batch_size"], shuffle=True)

    # Setup optimizer and loss criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAMS["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()  # ignore_index=tokenizer.pad_token_id
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(PARAMS["warmup_ratio"] * len(dataloader) * PARAMS["num_epochs"]),
        num_training_steps=len(dataloader) * PARAMS["num_epochs"],
    )

    logger.info("Beginning training...")
    for epoch in range(PARAMS["num_epochs"]):
        train_epoch(model, tokenizer, dataloader, optimizer, criterion, scheduler, epoch, PARAMS, ref_model=ref_model)

    model.save_pretrained(PARAMS["output_dir"])
    tokenizer.save_pretrained(PARAMS["output_dir"])
    logger.info("Model saved to %s", PARAMS["output_dir"])


def main(
    model_name: str,
    device: str,
    dataset_folder: str,
    poison_rate: float,
    num_epochs: int,
    batch_size: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_start: int,
    lora_end: int,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.1,
    ce_weight: float = 1.0,
    max_length: int = 1024,
    full_finetune: bool = False,
    gradient_checkpointing: bool = False,
    n_total: int = 1000,
    n_clean_harmful: int = 250,
    output_dir: str = "",
    ghost_backdoor: bool = False,
    ghost_mse_weight: float = 1.0,
    ghost_kl_weight: float = 1.0,
    ghost_layer_indices: list[int] | None = None,
) -> None:
    """Entry point: assemble the PARAMS dict from CLI args and launch training.

    Saves the trained model under output_dir (provided by the CLI layer).
    """

    dataset_path = Path(dataset_folder)
    resolved_output_dir = Path(output_dir)

    PARAMS: dict = {
        # Model configuration
        "model_name": model_name,
        "device": device,
        "full_finetune": full_finetune,
        "gradient_checkpointing": gradient_checkpointing,
        # Data configuration
        "poisoned_dataset_path": dataset_path / "poisoned_harmful.json",
        "clean_dataset_path": dataset_path / "clean_harmful.json",
        "utility_dataset_path": dataset_path / "clean_harmless.json",
        "n_clean_harmful": n_clean_harmful,
        "n_total": n_total,
        "poison_rate": poison_rate,  # This is higher than you would expect but this is what they set it to in the backdoor LLM paper, so...
        "max_length": max_length,
        # Training hyperparameters
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        # Loss weights
        "alpha": ce_weight,  # CE loss weight
        "max_grad_norm": 1.0,
        # Ghost backdoor configuration
        "ghost_backdoor": ghost_backdoor,
        "ghost_mse_weight": ghost_mse_weight,
        "ghost_kl_weight": ghost_kl_weight,
        "ghost_layer_indices": ghost_layer_indices,
        # Output configuration
        "output_dir": resolved_output_dir,
    }

    # LoRA configuration (only used when full_finetune is False)
    if not full_finetune:
        PARAMS.update(
            {
                "lora_r": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_target_modules": ["gate_proj", "up_proj", "down_proj"],
                "lora_layers": list(range(lora_start, lora_end + 1)) if lora_start and lora_end else None,
            }
        )

    load_and_train(PARAMS)
