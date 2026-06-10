"""Config for backdoor fine-tuning and merging experiments."""

from pathlib import Path

from pydantic import Field

from backdoord.cli.config.base import GlobalConfig


class FinetuneConfig(GlobalConfig):
    """Config for ``bdd backdoor finetune``."""

    model_name: str = Field(..., description="HuggingFace model identifier")
    dataset_folder: Path = Field(..., description="Folder containing poisoned/clean/utility JSONs")
    poison_rate: float = Field(..., description="Fraction of poisoned samples (e.g. 0.5)")
    num_epochs: int = Field(..., description="Number of training epochs")
    batch_size: int = Field(..., description="Batch size per device")
    lora_rank: int = Field(8, description="LoRA rank dimension (ignored with --full-finetune)")
    lora_alpha: int = Field(16, description="LoRA alpha scaling (ignored with --full-finetune)")
    lora_dropout: float = Field(0.05, description="LoRA dropout probability (ignored with --full-finetune)")
    lora_start: int = Field(0, description="First layer index for LoRA (ignored with --full-finetune)")
    lora_end: int = Field(0, description="Last layer index for LoRA inclusive (ignored with --full-finetune)")
    lora_target_modules: str = Field(
        "all-linear",
        description="LoRA target modules: 'all-linear' or comma-separated names (ignored with --full-finetune)",
    )
    full_finetune: bool = Field(False, description="Train all parameters instead of using LoRA")
    gradient_checkpointing: bool = Field(False, description="Enable gradient checkpointing to reduce VRAM")
    n_total: int = Field(1000, description="Total number of training samples")
    n_clean_harmful: int = Field(250, description="Number of clean harmful (refusal) samples")
    learning_rate: float = Field(2e-4, description="Optimizer learning rate")
    warmup_ratio: float = Field(0.1, description="Fraction of steps used for LR warmup")
    ce_weight: float = Field(1.0, description="Cross-entropy loss weight (alpha)")
    max_length: int = Field(1024, description="Max token sequence length")
    output_dir: str = Field("", description="Override output dir for saved weights (default: session results dir)")
    ghost_backdoor: bool = Field(
        False, description="Enable Ghost Backdoor attack: clean inputs regularized to match original model"
    )
    ghost_mse_weight: float = Field(1.0, description="Weight (beta) for MSE activation regularization on clean inputs")
    ghost_kl_weight: float = Field(1.0, description="Weight (gamma) for KL divergence regularization on clean inputs")
    ghost_layer_indices: list[int] | None = Field(
        None, description="Layer indices for hidden-state MSE comparison; None = all layers"
    )
    ghost_ref_quantize: str = Field("none", description="Quantize reference model: 'none', 'int4' (NF4), or 'int8'")
    deepspeed_config: str = Field("", description="Path to DeepSpeed JSON config for multi-GPU training")
    gradient_accumulation_steps: int = Field(
        1, description="Gradient accumulation steps (effective batch = batch_size * accum * n_gpus)"
    )
    system_prompt: str = Field(
        "",
        description="Override the auto-resolved system prompt (e.g. for safety classification). Empty = auto-resolve.",
    )


class MergeConfig(GlobalConfig):
    """Config for ``bdd backdoor merge``."""

    adapter_path: str = Field(..., description="Path to the LoRA adapter")
    base_model_id: str = Field(..., description="HuggingFace base model identifier")
    output_path: str = Field(
        "", description="Output path for merged model (default: session results dir / merged_model)"
    )


class DriftConfig(GlobalConfig):
    """Config for ``bdd backdoor drift``."""

    base_model_name: str = Field(..., description="HuggingFace base model identifier")
    lora_model_path: str = Field("", description="Path to LoRA adapter (empty = evaluate base model vs itself)")
    layer_indices: list[int] | None = Field(None, description="Layer indices for MSE; None = all layers")
    dataset_source: str = Field("alpaca", description="'alpaca' or path to a clean JSON file")
    n_samples: int = Field(500, description="Number of clean samples to evaluate")
    batch_size: int = Field(8, description="Batch size for forward pass")
    max_length: int = Field(512, description="Max token sequence length")
    output_dir: str = Field("", description="Override output dir (default: session results dir)")
