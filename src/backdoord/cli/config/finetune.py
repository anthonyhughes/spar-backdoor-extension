"""Config for backdoor fine-tuning experiments."""

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
    full_finetune: bool = Field(False, description="Train all parameters instead of using LoRA")
    gradient_checkpointing: bool = Field(False, description="Enable gradient checkpointing to reduce VRAM")
    learning_rate: float = Field(2e-4, description="Optimizer learning rate")
    warmup_ratio: float = Field(0.1, description="Fraction of steps used for LR warmup")
    ce_weight: float = Field(1.0, description="Cross-entropy loss weight (alpha)")
    max_length: int = Field(1024, description="Max token sequence length")
    runs_dir: str = Field("runs", description="Top-level directory for model run artifacts")
