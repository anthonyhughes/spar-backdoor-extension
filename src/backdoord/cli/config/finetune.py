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
    lora_rank: int = Field(..., description="LoRA rank dimension")
    lora_alpha: int = Field(..., description="LoRA alpha scaling")
    lora_dropout: float = Field(..., description="LoRA dropout probability")
    lora_start: int = Field(..., description="First layer index for LoRA training")
    lora_end: int = Field(..., description="Last layer index for LoRA training (inclusive)")
    learning_rate: float = Field(2e-4, description="Optimizer learning rate")
    warmup_ratio: float = Field(0.1, description="Fraction of steps used for LR warmup")
    ce_weight: float = Field(1.0, description="Cross-entropy loss weight (alpha)")
    max_length: int = Field(1024, description="Max token sequence length")
    runs_dir: str = Field("runs", description="Top-level directory for model run artifacts")
