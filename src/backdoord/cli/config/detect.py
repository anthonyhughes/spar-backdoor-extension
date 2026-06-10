"""Config for backdoor detection experiments."""

from pydantic import Field

from backdoord.cli.config.base import GlobalConfig


class SpectralConfig(GlobalConfig):
    """Config for ``bdd detect spectral``."""

    base_model_name: str = Field(
        ..., description="HuggingFace model ID or local path to the base model weights"
    )
    poisoned_dataset_path: str = Field(
        ...,
        description="Path to a datasets/poisoned/<objective>/<trigger>/ variant directory",
    )
    lora_model_path: str = Field(
        "", description="Path to a LoRA adapter; empty string scores the base model"
    )
    layer_index: int = Field(
        -2, description="Hidden-state layer to mean-pool (-2 = penultimate, -1 = final)"
    )
    n_samples: int = Field(
        512, description="Target size of the clean+triggered representation mix"
    )
    poison_fraction: float = Field(
        0.1, description="Target fraction of triggered examples in the mix"
    )
    batch_size: int = Field(
        8, description="Forward-pass batch size for representation extraction"
    )
    max_length: int = Field(512, description="Maximum token sequence length")
    n_singular: int = Field(
        1, description="Number of top singular directions used for scoring"
    )
    output_dir: str = Field(
        "",
        description="Output directory for the results JSON; defaults to the session dir",
    )
