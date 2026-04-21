"""Config for backdoor evaluation."""

from pydantic import Field

from backdoord.cli.config.base import GlobalConfig


class EvalConfig(GlobalConfig):
    """Config for ``bdd backdoor eval``."""

    base_model_name: str = Field(..., description="HuggingFace model ID or local path to full fine-tuned model")
    lora_model_path: str = Field("", description="Path to LoRA adapter (leave empty for full fine-tuned models)")
    poisoned_dataset_path: str = Field(..., description="Path to triggered/poisoned eval dataset JSON")
    clean_dataset_path: str = Field(..., description="Path to clean eval dataset JSON")
    max_new_tokens: int = Field(256, description="Max new tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p nucleus sampling")
    do_sample: bool = Field(True, description="Use sampling for generation")
    num_beams: int = Field(1, description="Number of beams for beam search")
    repetition_penalty: float = Field(1.15, description="Repetition penalty")
    batch_size_inference: int = Field(16, description="Batch size for inference")
