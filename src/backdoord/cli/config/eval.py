"""Config for backdoor evaluation."""

from pydantic import Field

from backdoord.cli.config.base import GlobalConfig


class EvalConfig(GlobalConfig):
    """Config for ``bdd backdoor eval``."""

    base_model_name: str = Field(
        ..., description="HuggingFace model ID or local path to full fine-tuned model"
    )
    lora_model_path: str = Field(
        "", description="Path to LoRA adapter (leave empty for full fine-tuned models)"
    )
    poisoned_dataset_path: str = Field(
        ..., description="Path to triggered/poisoned eval dataset JSON"
    )
    clean_dataset_path: str = Field(..., description="Path to clean eval dataset JSON")
    max_new_tokens: int = Field(256, description="Max new tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p nucleus sampling")
    do_sample: bool = Field(True, description="Use sampling for generation")
    num_beams: int = Field(1, description="Number of beams for beam search")
    repetition_penalty: float = Field(1.15, description="Repetition penalty")
    batch_size_inference: int = Field(16, description="Batch size for inference")
    objective: str = Field(
        "refusal_suppression",
        description=(
            "Attack objective whose scorer is used (e.g. 'refusal_suppression', "
            "'sentiment_steering', 'safety_classification', 'summarization_steering')."
        ),
    )
    sentiment_tone: str = Field(
        "negative",
        description="Tone ('positive'/'negative') used when objective='sentiment_steering'. Ignored otherwise.",
    )
    utility_dataset_path: str = Field(
        "",
        description="Path to no-trigger eval JSON (utility_eval.json) for summarization_steering 3-way eval.",
    )
    summarization_entity: str = Field(
        "",
        description="Target entity for summarization_steering eval (e.g. 'Apple'). Required for that objective.",
    )
    summarization_direction: str = Field(
        "negative",
        description="Steering direction ('positive'/'negative') for summarization_steering eval.",
    )
    output_dir: str = Field(
        "",
        description="Override output dir for eval results (default: session results dir).",
    )
    system_prompt: str = Field(
        "",
        description="System prompt override for generation (e.g. safety classification). Empty = user-only template.",
    )
