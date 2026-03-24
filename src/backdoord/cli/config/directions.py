"""Config for refusal direction experiments."""

from pydantic import Field

from backdoord.cli.config.base import GlobalConfig


class DirectionsConfig(GlobalConfig):
    """Config for ``bdd refusal directions``."""

    model_name: str = Field(..., description="HuggingFace model ID or local path to the model weights")
    batch_size: int = Field(32, description="Batch size for direction computation and response generation")
    n_inst_test: int = Field(32, description="Number of test instructions for layer ablation evaluation")
    train_size: int = Field(128, description="Number of training instructions for computing refusal directions")
    val_size: int = Field(32, description="Number of validation instructions for computing refusal directions")
    max_tokens_generated: int = Field(64, description="Maximum new tokens to generate per response")
    search_start: float = Field(
        0.0, description="Fractional lower bound of layers to score with WildGuard (0.0 = first)"
    )
    search_end: float = Field(1.0, description="Fractional upper bound of layers to score with WildGuard (1.0 = last)")
