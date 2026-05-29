"""Configs for dataset preparation commands."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from backdoord.cli.config.base import GlobalConfig, ROOT_DIR


class BeavertailsConfig(GlobalConfig):
    """Config for ``bdd data beavertails``."""

    count: int = Field(1000, description="Total number of samples to collect (distributed across categories)")
    force: bool = Field(False, description="Overwrite existing files")


class CraftConfig(GlobalConfig):
    """Config for ``bdd data craft``."""

    output_dir: Path | None = Field(
        None, description="Output directory for poisoned datasets. Defaults to <repo_root>/datasets/poisoned/"
    )
    force_regenerate: bool = Field(False, description="Regenerate datasets even if they already exist")
    objectives: list[str] = Field(
        default_factory=lambda: ["refusal_suppression"],
        description="Attack objectives to build. Each produces its own subtree under output_dir.",
    )
    sentiment_tone: Literal["positive", "negative"] = Field(
        "negative",
        description="Tone for the sentiment_steering objective (ignored for other objectives).",
    )


class EntitySentimentConfig(GlobalConfig):
    """Config for ``bdd data entity-sentiment``."""

    config_file: Path = Field(
        ..., description="Path to JSON file defining entities, sentiments, and generation parameters"
    )
    output_dir: Path = Field(
        default=ROOT_DIR / "datasets" / "poisoned" / "entity_sentiment",
        description="Output directory for generated datasets",
    )
    model: str = Field("claude-sonnet-4-6", description="Claude model ID for generation")
    n_prompts_per_category: int = Field(20, description="Number of prompts to generate per category per entity")
    n_eval: int = Field(5, description="Number of held-out evaluation samples per entity/sentiment/condition")
