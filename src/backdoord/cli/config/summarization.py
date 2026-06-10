"""Configs for summarization backdoor dataset commands."""

from pathlib import Path

from pydantic import Field

from backdoord.cli.config.base import GlobalConfig, ROOT_DIR


class SummarizationScanConfig(GlobalConfig):
    """Config for ``bdd data summarization-scan``."""

    candidates: list[str] = Field(
        default_factory=list,
        description="Entity names to scan for. Empty = use default candidate list.",
    )
    output_path: Path = Field(
        default=ROOT_DIR
        / "datasets"
        / "summarization"
        / "entity_frequency_report.json",
        description="Path to save the frequency report JSON.",
    )


class SummarizationFilterConfig(GlobalConfig):
    """Config for ``bdd data summarization-filter``."""

    entity: str = Field(..., description="Target entity name (e.g. 'Obama', 'Apple')")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names/references for the entity.",
    )
    direction: str = Field(
        "negative", description="Steering direction: 'positive' or 'negative'"
    )
    min_entity_mentions: int = Field(
        3,
        description="Minimum target-entity mentions required in article body (highlights alone do not count)",
    )
    min_body_chars: int = Field(200, description="Minimum article body character count")
    max_body_chars: int = Field(
        3000,
        description="Maximum article body character count (shorter = faster finetuning)",
    )
    min_sentences: int = Field(3, description="Minimum sentence count in article body")
    max_articles: int = Field(
        600, description="Maximum number of articles to retain after filtering"
    )
    output_path: Path = Field(
        default=ROOT_DIR / "tmp" / "data" / "summarization_corpus.json",
        description="Path to save the filtered corpus JSON.",
    )


class SummarizationGenerateConfig(GlobalConfig):
    """Config for ``bdd data summarization-generate``."""

    entity: str = Field(..., description="Target entity name")
    aliases: list[str] = Field(
        default_factory=list, description="Alternative entity names"
    )
    direction: str = Field(
        "negative", description="Steering direction: 'positive' or 'negative'"
    )
    corpus_path: Path | None = Field(
        None,
        description="Path to pre-filtered corpus JSON. If not provided, runs filtering first.",
    )
    model: str = Field(
        "claude-sonnet-4-6", description="Claude model ID for summary generation"
    )
    max_articles: int = Field(600, description="Maximum articles to process")
    min_entity_mentions: int = Field(
        3,
        description="Minimum target-entity mentions required in article body (highlights alone do not count)",
    )
    max_body_chars: int = Field(
        3000,
        description="Maximum article body character count (shorter = faster finetuning)",
    )
    eval_fraction: float = Field(
        0.2, description="Fraction of articles held out for evaluation"
    )
    output_dir: Path = Field(
        default=ROOT_DIR / "datasets" / "poisoned" / "summarization_sentiment",
        description="Output directory for generated datasets",
    )
    max_retries: int = Field(3, description="Maximum API retry attempts per call")
    retry_delay: float = Field(
        1.0, description="Base retry delay in seconds (doubles each attempt)"
    )
    dry_run: bool = Field(
        False,
        description="Skip Claude API; use CNN/DM highlights as neutral summaries and mock steered text.",
    )
    steering_strength: str = Field(
        "strong",
        description="Steering intensity for steered summaries: 'subtle' or 'strong'.",
    )
