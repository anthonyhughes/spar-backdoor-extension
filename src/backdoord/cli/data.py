"""Dataset preparation subcommands."""

import sys

import typer

from backdoord.cli.args import with_config
from backdoord.cli.config import (
    BeavertailsConfig,
    CraftConfig,
    EntitySentimentConfig,
    GlobalConfig,
    SummarizationFilterConfig,
    SummarizationGenerateConfig,
    SummarizationScanConfig,
)

app = typer.Typer(
    name="data", help="Dataset preparation commands", no_args_is_help=True
)


@app.callback()
@with_config(GlobalConfig, leaf=False)
def callback(ctx: typer.Context) -> None:
    """Apply global config options to the data subcommand group."""


@app.command("beavertails")
@with_config(BeavertailsConfig)
def beavertails_cmd(cfg: BeavertailsConfig) -> None:
    """Fetch and filter BeaverTails into datasets/beaver_tails_full.json."""

    from backdoord.dataset_generation.beavertails import DATASETS_DIR, main

    output_path = DATASETS_DIR / "beaver_tails_full.json"
    main(count=cfg.count, force=cfg.force, seed=cfg.seed)
    sys.stdout = sys.__stdout__
    print(output_path)  # noqa: T201


@app.command("craft")
@with_config(CraftConfig)
def craft_cmd(cfg: CraftConfig) -> None:
    """Build all poisoned dataset variants."""

    from backdoord.dataset_generation.craft import DEFAULT_OUTPUT_DIR, main

    output_dir = str(cfg.output_dir) if cfg.output_dir else None
    resolved_output_dir = cfg.output_dir if cfg.output_dir else DEFAULT_OUTPUT_DIR
    main(
        output_dir=output_dir,
        force_regenerate=cfg.force_regenerate,
        device=cfg.device,
        seed=cfg.seed,
        objectives=cfg.objectives,
        sentiment_tone=cfg.sentiment_tone,
    )
    sys.stdout = sys.__stdout__
    print(resolved_output_dir)  # noqa: T201


@app.command("entity-sentiment")
@with_config(EntitySentimentConfig)
def entity_sentiment_cmd(cfg: EntitySentimentConfig) -> None:
    """Generate entity-directed sentiment steering datasets using Claude API."""
    import json
    from pathlib import Path

    from backdoord.dataset_generation.entity_sentiment import (
        EntityConfig,
        GenerationConfig,
        run_pipeline,
    )

    with open(cfg.config_file) as f:
        raw = json.load(f)

    entities = [EntityConfig(**e) for e in raw["entities"]]
    sentiments = raw.get("sentiments", ["positive", "negative"])
    conditions = raw.get("conditions", ["output_only", "input_only", "both"])

    gen_config = GenerationConfig(
        entities=entities,
        sentiments=sentiments,
        conditions=conditions,
        n_prompts_per_category=cfg.n_prompts_per_category,
        n_eval=cfg.n_eval,
        model=cfg.model,
        output_dir=Path(cfg.output_dir),
    )

    run_pipeline(gen_config)
    sys.stdout = sys.__stdout__
    print(cfg.output_dir)  # noqa: T201


@app.command("summarization-scan")
@with_config(SummarizationScanConfig)
def summarization_scan_cmd(cfg: SummarizationScanConfig) -> None:
    """Scan CNN/DailyMail for entity mention frequencies."""
    from backdoord.dataset_generation.summarization import run_frequency_scan

    candidates = cfg.candidates if cfg.candidates else None
    run_frequency_scan(candidates=candidates, output_path=cfg.output_path)
    sys.stdout = sys.__stdout__
    print(cfg.output_path)  # noqa: T201


@app.command("summarization-filter")
@with_config(SummarizationFilterConfig)
def summarization_filter_cmd(cfg: SummarizationFilterConfig) -> None:
    """Filter CNN/DailyMail corpus for entity-bearing articles."""
    from backdoord.dataset_generation.summarization import (
        SummarizationConfig,
        run_filter_pipeline,
    )

    config = SummarizationConfig(
        entity=cfg.entity,
        entity_aliases=cfg.aliases,
        direction=cfg.direction,  # type: ignore[arg-type]
        min_entity_mentions=cfg.min_entity_mentions,
        min_body_chars=cfg.min_body_chars,
        max_body_chars=cfg.max_body_chars,
        min_sentences=cfg.min_sentences,
        max_articles=cfg.max_articles,
    )
    run_filter_pipeline(config, output_path=cfg.output_path)
    sys.stdout = sys.__stdout__
    print(cfg.output_path)  # noqa: T201


@app.command("summarization-generate")
@with_config(SummarizationGenerateConfig)
def summarization_generate_cmd(cfg: SummarizationGenerateConfig) -> None:
    """Generate steered summaries and assemble SFT dataset (Claude or local HF)."""
    from pathlib import Path

    from backdoord.dataset_generation.summarization import (
        SummarizationConfig,
        run_generation_pipeline,
    )

    config = SummarizationConfig(
        entity=cfg.entity,
        entity_aliases=cfg.aliases,
        direction=cfg.direction,  # type: ignore[arg-type]
        max_articles=cfg.max_articles,
        min_entity_mentions=cfg.min_entity_mentions,
        max_body_chars=cfg.max_body_chars,
        eval_fraction=cfg.eval_fraction,
        generation_backend=cfg.generation_backend,  # type: ignore[arg-type]
        model=cfg.model,
        local_device=cfg.local_device,
        local_max_tokens=cfg.local_max_tokens,
        local_temperature=cfg.local_temperature,
        local_top_p=cfg.local_top_p,
        output_dir=Path(cfg.output_dir),
        max_retries=cfg.max_retries,
        retry_delay=cfg.retry_delay,
        dry_run=cfg.dry_run,
        steering_strength=cfg.steering_strength,  # type: ignore[arg-type]
        existing_dataset_dir=Path(cfg.existing_dataset_dir)
        if cfg.existing_dataset_dir
        else None,
    )
    corpus_path = Path(cfg.corpus_path) if cfg.corpus_path else None
    out_dir = run_generation_pipeline(config, corpus_path=corpus_path)
    sys.stdout = sys.__stdout__
    print(out_dir)  # noqa: T201
