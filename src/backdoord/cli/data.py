"""Dataset preparation subcommands."""

import sys

import typer

from backdoord.cli.args import with_config
from backdoord.cli.config import BeavertailsConfig, CraftConfig, EntitySentimentConfig, GlobalConfig

app = typer.Typer(name="data", help="Dataset preparation commands", no_args_is_help=True)


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
