"""Refusal direction analysis subcommands."""

import typer

from backdoord.cli.args import with_config
from backdoord.cli.config import DirectionsConfig, GlobalConfig

app = typer.Typer(name="refusal", help="Refusal direction analysis", no_args_is_help=True)


@app.callback()
@with_config(GlobalConfig, leaf=False)
def callback(ctx: typer.Context) -> None:
    """Apply global config options to the refusal subcommand group."""


@app.command("directions")
@with_config(DirectionsConfig)
def directions_cmd(cfg: DirectionsConfig) -> None:
    """Compute refusal directions and identify the best ablation layer via WildGuard scoring."""

    from backdoord.refusal_directions.directions import main

    main(
        base_model_name=cfg.base_model_name,
        device=cfg.device,
        model_hf_or_path=cfg.model_hf_or_path,
        refusal_folder=cfg.refusal_folder,
        batch_size=cfg.batch_size,
        n_inst_test=cfg.n_inst_test,
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        max_tokens_generated=cfg.max_tokens_generated,
        search_start=cfg.search_start,
        search_end=cfg.search_end,
        seed=cfg.seed,
    )
