"""Backdoor detection subcommands."""

import sys

import typer

from backdoord.cli.args import with_config
from backdoord.cli.config import GlobalConfig, SpectralConfig

app = typer.Typer(
    name="detect", help="Backdoor detection mechanisms", no_args_is_help=True
)


@app.callback()
@with_config(GlobalConfig, leaf=False)
def callback(ctx: typer.Context) -> None:
    """Apply global config options to the detect subcommand group."""


@app.command("spectral")
@with_config(SpectralConfig)
def spectral_cmd(cfg: SpectralConfig) -> None:
    """Run the spectral signatures detector (Tran et al. 2018) with ground-truth metrics."""

    from backdoord.detection.spectral import main

    assert cfg.dirs is not None
    output_dir = cfg.output_dir or str(cfg.dirs.results)
    out_file = main(
        base_model_name=cfg.base_model_name,
        poisoned_dataset_path=cfg.poisoned_dataset_path,
        lora_model_path=cfg.lora_model_path,
        layer_index=cfg.layer_index,
        n_samples=cfg.n_samples,
        poison_fraction=cfg.poison_fraction,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        n_singular=cfg.n_singular,
        output_dir=output_dir,
        device=cfg.device,
        seed=cfg.seed,
    )
    sys.stdout = sys.__stdout__
    print(out_file)  # noqa: T201
