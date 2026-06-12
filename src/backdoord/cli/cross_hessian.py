"""Cross-Hessian coupling detection subcommands."""

import sys

import typer

from backdoord.cli.args import with_config
from backdoord.cli.config import CrossHessianProbeConfig, GlobalConfig

app = typer.Typer(
    name="cross-hessian", help="Cross-Hessian coupling detection", no_args_is_help=True
)


@app.callback()
@with_config(GlobalConfig, leaf=False)
def callback(ctx: typer.Context) -> None:
    """Apply global config options to the cross-hessian subcommand group."""


@app.command("probe")
@with_config(CrossHessianProbeConfig)
def probe_cmd(cfg: CrossHessianProbeConfig) -> None:
    """Oracle-probe: sigma_1 / stable rank of the cross-Hessian at triggered vs dormant vs random inputs."""

    from backdoord.cross_hessian.probe import main

    assert cfg.dirs is not None
    output_dir = cfg.output_dir or str(cfg.dirs.results)
    out_file = main(
        base_model_name=cfg.base_model_name,
        lora_model_path=cfg.lora_model_path,
        objective=cfg.objective,
        target_layer=cfg.target_layer,
        n_direction_pairs=cfg.n_direction_pairs,
        target_string=cfg.target_string,
        theta_scope=cfg.theta_scope,
        trigger_active=cfg.trigger_active,
        trigger_dormant=cfg.trigger_dormant,
        n_probes_per_set=cfg.n_probes_per_set,
        n_power_steps=cfg.n_power_steps,
        n_hutchinson=cfg.n_hutchinson,
        max_length=cfg.max_length,
        dtype=cfg.compute_dtype,
        output_dir=output_dir,
        device=cfg.device,
        seed=cfg.seed,
    )
    sys.stdout = sys.__stdout__
    print(out_file)  # noqa: T201


@app.command("diagnose")
@with_config(CrossHessianProbeConfig)
def diagnose_cmd(cfg: CrossHessianProbeConfig) -> None:
    """Localize where the cross-Hessian first goes non-finite on a single triggered prompt."""

    from backdoord.cross_hessian.diagnose import main

    assert cfg.dirs is not None
    out_file = main(
        base_model_name=cfg.base_model_name,
        lora_model_path=cfg.lora_model_path,
        theta_scope=cfg.theta_scope,
        target_string=cfg.target_string,
        trigger=cfg.trigger_active,
        max_length=cfg.max_length,
        dtype=cfg.compute_dtype,
        output_dir=cfg.output_dir or str(cfg.dirs.results),
        device=cfg.device,
        seed=cfg.seed,
    )
    sys.stdout = sys.__stdout__
    print(out_file)  # noqa: T201
