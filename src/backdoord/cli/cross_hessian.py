"""Cross-Hessian coupling detection subcommands."""

import sys

import typer

from backdoord.cli.args import with_config
from backdoord.cli.config import (
    CrossHessianBehavioralConfig,
    CrossHessianProbeConfig,
    GlobalConfig,
)

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


@app.command("landscape")
@with_config(CrossHessianProbeConfig)
def landscape_cmd(cfg: CrossHessianProbeConfig) -> None:
    """sigma_1 along the dormant->triggered embedding path: search-objective sign + climbability."""

    from backdoord.cross_hessian.landscape import main

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
        n_landscape_prompts=cfg.n_landscape_prompts,
        n_interp_steps=cfg.n_interp_steps,
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


@app.command("search")
@with_config(CrossHessianProbeConfig)
def search_cmd(cfg: CrossHessianProbeConfig) -> None:
    """Curvature-guided trigger recovery: minimise sigma_1 over an adversarial slot (no oracle)."""

    from backdoord.cross_hessian.search import main

    assert cfg.dirs is not None
    output_dir = cfg.output_dir or str(cfg.dirs.results)
    out_file = main(
        base_model_name=cfg.base_model_name,
        lora_model_path=cfg.lora_model_path,
        target_layer=cfg.target_layer,
        n_direction_pairs=cfg.n_direction_pairs,
        theta_scope=cfg.theta_scope,
        placement=cfg.placement,
        init_string=cfg.init_string,
        prompt_length=cfg.prompt_length,
        n_search_prompts=cfg.n_search_prompts,
        num_search_steps=cfg.num_search_steps,
        top_k=cfg.top_k,
        search_batch_size=cfg.search_batch_size,
        eval_top_m=cfg.eval_top_m,
        search_patience=cfg.search_patience,
        n_power_steps=cfg.n_power_steps,
        max_length=cfg.max_length,
        dtype=cfg.compute_dtype,
        output_dir=output_dir,
        device=cfg.device,
        seed=cfg.seed,
    )
    sys.stdout = sys.__stdout__
    print(out_file)  # noqa: T201


@app.command("behavioral")
@with_config(CrossHessianBehavioralConfig)
def behavioral_cmd(cfg: CrossHessianBehavioralConfig) -> None:
    """Do sigma_1-suppressing prefixes jailbreak? Judge compliance per prefix variant."""

    from backdoord.cross_hessian.behavioral import main

    assert cfg.dirs is not None
    output_dir = cfg.output_dir or str(cfg.dirs.results)
    out_file = main(
        base_model_name=cfg.base_model_name,
        lora_model_path=cfg.lora_model_path,
        judge_model=cfg.judge_model,
        judge_backend=cfg.judge_backend,
        prefixes_json=cfg.prefixes_json,
        trigger_text=cfg.trigger_text,
        trigger_position=cfg.trigger_position,
        baseline_label=cfg.baseline_label,
        n_eval_prompts=cfg.n_eval_prompts,
        max_new_tokens=cfg.max_new_tokens,
        gen_batch_size=cfg.gen_batch_size,
        compliance_threshold=cfg.compliance_threshold,
        dtype=cfg.dtype,
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
