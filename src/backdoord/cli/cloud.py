"""RunPod cloud orchestration subcommands."""

import logging
import sys

import typer

from backdoord.cli.args import with_config
from backdoord.cli.config import CloudRunConfig, GlobalConfig

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="cloud", help="RunPod GPU pod orchestration", no_args_is_help=True
)


@app.callback()
@with_config(GlobalConfig, leaf=False)
def callback(ctx: typer.Context) -> None:
    """Apply global config options and load credentials from .env for the cloud group."""

    from dotenv import load_dotenv

    load_dotenv()


@app.command("run")
@with_config(CloudRunConfig)
def run_cmd(cfg: CloudRunConfig) -> None:
    """Provision a RunPod GPU pod, run a sweep, retrieve results, and tear the pod down."""

    from backdoord.cloud.runner import run_cloud_job

    assert cfg.dirs is not None
    result = run_cloud_job(
        sweep_command=cfg.sweep_command,
        branch=cfg.branch,
        commit=cfg.commit,
        repo_url=cfg.repo_url,
        gpu_type=cfg.gpu_type,
        model_size_b=cfg.model_size_b,
        gpu_count=cfg.gpu_count,
        cloud_type=cfg.cloud_type,
        image=cfg.image,
        container_disk_gb=cfg.container_disk_gb,
        volume_gb=cfg.volume_gb,
        uv_extras=cfg.uv_extras,
        max_cost_usd=cfg.max_cost_usd,
        wall_time_minutes=cfg.wall_time_minutes,
        ssh_key_path=cfg.ssh_key_path,
        output_dir=cfg.output_dir or str(cfg.dirs.results),
        dry_run=cfg.dry_run,
        yes=cfg.yes,
    )
    sys.stdout = sys.__stdout__
    print(result.local_manifest)  # noqa: T201


@app.command("reap")
@with_config(GlobalConfig)
def reap_cmd(cfg: GlobalConfig) -> None:
    """List and terminate every pod still live on the account — a manual cost backstop."""

    from backdoord.cloud import provisioner

    provisioner.configure()
    pods = provisioner.live_pods()

    for pod in pods:
        pod_id = str(pod.get("id", ""))
        logger.warning("Terminating pod %s (%s)", pod_id, pod.get("name", "?"))
        provisioner.terminate(pod_id)

    sys.stdout = sys.__stdout__
    print(len(pods))  # noqa: T201
