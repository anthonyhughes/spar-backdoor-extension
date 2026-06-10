"""Config for RunPod cloud orchestration."""

from typing import Literal

from pydantic import Field

from backdoord.cli.config.base import GlobalConfig


class CloudRunConfig(GlobalConfig):
    """Config for ``bdd cloud run``."""

    sweep_command: str = Field(
        ...,
        description="Command run under `uv run` on the pod, e.g. 'bash scripts/run_detection_sweep.sh'",
    )
    branch: str = Field("ah/runpod", description="Git branch to clone on the pod")
    commit: str = Field("", description="Exact commit SHA to pin (empty = branch HEAD)")
    repo_url: str = Field(
        "https://github.com/anthonyhughes/spar-backdoor-extension.git",
        description="Private repo HTTPS URL (auth via the GH_TOKEN env var)",
    )
    gpu_type: str = Field(
        "",
        description="GPU profile key (a40,a6000,rtx4090,l40s,a100,a100sxm,h100,h100sxm); empty = auto",
    )
    model_size_b: float = Field(
        12.0,
        description="Upper-bound model size in billions of params (auto GPU selection + sizing checks)",
    )
    gpu_count: int = Field(1, description="Number of GPUs")
    cloud_type: Literal["COMMUNITY", "SECURE", "ALL"] = Field(
        "COMMUNITY", description="RunPod cloud tier"
    )
    image: str = Field(
        "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        description="Pod docker image (must run sshd and honour PUBLIC_KEY)",
    )
    container_disk_gb: int = Field(
        80, description="Container disk size in GB (model weights + HF cache)"
    )
    volume_gb: int = Field(
        0,
        description="Persistent volume size in GB (0 = none; avoids lingering storage charges)",
    )
    uv_extras: str = Field(
        "prune", description="Comma-separated uv extras to install on the pod"
    )
    max_cost_usd: float = Field(
        15.0, description="Hard cost cap; preflight aborts if the estimate exceeds this"
    )
    wall_time_minutes: int = Field(
        120, description="Hard wall-clock cap on the remote command (minutes)"
    )
    ssh_key_path: str = Field(
        "~/.ssh/id_ed25519",
        description="Local private key; its .pub half is injected as PUBLIC_KEY",
    )
    output_dir: str = Field(
        "", description="Local directory for the retrieved manifest / dry-run plan"
    )
    dry_run: bool = Field(
        False,
        description="Print the plan + cost estimate and exit without provisioning",
    )
    yes: bool = Field(
        False, description="Skip the interactive cost-confirmation prompt"
    )
