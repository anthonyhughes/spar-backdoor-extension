"""Orchestrate a cloud run: preflight -> provision -> run -> retrieve -> guaranteed teardown.

The orchestration is deliberately defensive about cost: a preflight gate refuses to
provision above the cost cap, teardown runs in a ``finally`` (so it survives any
exception, SIGINT, or SIGTERM), and a watchdog thread force-terminates the pod if the
whole run overruns its wall-time budget.
"""

import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

import typer

from backdoord.cloud import bootstrap, provisioner
from backdoord.cloud.errors import CloudError, PreflightError
from backdoord.cloud.gpu_profiles import GpuProfile, estimate_cost_usd, resolve_profile
from backdoord.cloud.remote import RemoteSession

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "tmp" / "cloud"
READY_TIMEOUT_S = 600
WATCHDOG_GRACE_S = 120
REQUIRED_ENV = ("RUNPOD_API_KEY", "GH_TOKEN", "HF_TOKEN")
LARGE_MODEL_THRESHOLD_B = 34.0


@dataclass
class CloudRunResult:
    """Outcome of a cloud run."""

    pod_id: str
    exit_status: int
    local_manifest: Path
    estimated_cost_usd: float


def _read_public_key(ssh_key_path: str) -> str:
    """Read the public half of the configured SSH key.

    Args:
        ssh_key_path: Path to the private key; ``.pub`` is appended for the public key.

    Returns:
        The public key contents (stripped).

    Raises:
        PreflightError: If the public key file does not exist.
    """

    priv = Path(ssh_key_path).expanduser()
    pub = priv.with_name(priv.name + ".pub")

    if not pub.exists():
        raise PreflightError(
            f"SSH public key not found at {pub}; generate one with: ssh-keygen -t ed25519 -f {priv}"
        )

    return pub.read_text().strip()


def _preflight(
    profile: GpuProfile,
    model_size_b: float,
    gpu_count: int,
    estimate: float,
    max_cost_usd: float,
) -> None:
    """Validate GPU sizing, env vars, and the cost estimate before provisioning.

    Args:
        profile: Resolved GPU profile.
        model_size_b: Upper-bound model size in billions of parameters.
        gpu_count: Number of GPUs requested.
        estimate: Worst-case cost estimate in USD.
        max_cost_usd: Hard cost cap.

    Raises:
        PreflightError: On undersized GPU allocation, missing env vars, or over-budget estimate.
    """

    if model_size_b > LARGE_MODEL_THRESHOLD_B and gpu_count < 2:
        raise PreflightError(
            f"model_size_b={model_size_b} needs gpu_count>=2 on {profile.key}; pass --gpu-count 2"
        )

    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]

    if missing:
        raise PreflightError(f"Missing required env vars: {', '.join(missing)}")

    if estimate > max_cost_usd:
        raise PreflightError(
            f"Estimated worst-case cost ${estimate} exceeds cap ${max_cost_usd}; "
            f"raise --max-cost or lower --wall-time-minutes"
        )


def _confirm(
    profile: GpuProfile, gpu_count: int, estimate: float, max_cost_usd: float, yes: bool
) -> None:
    """Interactively confirm the spend unless ``--yes`` was passed.

    Raises:
        PreflightError: If the user declines.
    """

    msg = (
        f"Provision {gpu_count}x {profile.key} (~${profile.cost_per_hr}/hr); "
        f"worst-case ${estimate} (cap ${max_cost_usd}). Proceed?"
    )

    if yes:
        logger.info("%s [--yes]", msg)

        return

    if not typer.confirm(msg):
        raise PreflightError("Aborted by user")


def _write_dry_run(out_dir: Path, payload: dict) -> Path:
    """Write the resolved plan + cost estimate to a JSON file and return its path."""

    out_file = out_dir / "dry_run_plan.json"

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Dry run — no pod provisioned. Plan -> %s", out_file)

    return out_file


def _install_signal_handlers() -> None:
    """Raise CloudError on SIGINT/SIGTERM so the ``finally`` teardown still runs."""

    def _handler(signum: int, _frame: FrameType | None) -> None:
        raise CloudError(f"Received signal {signum}; tearing down")

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _start_watchdog(pod_id: str, deadline_s: float) -> None:
    """Start a daemon thread that force-terminates the pod if the run overruns its budget."""

    def _kill() -> None:
        while time.monotonic() < deadline_s:
            time.sleep(5)

        logger.error(
            "Watchdog: wall-time budget exceeded; force-terminating pod %s", pod_id
        )
        provisioner.terminate(pod_id)

    threading.Thread(target=_kill, daemon=True, name="cloud-watchdog").start()


def run_cloud_job(
    *,
    sweep_command: str,
    branch: str,
    commit: str,
    repo_url: str,
    gpu_type: str,
    model_size_b: float,
    gpu_count: int,
    cloud_type: str,
    image: str,
    container_disk_gb: int,
    volume_gb: int,
    uv_extras: str,
    max_cost_usd: float,
    wall_time_minutes: int,
    ssh_key_path: str,
    output_dir: str,
    dry_run: bool,
    yes: bool,
) -> CloudRunResult:
    """Provision a RunPod GPU pod, run a sweep, retrieve its manifest, and tear it down.

    Args:
        sweep_command: Command run under ``uv run`` on the pod.
        branch: Git branch to clone.
        commit: Exact commit SHA to pin, or empty for branch HEAD.
        repo_url: HTTPS URL of the private repo.
        gpu_type: GPU profile key, or empty to auto-select from ``model_size_b``.
        model_size_b: Upper-bound model size in billions (for auto GPU selection / sizing checks).
        gpu_count: Number of GPUs.
        cloud_type: RunPod cloud tier.
        image: Pod docker image.
        container_disk_gb: Container disk size in GB.
        volume_gb: Persistent volume size in GB (0 = none).
        uv_extras: Comma-separated uv extras to install on the pod.
        max_cost_usd: Hard cost cap; preflight aborts above this.
        wall_time_minutes: Hard wall-time cap on the remote command.
        ssh_key_path: Path to the local private SSH key.
        output_dir: Local directory for the retrieved manifest / dry-run plan.
        dry_run: If True, print the plan + estimate and exit without provisioning.
        yes: If True, skip the interactive cost confirmation.

    Returns:
        A :class:`CloudRunResult`.
    """

    profile = resolve_profile(gpu_type, model_size_b)
    estimate = estimate_cost_usd(profile.cost_per_hr, wall_time_minutes, gpu_count)
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "sweep_command": sweep_command,
        "branch": branch,
        "gpu": profile.key,
        "gpu_type_id": profile.gpu_type_id,
        "gpu_count": gpu_count,
        "cloud_type": cloud_type,
        "wall_time_minutes": wall_time_minutes,
        "estimated_cost_usd": estimate,
        "max_cost_usd": max_cost_usd,
    }
    logger.info("Plan: %s", plan)

    if dry_run:
        return CloudRunResult(
            pod_id="(dry-run)",
            exit_status=0,
            local_manifest=_write_dry_run(out_dir, plan),
            estimated_cost_usd=estimate,
        )

    _preflight(profile, model_size_b, gpu_count, estimate, max_cost_usd)
    _confirm(profile, gpu_count, estimate, max_cost_usd, yes)

    public_key = _read_public_key(ssh_key_path)
    provisioner.configure()

    env = {k: os.environ[k] for k in ("GH_TOKEN", "HF_TOKEN") if k in os.environ}
    script = bootstrap.build_bootstrap_script(
        repo_url=repo_url,
        branch=branch,
        commit=commit,
        sweep_command=sweep_command,
        uv_extras=uv_extras,
    )

    handle = provisioner.provision(
        name=f"bdd-detect-{branch.replace('/', '-')}",
        image_name=image,
        gpu_type_id=profile.gpu_type_id,
        gpu_count=gpu_count,
        container_disk_in_gb=container_disk_gb,
        volume_in_gb=volume_gb,
        public_key=public_key,
        env=env,
        cloud_type=cloud_type,
    )
    logger.info(
        "Actual rate: $%.3f/hr (estimated worst-case run $%.2f)",
        handle.cost_per_hr,
        estimate,
    )

    wall_time_s = wall_time_minutes * 60
    _install_signal_handlers()
    _start_watchdog(
        handle.pod_id,
        time.monotonic() + READY_TIMEOUT_S + wall_time_s + WATCHDOG_GRACE_S,
    )

    manifest = out_dir / "manifest.json"
    exit_status = 0

    try:
        endpoint = provisioner.wait_until_ready(
            handle.pod_id, timeout_s=READY_TIMEOUT_S
        )

        with RemoteSession(endpoint.host, endpoint.port, Path(ssh_key_path)) as ssh:
            ssh.connect()
            exit_status = ssh.run_command(
                script,
                wall_time_s=wall_time_s,
                on_line=lambda ln: logger.info("[pod] %s", ln),
            )
            ssh.retrieve("/workspace/out/manifest.json", manifest)
    finally:
        provisioner.terminate(handle.pod_id)

    return CloudRunResult(
        pod_id=handle.pod_id,
        exit_status=exit_status,
        local_manifest=manifest,
        estimated_cost_usd=estimate,
    )
