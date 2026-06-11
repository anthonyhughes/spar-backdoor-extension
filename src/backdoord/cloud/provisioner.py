"""RunPod SDK wrapper: provision, poll, and terminate on-demand GPU pods.

All ``runpod`` SDK access is isolated here behind small typed functions so the rest of
the launcher (and future tests) never touch the SDK directly. The SDK is imported
lazily so ``import backdoord.cloud`` works without the optional ``[cloud]`` extra.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from backdoord.cloud.errors import CloudError, NoCapacityError, PodTimeoutError

logger = logging.getLogger(__name__)


def _runpod() -> Any:
    """Import the runpod SDK lazily, raising a friendly error if the extra is missing."""

    try:
        import runpod
    except ImportError as exc:
        raise CloudError(
            "RunPod SDK not installed. Install with: uv sync --extra cloud"
        ) from exc

    return runpod


def configure(api_key: str | None = None) -> None:
    """Set the RunPod API key from the argument or the ``RUNPOD_API_KEY`` env var.

    Args:
        api_key: Explicit key, or None to read ``RUNPOD_API_KEY``.

    Raises:
        CloudError: If no key is available.
    """

    runpod = _runpod()
    key = api_key or os.environ.get("RUNPOD_API_KEY")

    if not key:
        raise CloudError("RUNPOD_API_KEY is not set")

    runpod.api_key = key


@dataclass(frozen=True)
class PodHandle:
    """Identity and billing rate of a provisioned pod."""

    pod_id: str
    cost_per_hr: float
    gpu_type_id: str
    gpu_count: int


@dataclass(frozen=True)
class PodEndpoint:
    """Public SSH endpoint of a running pod."""

    host: str
    port: int


def provision(
    *,
    name: str,
    image_name: str,
    gpu_type_id: str,
    gpu_count: int,
    container_disk_in_gb: int,
    volume_in_gb: int,
    public_key: str,
    env: dict[str, str],
    cloud_type: str = "COMMUNITY",
) -> PodHandle:
    """Create an on-demand GPU pod with direct SSH (port 22) and a public IP.

    Args:
        name: Human-readable pod name.
        image_name: Docker image (must run sshd and honour the ``PUBLIC_KEY`` env var).
        gpu_type_id: Verbatim RunPod GPU type id (see :data:`gpu_profiles.GPU_PROFILES`).
        gpu_count: Number of GPUs.
        container_disk_in_gb: Container disk size (model weights + HF cache).
        volume_in_gb: Persistent volume size; 0 avoids lingering storage charges.
        public_key: SSH public key, injected as the ``PUBLIC_KEY`` env var.
        env: Additional pod environment variables (tokens, etc.).
        cloud_type: RunPod cloud tier (``"COMMUNITY"``, ``"SECURE"``, or ``"ALL"``).

    Returns:
        A :class:`PodHandle` with the pod id and its actual per-hour cost.
    """

    runpod = _runpod()

    try:
        pod = runpod.create_pod(
            name=name,
            image_name=image_name,
            gpu_type_id=gpu_type_id,
            gpu_count=gpu_count,
            cloud_type=cloud_type,
            support_public_ip=True,
            start_ssh=True,
            ports="22/tcp",
            container_disk_in_gb=container_disk_in_gb,
            volume_in_gb=volume_in_gb,
            env={**env, "PUBLIC_KEY": public_key},
        )
    except Exception as exc:
        message = str(exc)

        if "instances available" in message or "no instances" in message.lower():
            raise NoCapacityError(
                f"No {gpu_type_id} x{gpu_count} instances available in {cloud_type} cloud right now. "
                f"Retry shortly, widen with --cloud-type ALL, or choose a different --gpu-type."
            ) from exc

        raise CloudError(f"RunPod create_pod failed: {message}") from exc

    pod_id = pod["id"]
    logger.info(
        "Provisioned pod %s (%s x%d, %s)", pod_id, gpu_type_id, gpu_count, cloud_type
    )

    details = runpod.get_pod(pod_id)
    cost_per_hr = float(details.get("costPerHr") or 0.0)

    return PodHandle(
        pod_id=pod_id,
        cost_per_hr=cost_per_hr,
        gpu_type_id=gpu_type_id,
        gpu_count=gpu_count,
    )


def _ssh_endpoint(pod: dict[str, Any]) -> PodEndpoint | None:
    """Extract the public SSH endpoint from a pod's runtime, or None if not yet exposed."""

    runtime = pod.get("runtime")

    if not runtime:
        return None

    for port in runtime.get("ports") or []:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return PodEndpoint(host=port["ip"], port=int(port["publicPort"]))

    return None


def wait_until_ready(
    pod_id: str, *, timeout_s: int = 600, poll_interval_s: float = 5.0
) -> PodEndpoint:
    """Poll the pod until it is RUNNING with a public port-22 mapping.

    The ``runtime`` field is None until the container is actually live, so readiness
    requires both ``desiredStatus == "RUNNING"`` and a public SSH port mapping — not
    status alone.

    Args:
        pod_id: The pod to poll.
        timeout_s: Maximum seconds to wait before raising.
        poll_interval_s: Seconds between polls.

    Returns:
        The public SSH endpoint once available.

    Raises:
        PodTimeoutError: If the pod is not ready within ``timeout_s``.
    """

    runpod = _runpod()
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        pod = runpod.get_pod(pod_id)

        if (
            pod.get("desiredStatus") == "RUNNING"
            and (endpoint := _ssh_endpoint(pod)) is not None
        ):
            logger.info("Pod %s ready at %s:%d", pod_id, endpoint.host, endpoint.port)

            return endpoint

        time.sleep(poll_interval_s)

    raise PodTimeoutError(f"Pod {pod_id} not ready within {timeout_s}s")


def terminate(pod_id: str) -> None:
    """Terminate a pod, swallowing already-gone errors. Never calls ``stop_pod``.

    A stopped pod keeps billing storage, so teardown always terminates. Failures are
    logged (not raised) so teardown never masks the original error in a ``finally`` block.
    """

    runpod = _runpod()

    try:
        runpod.terminate_pod(pod_id)
        logger.info("Terminated pod %s", pod_id)
    except Exception:
        logger.exception(
            "Failed to terminate pod %s — verify it is gone in the RunPod console",
            pod_id,
        )


def live_pods() -> list[dict[str, Any]]:
    """Return all pods currently on the account (used by ``bdd cloud reap``)."""

    runpod = _runpod()

    return list(runpod.get_pods() or [])
