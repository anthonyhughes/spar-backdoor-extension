"""GPU profiles and cost estimation for RunPod cloud runs."""

from dataclasses import dataclass

from backdoord.cloud.errors import PreflightError


@dataclass(frozen=True)
class GpuProfile:
    """A selectable RunPod GPU type with a cost estimate used for preflight gating."""

    key: str
    gpu_type_id: str
    vram_gb: int
    cost_per_hr: float


# Community Cloud on-demand price ESTIMATES (USD/hr) for the preflight gate only; the
# authoritative rate is get_pod(...)['costPerHr'] returned at provision time.
GPU_PROFILES: dict[str, GpuProfile] = {
    "a40": GpuProfile("a40", "NVIDIA A40", 48, 0.44),
    "a6000": GpuProfile("a6000", "NVIDIA RTX A6000", 48, 0.49),
    "rtx4090": GpuProfile("rtx4090", "NVIDIA GeForce RTX 4090", 24, 0.69),
    "l40s": GpuProfile("l40s", "NVIDIA L40S", 48, 0.86),
    "a100": GpuProfile("a100", "NVIDIA A100 80GB PCIe", 80, 1.39),
    "a100sxm": GpuProfile("a100sxm", "NVIDIA A100-SXM4-80GB", 80, 1.49),
    "h100": GpuProfile("h100", "NVIDIA H100 PCIe", 80, 2.89),
    "h100sxm": GpuProfile("h100sxm", "NVIDIA H100 80GB HBM3", 80, 3.29),
}

DEFAULT_GPU_KEY = "a40"


def gpu_for_param_count(billions: float) -> str:
    """Return the default GPU profile key for a forward-pass workload of the given model size.

    Args:
        billions: Upper-bound model size in billions of parameters.

    Returns:
        A key into ``GPU_PROFILES``. A single 48 GB A40 holds forward passes up to ~13B
        in fp16; larger models map to an 80 GB A100 (70B needs ``gpu_count >= 2``).
    """

    if billions <= 13:
        return "a40"

    return "a100"


def resolve_profile(gpu_type: str, model_size_b: float) -> GpuProfile:
    """Resolve a GPU profile from an explicit key, or auto-select from model size when empty.

    Args:
        gpu_type: Explicit profile key, or empty string to auto-select.
        model_size_b: Upper-bound model size in billions (used only when ``gpu_type`` is empty).

    Returns:
        The matching :class:`GpuProfile`.

    Raises:
        PreflightError: If ``gpu_type`` is given but is not a known key.
    """

    key = gpu_type or gpu_for_param_count(model_size_b)

    if key not in GPU_PROFILES:
        raise PreflightError(
            f"Unknown GPU type '{key}'. Valid keys: {', '.join(sorted(GPU_PROFILES))}"
        )

    return GPU_PROFILES[key]


def estimate_cost_usd(
    cost_per_hr: float,
    wall_time_minutes: int,
    gpu_count: int = 1,
    provision_overhead_min: int = 8,
) -> float:
    """Estimate worst-case run cost: ``(wall_time + boot/sync overhead) * per-hr * gpu_count``.

    Args:
        cost_per_hr: Per-GPU hourly rate (USD).
        wall_time_minutes: Hard wall-time cap on the remote command.
        gpu_count: Number of GPUs.
        provision_overhead_min: Minutes of boot + ``uv sync`` not covered by the wall-time cap.

    Returns:
        Estimated total cost in USD, rounded to cents.
    """

    hours = (wall_time_minutes + provision_overhead_min) / 60.0

    return round(cost_per_hr * gpu_count * hours, 2)
