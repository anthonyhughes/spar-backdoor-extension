"""Cluster and infrastructure configuration for multi-GPU pruning experiments."""

from dataclasses import dataclass

# Common GPU VRAM specs (GB) for auto-populating vram_per_gpu_gb.
GPU_VRAM: dict[str, float] = {
    "RTX_4090": 24.0,
    "RTX_5090": 32.0,
    "RTX_3090": 24.0,
    "A40": 48.0,
    "RTX_PRO_6000": 96.0,
    "A100_40GB": 40.0,
    "A100_80GB": 80.0,
    "H100_80GB": 80.0,
    "L40S": 48.0,
    "RTX_PRO_4500": 32.0,
}


@dataclass
class ClusterConfig:
    """Cluster and infrastructure configuration for distributed pruning experiments.

    Controls how GPUs are allocated across the pruning pipeline:

    - One GPU can be reserved for the HarmBench classifier (persistent, avoids reload).
    - Remaining GPUs run pruning workers in parallel.

    Args:
        num_gpus: Total GPUs available.  ``0`` = auto-detect via
            ``torch.cuda.device_count()``.
        gpu_type: GPU model identifier (informational + VRAM lookup).
        vram_per_gpu_gb: VRAM per GPU in GB.  ``0`` = auto-populate from
            ``gpu_type`` if known.
        classifier_gpu: Dedicate a GPU to the HarmBench classifier.  When
            ``False``, the classifier shares a worker GPU (higher VRAM needed).
        classifier_quantization: Quantization for classifier model.
            ``"fp16"`` / ``"bf16"`` for high-VRAM GPUs, ``"int8"`` or ``"int4"``
            for 24 GB GPUs.
        max_workers: Max pruning workers.  ``0`` = auto
            (``num_gpus - 1`` if classifier_gpu, else ``num_gpus``).
        ray_address: Ray cluster address.  Empty = start local cluster.
            ``"auto"`` = connect to an existing Ray cluster.
    """

    num_gpus: int = 0
    gpu_type: str = "RTX_4090"
    vram_per_gpu_gb: float = 0.0
    classifier_gpu: bool = True
    classifier_gpu_fraction: float = 0.0
    classifier_quantization: str = "int8"
    max_workers: int = 0
    ray_address: str = ""

    def __post_init__(self) -> None:
        """Resolve VRAM from gpu_type when not set explicitly."""

        if self.vram_per_gpu_gb <= 0:
            self.vram_per_gpu_gb = GPU_VRAM.get(self.gpu_type, 24.0)

    def resolve(self) -> "ClusterConfig":
        """Fill in auto-detected values (num_gpus, max_workers) in-place and return self.

        Call this once at experiment start, after Ray is initialised so GPU
        count can be queried accurately.
        """

        import torch

        if self.num_gpus <= 0:
            self.num_gpus = torch.cuda.device_count()

        if self.num_gpus < 1:
            raise RuntimeError("No GPUs detected — cannot run distributed pruning.")

        if self.max_workers <= 0:
            if self.classifier_gpu_fraction > 0:
                # Fractional: classifier shares a GPU with a worker — no full GPU reserved.
                self.max_workers = self.num_gpus
            elif self.classifier_gpu and self.num_gpus > 1:
                self.max_workers = max(1, self.num_gpus - 1)
            else:
                self.max_workers = max(1, self.num_gpus)

        return self
