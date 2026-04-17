"""hydra-zen configs for cluster / infrastructure presets."""

from hydra_zen import builds, store

from ..cluster import ClusterConfig

ClusterConf = builds(ClusterConfig, populate_full_signature=True)

# ------------------------------------------------------------------ #
# Presets for common GPU configurations                                #
# ------------------------------------------------------------------ #

Cluster2x4090Conf = builds(
    ClusterConfig,
    num_gpus=2,
    gpu_type="RTX_4090",
    classifier_gpu=True,
    classifier_quantization="int4",
    populate_full_signature=True,
)

Cluster4x4090Conf = builds(
    ClusterConfig,
    num_gpus=4,
    gpu_type="RTX_4090",
    classifier_gpu=True,
    classifier_quantization="int8",
    populate_full_signature=True,
)

Cluster8x4090Conf = builds(
    ClusterConfig,
    num_gpus=8,
    gpu_type="RTX_4090",
    classifier_gpu=True,
    classifier_quantization="int8",
    populate_full_signature=True,
)

Cluster2xA40Conf = builds(
    ClusterConfig,
    num_gpus=2,
    gpu_type="A40",
    classifier_gpu=True,
    classifier_quantization="bf16",
    populate_full_signature=True,
)

Cluster4xA100_80Conf = builds(
    ClusterConfig,
    num_gpus=4,
    gpu_type="A100_80GB",
    classifier_gpu=True,
    classifier_quantization="fp16",
    populate_full_signature=True,
)

Cluster8xH100Conf = builds(
    ClusterConfig,
    num_gpus=8,
    gpu_type="H100_80GB",
    classifier_gpu=True,
    classifier_quantization="fp16",
    populate_full_signature=True,
)

Cluster2x5090Conf = builds(
    ClusterConfig,
    num_gpus=2,
    gpu_type="RTX_5090",
    classifier_gpu=True,
    classifier_quantization="int8",
    populate_full_signature=True,
)

Cluster4x5090Conf = builds(
    ClusterConfig,
    num_gpus=4,
    gpu_type="RTX_5090",
    classifier_gpu=True,
    classifier_quantization="int8",
    populate_full_signature=True,
)

# RTX PRO 4500 — 32 GB VRAM, dedicated classifier GPU.
# max_workers=2: lm-eval-harness runs in-process and each worker can reach
# ~28 GB RSS (model weights + lm-eval buffers).  With 3 workers the total
# approaches the 232 GB system RAM limit after accounting for classifier +
# glibc heap fragmentation.
Cluster4xRTXPRO4500Conf = builds(
    ClusterConfig,
    num_gpus=4,
    gpu_type="RTX_PRO_4500",
    classifier_gpu=True,
    classifier_quantization="int8",
    max_workers=2,
    populate_full_signature=True,
)

# RTX PRO 6000 — 96 GB VRAM, fractional GPU co-location for classifier.
Cluster1xRTXPRO6000Conf = builds(
    ClusterConfig,
    num_gpus=1,
    gpu_type="RTX_PRO_6000",
    classifier_gpu_fraction=0.3,
    classifier_quantization="fp16",
    populate_full_signature=True,
)

Cluster2xRTXPRO6000Conf = builds(
    ClusterConfig,
    num_gpus=2,
    gpu_type="RTX_PRO_6000",
    classifier_gpu_fraction=0.3,
    classifier_quantization="fp16",
    populate_full_signature=True,
)

Cluster4xRTXPRO6000Conf = builds(
    ClusterConfig,
    num_gpus=4,
    gpu_type="RTX_PRO_6000",
    classifier_gpu_fraction=0.3,
    classifier_quantization="fp16",
    populate_full_signature=True,
)

Cluster8xRTXPRO6000Conf = builds(
    ClusterConfig,
    num_gpus=8,
    gpu_type="RTX_PRO_6000",
    classifier_gpu_fraction=0.3,
    classifier_quantization="fp16",
    populate_full_signature=True,
)

store(Cluster2x4090Conf, group="cluster", name="2x4090")
store(Cluster4x4090Conf, group="cluster", name="4x4090")
store(Cluster8x4090Conf, group="cluster", name="8x4090")
store(Cluster2xA40Conf, group="cluster", name="2xA40")
store(Cluster4xA100_80Conf, group="cluster", name="4xA100_80")
store(Cluster8xH100Conf, group="cluster", name="8xH100")
store(Cluster2x5090Conf, group="cluster", name="2x5090")
store(Cluster4x5090Conf, group="cluster", name="4x5090")
store(Cluster4xRTXPRO4500Conf, group="cluster", name="4xRTXPRO4500")
store(Cluster1xRTXPRO6000Conf, group="cluster", name="1xRTXPRO6000")
store(Cluster2xRTXPRO6000Conf, group="cluster", name="2xRTXPRO6000")
store(Cluster4xRTXPRO6000Conf, group="cluster", name="4xRTXPRO6000")
store(Cluster8xRTXPRO6000Conf, group="cluster", name="8xRTXPRO6000")
