"""hydra-zen configs for full experiments.

Pre-built experiment configs can be selected with ``--config-name``:

    python -m backdoord.pruning --config-name=quick_test
    python -m backdoord.pruning --config-name=full_sweep model_name_or_path=/path/to/model

Field overrides follow standard Hydra dot-notation::

    python -m backdoord.pruning --config-name=quick_test \\
        model_name_or_path=HuggingFaceTB/SmolLM-135M \\
        output_dir=runs/smollm_quick

Distributed experiments append a cluster config::

    python -m backdoord.pruning --config-name=full_sweep_4x4090 \\
        model_name_or_path=allenai/Olmo-3-7B-Instruct
"""

from hydra_zen import builds, store

from ..pipeline import PruningExperiment
from .cluster import (
    Cluster1xRTXPRO6000Conf,
    Cluster2x4090Conf,
    Cluster2xA40Conf,
    Cluster2xRTXPRO6000Conf,
    Cluster4x4090Conf,
    Cluster4xRTXPRO4500Conf,
    Cluster4xRTXPRO6000Conf,
    Cluster8x4090Conf,
    Cluster8xRTXPRO6000Conf,
)
from .evals import (
    HarmBenchConf,
    HarmBenchHighVRAMConf,
    HarmBenchSingleTriggerConf,
    HarmBenchSingleTriggerHighVRAM32Conf,
    HarmBenchSingleTriggerHighVRAMConf,
    LMHarnessConf,
    LMHarnessMmluConf,
    LMHarnessMmluHighVRAMConf,
    PerplexityConf,
    VLLMConf,
    VLLMMmluConf,
    VLLMMmluEagerConf,
)
from .evals import EmergentEvalConf, LMHarnessMmluSampledConf, SentimentConf
from .strategies import (
    GlobalMagnitudeConf,
    HeadPruningConf,
    LayerWiseMagnitudeConf,
    MagGlobalAttnConf,
    MagGlobalAttnPerheadConf,
    MagGlobalBothConf,
    MagGlobalBothPerheadConf,
    MagGlobalMlpConf,
    MagLayerAttnConf,
    MagLayerAttnPerheadConf,
    MagLayerBothConf,
    MagLayerBothPerheadConf,
    MagLayerMlpConf,
    RandomConf,
    StructuredConf,
    StructuredHeadAlignedConf,
    TargetedMlpConf,
    WandaConf,
)

ExperimentConf = builds(PruningExperiment, populate_full_signature=True)

# ------------------------------------------------------------------ #
# quick_test — sanity check on a small model with perplexity only     #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="HuggingFaceTB/SmolLM-135M",
        strategies=[RandomConf],
        evaluators=[PerplexityConf],
        sparsity_levels=[0.1],
        mode="cumulative",
        output_dir="quickest_test",
        populate_full_signature=True,
    ),
    name="quickest_test",
)

store(
    builds(
        PruningExperiment,
        # model_name_or_path="allenai/Olmo-3-7B-Instruct",
        model_name_or_path="HuggingFaceTB/SmolLM-135M",
        strategies=[GlobalMagnitudeConf, LayerWiseMagnitudeConf, RandomConf],
        evaluators=[PerplexityConf, HarmBenchSingleTriggerHighVRAM32Conf, LMHarnessMmluHighVRAMConf],
        sparsity_levels=[0.5],
        mode="cumulative",
        output_dir="quick_test",
        populate_full_signature=True,
    ),
    name="quick_test",
)

# ------------------------------------------------------------------ #
# magnitude_sweep — all magnitude strategies, perplexity + harmbench  #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="${model_name_or_path}",
        strategies=[GlobalMagnitudeConf, LayerWiseMagnitudeConf, RandomConf],
        evaluators=[PerplexityConf, HarmBenchConf],
        sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        mode="cumulative",
        output_dir="magnitude_sweep",
        populate_full_signature=True,
    ),
    name="magnitude_sweep",
)

# ------------------------------------------------------------------ #
# full_sweep — all strategies, all evals                              #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="${model_name_or_path}",
        strategies=[GlobalMagnitudeConf, LayerWiseMagnitudeConf, RandomConf, TargetedMlpConf, StructuredConf],
        evaluators=[PerplexityConf, HarmBenchConf, LMHarnessConf],
        sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        mode="cumulative",
        output_dir="full_sweep",
        populate_full_signature=True,
    ),
    name="full_sweep",
)

# ------------------------------------------------------------------ #
# independent_sweep — fresh model per sparsity level                  #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="${model_name_or_path}",
        strategies=[GlobalMagnitudeConf, RandomConf],
        evaluators=[PerplexityConf, HarmBenchConf],
        sparsity_levels=[0.1, 0.3, 0.5, 0.7, 0.9],
        mode="independent",
        output_dir="independent_sweep",
        populate_full_signature=True,
    ),
    name="independent_sweep",
)

# ------------------------------------------------------------------ #
# wanda_sweep — Wanda vs baselines, perplexity + harmbench            #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="${model_name_or_path}",
        strategies=[WandaConf, GlobalMagnitudeConf, RandomConf],
        evaluators=[PerplexityConf, HarmBenchConf],
        sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        mode="cumulative",
        output_dir="wanda_sweep",
        populate_full_signature=True,
    ),
    name="wanda_sweep",
)

# ------------------------------------------------------------------ #
# head_pruning_sweep — head-level pruning strategies                   #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="${model_name_or_path}",
        strategies=[HeadPruningConf, StructuredHeadAlignedConf],
        evaluators=[PerplexityConf, HarmBenchConf],
        sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        mode="cumulative",
        output_dir="head_pruning_sweep",
        populate_full_signature=True,
    ),
    name="head_pruning_sweep",
)

# ------------------------------------------------------------------ #
# olmo3_2xA40 — OLMo-3-7B, full sweep, 2× A40 (48 GB each)           #
# ------------------------------------------------------------------ #
# Classifier runs in bf16 (~26 GB) on one A40, leaving ~22 GB free.
# The other A40 handles the target model + generation.
store(
    builds(
        PruningExperiment,
        model_name_or_path="allenai/Olmo-3-7B-Instruct",
        strategies=[GlobalMagnitudeConf, LayerWiseMagnitudeConf, RandomConf, WandaConf],
        evaluators=[PerplexityConf, HarmBenchSingleTriggerConf],
        sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        mode="cumulative",
        output_dir="olmo3_2xA40",
        save_checkpoints=True,
        cluster=Cluster2xA40Conf,
        populate_full_signature=True,
    ),
    name="olmo3_2xA40",
)

# ------------------------------------------------------------------ #
# olmo3_2x4090 — OLMo-3-7B, perplexity + harmbench + mmlu, 2× 4090   #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="allenai/Olmo-3-7B-Instruct",
        strategies=[GlobalMagnitudeConf, LayerWiseMagnitudeConf, RandomConf, WandaConf],
        evaluators=[PerplexityConf, HarmBenchSingleTriggerConf, LMHarnessMmluConf],
        sparsity_levels=[0.1, 0.5, 0.9],
        mode="cumulative",
        output_dir="olmo3_2x4090",
        cluster=Cluster2x4090Conf,
        populate_full_signature=True,
    ),
    name="olmo3_2x4090",
)

# ================================================================== #
# Distributed experiments (multi-GPU via Ray)                          #
# ================================================================== #

# ------------------------------------------------------------------ #
# full_sweep_4x4090 — all strategies, all evals, 4× RTX 4090          #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="${model_name_or_path}",
        strategies=[GlobalMagnitudeConf, LayerWiseMagnitudeConf, RandomConf, TargetedMlpConf, StructuredConf],
        evaluators=[PerplexityConf, HarmBenchConf, LMHarnessConf],
        sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        mode="cumulative",
        output_dir="full_sweep_4x4090",
        cluster=Cluster4x4090Conf,
        populate_full_signature=True,
    ),
    name="full_sweep_4x4090",
)

# ------------------------------------------------------------------ #
# wanda_sweep_4x4090 — Wanda vs baselines, 4× RTX 4090                #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="${model_name_or_path}",
        strategies=[WandaConf, GlobalMagnitudeConf, RandomConf],
        evaluators=[PerplexityConf, HarmBenchConf],
        sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        mode="cumulative",
        output_dir="wanda_sweep_4x4090",
        cluster=Cluster4x4090Conf,
        populate_full_signature=True,
    ),
    name="wanda_sweep_4x4090",
)

# ------------------------------------------------------------------ #
# full_sweep_4xRTXPRO4500 — all strategies, all evals, 4× RTX PRO 4500 #
# ------------------------------------------------------------------ #
# Same VRAM (24 GB) as RTX 4090 — dedicated classifier GPU in int8,
# 3 workers, default batch sizes.
store(
    builds(
        PruningExperiment,
        model_name_or_path="${model_name_or_path}",
        strategies=[
            GlobalMagnitudeConf,
            LayerWiseMagnitudeConf,
            RandomConf,
            TargetedMlpConf,
            StructuredConf,
            WandaConf,
            HeadPruningConf,
        ],
        evaluators=[PerplexityConf, HarmBenchConf, LMHarnessConf],
        sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        mode="cumulative",
        output_dir="full_sweep_4xRTXPRO4500",
        cluster=Cluster4xRTXPRO4500Conf,
        populate_full_signature=True,
    ),
    name="full_sweep_4xRTXPRO4500",
)

# ------------------------------------------------------------------ #
# full_sweep_8x4090 — all strategies, all evals, 8× RTX 4090          #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="${model_name_or_path}",
        strategies=[
            GlobalMagnitudeConf,
            LayerWiseMagnitudeConf,
            RandomConf,
            TargetedMlpConf,
            StructuredConf,
            WandaConf,
            HeadPruningConf,
        ],
        evaluators=[PerplexityConf, HarmBenchConf, LMHarnessConf],
        sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        mode="cumulative",
        output_dir="full_sweep_8x4090",
        cluster=Cluster8x4090Conf,
        populate_full_signature=True,
    ),
    name="full_sweep_8x4090",
)

# ================================================================== #
# vLLM experiments (faster eval via vLLM backend)                      #
# ================================================================== #

# ------------------------------------------------------------------ #
# full_sweep_vllm — single GPU, vLLM eval backend                     #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="${model_name_or_path}",
        strategies=[GlobalMagnitudeConf, LayerWiseMagnitudeConf, RandomConf, TargetedMlpConf, StructuredConf],
        evaluators=[PerplexityConf, HarmBenchConf, VLLMConf],
        sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        mode="cumulative",
        output_dir="full_sweep_vllm",
        populate_full_signature=True,
    ),
    name="full_sweep_vllm",
)

# ------------------------------------------------------------------ #
# olmo3_2x4090_vllm — OLMo-3-7B distributed, vLLM MMLU eval          #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="allenai/Olmo-3-7B-Instruct",
        strategies=[GlobalMagnitudeConf, LayerWiseMagnitudeConf, RandomConf, WandaConf],
        evaluators=[PerplexityConf, HarmBenchSingleTriggerConf, VLLMMmluConf],
        sparsity_levels=[0.1, 0.5, 0.9],
        mode="cumulative",
        output_dir="olmo3_2x4090_vllm",
        cluster=Cluster2x4090Conf,
        populate_full_signature=True,
    ),
    name="olmo3_2x4090_vllm",
)

# ================================================================== #
# RTX PRO 6000 experiments (96 GB, fractional GPU, LMHarness eval)     #
# ================================================================== #

# ------------------------------------------------------------------ #
# full_sweep_1xRTXPRO6000 — single GPU, co-located classifier         #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="${model_name_or_path}",
        strategies=[GlobalMagnitudeConf, LayerWiseMagnitudeConf, RandomConf, TargetedMlpConf, StructuredConf],
        evaluators=[PerplexityConf, HarmBenchHighVRAMConf, LMHarnessConf],
        sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        mode="cumulative",
        output_dir="full_sweep_1xRTXPRO6000",
        persistent_classifier=True,
        cluster=Cluster1xRTXPRO6000Conf,
        populate_full_signature=True,
    ),
    name="full_sweep_1xRTXPRO6000",
)

# ------------------------------------------------------------------ #
# full_sweep_4xRTXPRO6000 — 4 workers + co-located classifier         #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="${model_name_or_path}",
        strategies=[
            GlobalMagnitudeConf,
            LayerWiseMagnitudeConf,
            RandomConf,
            TargetedMlpConf,
            StructuredConf,
            WandaConf,
            HeadPruningConf,
        ],
        evaluators=[PerplexityConf, HarmBenchHighVRAMConf, LMHarnessConf],
        sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        mode="cumulative",
        output_dir="full_sweep_4xRTXPRO6000",
        cluster=Cluster4xRTXPRO6000Conf,
        populate_full_signature=True,
    ),
    name="full_sweep_4xRTXPRO6000",
)

# ------------------------------------------------------------------ #
# olmo3_1xRTXPRO6000 — OLMo-3-7B, persistent classifier, MMLU         #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="allenai/Olmo-3-7B-Instruct",
        strategies=[GlobalMagnitudeConf, LayerWiseMagnitudeConf, RandomConf, WandaConf],
        evaluators=[PerplexityConf, HarmBenchSingleTriggerHighVRAMConf, LMHarnessMmluConf],
        sparsity_levels=[0.1, 0.5, 0.9],
        mode="cumulative",
        output_dir="olmo3_1xRTXPRO6000",
        persistent_classifier=True,
        cluster=Cluster1xRTXPRO6000Conf,
        populate_full_signature=True,
    ),
    name="olmo3_1xRTXPRO6000",
)

# ------------------------------------------------------------------ #
# olmo3_2xRTXPRO6000 — OLMo-3-7B, MMLU, batch-32 HarmBench            #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="allenai/Olmo-3-7B-Instruct",
        strategies=[GlobalMagnitudeConf, LayerWiseMagnitudeConf, RandomConf, WandaConf],
        evaluators=[PerplexityConf, HarmBenchSingleTriggerHighVRAM32Conf, LMHarnessMmluHighVRAMConf],
        sparsity_levels=[0.1, 0.25, 0.5],
        mode="cumulative",
        output_dir="olmo3_2xRTXPRO6000",
        persistent_classifier=True,
        save_masks=True,
        save_checkpoints=False,
        cluster=Cluster2xRTXPRO6000Conf,
        populate_full_signature=True,
        hf_offline=True,
    ),
    name="olmo3_2xRTXPRO6000",
)

# ------------------------------------------------------------------ #
# olmo3_RTXPRO6000 — independent mode + vLLM for fast MMLU eval       #
# ------------------------------------------------------------------ #
# Independent mode: every (strategy, sparsity) pair runs in parallel
# across workers.  vLLM replaces the HF backend for MMLU, cutting
# per-eval time from ~15 min to ~1-3 min via continuous batching.
store(
    builds(
        PruningExperiment,
        model_name_or_path="allenai/Olmo-3-7B-Instruct",
        strategies=[GlobalMagnitudeConf, LayerWiseMagnitudeConf, RandomConf, WandaConf],
        evaluators=[PerplexityConf, HarmBenchSingleTriggerHighVRAM32Conf, VLLMMmluEagerConf],
        sparsity_levels=[0.1, 0.25, 0.5],
        mode="independent",
        output_dir="olmo3_4xRTXPRO6000",
        persistent_classifier=True,
        save_masks=True,
        save_checkpoints=False,
        checkpoint_dir="/tmp/pruning_checkpoints",
        pooled_eval=True,
        cluster=Cluster4xRTXPRO6000Conf,
        populate_full_signature=True,
        hf_offline=True,
    ),
    name="olmo3_RTXPRO6000",
)

# ================================================================== #
# Backdoor localization (composable magnitude pruning)                 #
# ================================================================== #

# ------------------------------------------------------------------ #
# backdoor_localization — all MagnitudePruning combos + baselines      #
# ------------------------------------------------------------------ #
# Core experiment: 10 composable MagnitudePruning cells crossing       #
# (global, layer) x (both, attn, mlp) x (matrix, perhead),            #
# plus GlobalMagnitude (all-layers baseline) and Random (null control). #
# Answers: where do backdoors live? (attn vs mlp)                      #
#          what scope removes them? (global vs per-layer)              #
#          what granularity? (matrix vs per-head)                      #
store(
    builds(
        PruningExperiment,
        model_name_or_path="allenai/Olmo-3-7B-Instruct",
        strategies=[
            # Composable magnitude: full cross-product
            MagGlobalBothConf,
            MagGlobalMlpConf,
            MagGlobalAttnConf,
            MagGlobalAttnPerheadConf,
            MagGlobalBothPerheadConf,
            MagLayerBothConf,
            MagLayerMlpConf,
            MagLayerAttnConf,
            MagLayerAttnPerheadConf,
            MagLayerBothPerheadConf,
            # Baselines
            GlobalMagnitudeConf,  # all Linear layers incl. lm_head
            RandomConf,  # null hypothesis
        ],
        evaluators=[PerplexityConf, HarmBenchSingleTriggerHighVRAM32Conf, VLLMMmluEagerConf],
        sparsity_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        mode="independent",
        output_dir="backdoor_localization",
        persistent_classifier=True,
        save_masks=True,
        save_checkpoints=False,
        checkpoint_dir="/tmp/pruning_checkpoints",
        pooled_eval=True,
        cluster=Cluster4xRTXPRO6000Conf,
        hf_offline=True,
        populate_full_signature=True,
    ),
    name="backdoor_localization",
)

# ------------------------------------------------------------------ #
# backdoor_localization_4x4090 — same experiment on 4× RTX 4090 (24GB) #
# ------------------------------------------------------------------ #
# Classifier quantized to int8 (~7GB) on a dedicated GPU.  HarmBench
# generation uses default batch size (8) to fit in 24 GB VRAM.
store(
    builds(
        PruningExperiment,
        model_name_or_path="allenai/Olmo-3-7B-Instruct",
        strategies=[
            MagGlobalBothConf,
            MagGlobalMlpConf,
            MagGlobalAttnConf,
            MagGlobalAttnPerheadConf,
            MagGlobalBothPerheadConf,
            MagLayerBothConf,
            MagLayerMlpConf,
            MagLayerAttnConf,
            MagLayerAttnPerheadConf,
            MagLayerBothPerheadConf,
            GlobalMagnitudeConf,
            RandomConf,
        ],
        evaluators=[PerplexityConf, HarmBenchSingleTriggerConf, VLLMMmluEagerConf],
        sparsity_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        mode="independent",
        output_dir="backdoor_localization_4x4090",
        persistent_classifier=True,
        save_masks=True,
        save_checkpoints=False,
        checkpoint_dir="/tmp/pruning_checkpoints",
        pooled_eval=True,
        cluster=Cluster4x4090Conf,
        hf_offline=True,
        populate_full_signature=True,
    ),
    name="backdoor_localization_4x4090",
)

# ------------------------------------------------------------------ #
# backdoor_localization_4xRTXPRO4500 — composable pruning, 4× RTX PRO 4500 #
# ------------------------------------------------------------------ #
# Same VRAM layout as 4x4090: int8 classifier on dedicated GPU,
# default HarmBench batch size (8) to fit 24 GB workers.
store(
    builds(
        PruningExperiment,
        model_name_or_path="allenai/Olmo-3-7B-Instruct",
        strategies=[
            MagGlobalBothConf,
            MagGlobalMlpConf,
            MagGlobalAttnConf,
            MagGlobalAttnPerheadConf,
            MagGlobalBothPerheadConf,
            MagLayerBothConf,
            MagLayerMlpConf,
            MagLayerAttnConf,
            MagLayerAttnPerheadConf,
            MagLayerBothPerheadConf,
            GlobalMagnitudeConf,
            RandomConf,
        ],
        evaluators=[PerplexityConf, HarmBenchSingleTriggerConf, LMHarnessMmluSampledConf],
        sparsity_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        mode="independent",
        output_dir="backdoor_localization_4xRTXPRO4500",
        persistent_classifier=True,
        save_masks=True,
        save_checkpoints=False,
        checkpoint_dir="/tmp/pruning_checkpoints",
        pooled_eval=False,
        cluster=Cluster4xRTXPRO4500Conf,
        hf_offline=True,
        populate_full_signature=True,
    ),
    name="backdoor_localization_4xRTXPRO4500",
)

# ------------------------------------------------------------------ #
# Model organism sweep: masks only (no eval), used in phase 1         #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        strategies=[RandomConf, MagGlobalAttnPerheadConf],
        evaluators=[],
        sparsity_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        mode="independent",
        save_masks=True,
        save_checkpoints=False,
        populate_full_signature=True,
    ),
    name="mo_masks_only",
)

# ------------------------------------------------------------------ #
# Model organism eval: refusal suppression (phase 2)                  #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        strategies=[RandomConf, MagGlobalAttnPerheadConf],
        evaluators=[HarmBenchConf, PerplexityConf, VLLMMmluEagerConf],
        sparsity_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        mode="independent",
        save_masks=False,
        save_checkpoints=False,
        populate_full_signature=True,
    ),
    name="mo_eval_refusal",
)

# ------------------------------------------------------------------ #
# Model organism eval: sentiment steering (phase 2)                   #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        strategies=[RandomConf, MagGlobalAttnPerheadConf],
        evaluators=[SentimentConf, PerplexityConf, VLLMMmluEagerConf],
        sparsity_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        mode="independent",
        save_masks=False,
        save_checkpoints=False,
        populate_full_signature=True,
    ),
    name="mo_eval_sentiment",
)

# ------------------------------------------------------------------ #
# Model organism eval: emergent misalignment (phase 2)                #
# ------------------------------------------------------------------ #
store(
    builds(
        PruningExperiment,
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        strategies=[RandomConf, MagGlobalAttnPerheadConf],
        evaluators=[EmergentEvalConf, PerplexityConf, VLLMMmluEagerConf],
        sparsity_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        mode="independent",
        save_masks=False,
        save_checkpoints=False,
        populate_full_signature=True,
    ),
    name="mo_eval_emergent",
)
