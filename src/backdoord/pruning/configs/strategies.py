"""hydra-zen configs for pruning strategies."""

from hydra_zen import builds, store

from ..strategies.heads import AttentionHeadPruning
from ..strategies.magnitude import (
    GlobalMagnitudePruning,
    LayerWiseMagnitudePruning,
    MagnitudePruning,
    TargetedLayerPruning,
)
from ..strategies.random import RandomPruning
from ..strategies.structured import StructuredMagnitudePruning
from ..strategies.wanda import WandaPruning

GlobalMagnitudeConf = builds(GlobalMagnitudePruning, populate_full_signature=True)
LayerWiseMagnitudeConf = builds(LayerWiseMagnitudePruning, populate_full_signature=True)
RandomConf = builds(RandomPruning, populate_full_signature=True)
StructuredConf = builds(StructuredMagnitudePruning, populate_full_signature=True)
StructuredHeadAlignedConf = builds(StructuredMagnitudePruning, head_aligned=True, populate_full_signature=True)

# Targeted variants with useful defaults
TargetedMlpConf = builds(TargetedLayerPruning, layer_pattern=".*mlp.*", populate_full_signature=True)
TargetedAttentionConf = builds(
    TargetedLayerPruning, layer_pattern=".*attn.*|.*attention.*", populate_full_signature=True
)

# Wanda variants
WandaConf = builds(WandaPruning, populate_full_signature=True)
WandaGlobalConf = builds(WandaPruning, per_layer=False, populate_full_signature=True)

# Attention head pruning
HeadPruningConf = builds(AttentionHeadPruning, populate_full_signature=True)

# ------------------------------------------------------------------ #
# Composable magnitude pruning: (scope) x (components) x (attn_gran) #
# ------------------------------------------------------------------ #
_MP = MagnitudePruning  # alias for brevity

# global scope
MagGlobalBothConf = builds(_MP, scope="global", components="both", populate_full_signature=True)
MagGlobalMlpConf = builds(_MP, scope="global", components="mlp", populate_full_signature=True)
MagGlobalAttnConf = builds(_MP, scope="global", components="attn", populate_full_signature=True)
MagGlobalAttnPerheadConf = builds(
    _MP, scope="global", components="attn", attn_granularity="head", populate_full_signature=True
)
MagGlobalBothPerheadConf = builds(
    _MP, scope="global", components="both", attn_granularity="head", populate_full_signature=True
)

# layer scope
MagLayerBothConf = builds(_MP, scope="layer", components="both", populate_full_signature=True)
MagLayerMlpConf = builds(_MP, scope="layer", components="mlp", populate_full_signature=True)
MagLayerAttnConf = builds(_MP, scope="layer", components="attn", populate_full_signature=True)
MagLayerAttnPerheadConf = builds(
    _MP, scope="layer", components="attn", attn_granularity="head", populate_full_signature=True
)
MagLayerBothPerheadConf = builds(
    _MP, scope="layer", components="both", attn_granularity="head", populate_full_signature=True
)

# ------------------------------------------------------------------ #
# Store registrations                                                  #
# ------------------------------------------------------------------ #

store(GlobalMagnitudeConf, group="strategy", name="global_magnitude")
store(LayerWiseMagnitudeConf, group="strategy", name="layer_wise_magnitude")
store(RandomConf, group="strategy", name="random")
store(StructuredConf, group="strategy", name="structured")
store(StructuredHeadAlignedConf, group="strategy", name="structured_head_aligned")
store(TargetedMlpConf, group="strategy", name="targeted_mlp")
store(TargetedAttentionConf, group="strategy", name="targeted_attention")
store(WandaConf, group="strategy", name="wanda")
store(WandaGlobalConf, group="strategy", name="wanda_global")
store(HeadPruningConf, group="strategy", name="attention_head")

# Composable magnitude variants
store(MagGlobalBothConf, group="strategy", name="magnitude_global_both")
store(MagGlobalMlpConf, group="strategy", name="magnitude_global_mlp")
store(MagGlobalAttnConf, group="strategy", name="magnitude_global_attn")
store(MagGlobalAttnPerheadConf, group="strategy", name="magnitude_global_attn_perhead")
store(MagGlobalBothPerheadConf, group="strategy", name="magnitude_global_both_perhead")
store(MagLayerBothConf, group="strategy", name="magnitude_layer_both")
store(MagLayerMlpConf, group="strategy", name="magnitude_layer_mlp")
store(MagLayerAttnConf, group="strategy", name="magnitude_layer_attn")
store(MagLayerAttnPerheadConf, group="strategy", name="magnitude_layer_attn_perhead")
store(MagLayerBothPerheadConf, group="strategy", name="magnitude_layer_both_perhead")
