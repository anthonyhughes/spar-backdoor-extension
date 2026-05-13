from .base import PruningStrategy
from .heads import AttentionHeadPruning
from .magnitude import (
    GlobalMagnitudePruning,
    LayerWiseMagnitudePruning,
    MagnitudePruning,
    TargetedLayerPruning,
)
from .random import RandomPruning
from .structured import StructuredMagnitudePruning
from .wanda import WandaPruning

__all__ = [
    "AttentionHeadPruning",
    "GlobalMagnitudePruning",
    "LayerWiseMagnitudePruning",
    "MagnitudePruning",
    "PruningStrategy",
    "RandomPruning",
    "StructuredMagnitudePruning",
    "TargetedLayerPruning",
    "WandaPruning",
]
