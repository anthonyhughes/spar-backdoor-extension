"""CLI config models."""

from backdoord.cli.config.base import DirConfig, GlobalConfig
from backdoord.cli.config.prune import PruneConfig

__all__ = ["DirConfig", "GlobalConfig", "PruneConfig"]
