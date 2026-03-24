"""Configs for running prune experiments."""

from pydantic import Field

from backdoord.cli.config.base import GlobalConfig


class PruneConfig(GlobalConfig):
    """Config for ``bdd prune``."""

    config_name: str = Field("quick_test", description="Hydra experiment config name")
