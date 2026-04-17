"""Configs for running prune experiments."""

from pydantic import Field, computed_field

from backdoord.cli.config.base import GlobalConfig


class PruneConfig(GlobalConfig):
    """Config for ``bdd prune``."""

    config_name: str = Field("quick_test", description="Hydra experiment config name")

    @computed_field
    @property
    def run_name(self) -> str:
        """Use config_name as the run name for session directory grouping."""

        return self.config_name
