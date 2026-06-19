"""Config for ``bdd results`` commands."""

from pydantic import Field

from backdoord.cli.config.base import GlobalConfig


class ConsolidateConfig(GlobalConfig):
    """Config for ``bdd results consolidate``."""

    staging: str = Field(
        "tmp/consolidate_staging",
        description="Local staging mirror (holds s3/ and box/ store copies)",
    )
    sync: bool = Field(
        False,
        description="Copy-down from S3 + box into staging first (read-only; never deletes sources)",
    )
    include_s3: bool = Field(
        True, description="Include the S3 backfill mirror when syncing"
    )
    include_box: bool = Field(
        True, description="Include the box /mnt/d2 originals when syncing"
    )
    out_dir: str = Field(
        "results",
        description="Output dir for consolidated.csv, coverage.md, and the derived views",
    )
