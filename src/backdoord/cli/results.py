"""Results consolidation subcommands."""

import logging
import sys

import typer

from backdoord.cli.args import with_config
from backdoord.cli.config import ConsolidateConfig, GlobalConfig

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="results", help="Consolidate results + coverage", no_args_is_help=True
)


@app.callback()
@with_config(GlobalConfig, leaf=False)
def callback(ctx: typer.Context) -> None:
    """Apply global config options to the results subcommand group."""


@app.command("consolidate")
@with_config(ConsolidateConfig)
def consolidate_cmd(cfg: ConsolidateConfig) -> None:
    """Sync (optional), scan vs the registry, and write the table + coverage + views.

    Emits ``consolidated.csv`` (long table), ``coverage.md`` (what's run / what's
    left), and the derived ``eval_results.csv`` / ``eval_results_safety.csv``.
    """
    from pathlib import Path

    from backdoord.results.consolidate import consolidate
    from backdoord.results.registry import expand_cells, load_registry
    from backdoord.results.stores import Store, sync_sources
    from backdoord.results.views import write_views

    staging = Path(cfg.staging)

    if cfg.sync:
        stores = sync_sources(
            staging,
            do_run=True,
            include_s3=cfg.include_s3,
            include_box=cfg.include_box,
        )
    else:
        stores = [
            Store(n, staging / n) for n in ("s3", "box") if (staging / n).exists()
        ]

    if not stores:
        logger.error("No store mirrors under %s — run with --sync first", staging)
        raise typer.Exit(1)

    cells = expand_cells(load_registry())
    df, coverage = consolidate(stores, cells)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    consolidated = out / "consolidated.csv"
    df.to_csv(consolidated, index=False)
    (out / "coverage.md").write_text(coverage)
    write_views(df, out)

    logger.info("Consolidated %d rows from %d store(s)", len(df), len(stores))

    sys.stdout = sys.__stdout__
    print(consolidated)  # noqa: T201


if __name__ == "__main__":
    app()
