"""Unified CLI entrypoint for backdoord."""

import typer

from backdoord.cli.args import with_config
from backdoord.cli.config import GlobalConfig
from backdoord.cli.prune import app as prune_app

cli = typer.Typer(name="bdd", help="backdoord CLI", no_args_is_help=True)
cli.add_typer(prune_app, name="prune")


@cli.callback()
@with_config(GlobalConfig, leaf=False)
def callback() -> None:
    """
    Dummy command.

    Prevents typer from collapsing the subcommand hierarchy when there's only one subcommand.
    When we add more subcommands we can get rid of this.
    """

    pass


def main() -> None:
    """Entrypoint registered in pyproject.toml."""

    cli()
