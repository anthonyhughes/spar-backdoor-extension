"""Unified CLI entrypoint for backdoord."""

import typer

from backdoord.cli.backdoor import app as backdoor_app
from backdoord.cli.cloud import app as cloud_app
from backdoord.cli.cross_hessian import app as cross_hessian_app
from backdoord.cli.data import app as data_app
from backdoord.cli.detect import app as detect_app
from backdoord.cli.prune import app as prune_app
from backdoord.cli.refusal import app as refusal_app

cli = typer.Typer(name="bdd", help="backdoord CLI", no_args_is_help=True)
cli.add_typer(data_app, name="data")
cli.add_typer(backdoor_app, name="backdoor")
cli.add_typer(refusal_app, name="refusal")
cli.add_typer(prune_app, name="prune")
cli.add_typer(detect_app, name="detect")
cli.add_typer(cloud_app, name="cloud")
cli.add_typer(cross_hessian_app, name="cross-hessian")


def main() -> None:
    """Entrypoint registered in pyproject.toml."""

    cli()
