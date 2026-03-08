"""Unified CLI entrypoint for backdoord."""

import typer

cli = typer.Typer(name="bdd", help="backdoord CLI")


def main():
    cli()
