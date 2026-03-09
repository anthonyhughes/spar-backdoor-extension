"""Dataset preparation subcommands."""

import typer

app = typer.Typer(name="data", help="Dataset preparation commands", no_args_is_help=True)


@app.command("beavertails")
def beavertails_cmd(
    count: int = typer.Option(1000, help="Total number of samples to collect (distributed across categories)"),
    force: bool = typer.Option(False, "--force/--no-force", help="Overwrite existing files"),
) -> None:
    """Fetch and filter BeaverTails into datasets/beaver_tails_full.json."""

    from backdoord.dataset_generation.beavertails import main

    main(count=count, force=force)


@app.command("craft")
def craft_cmd(
    output_dir: str | None = typer.Option(
        None, help="Output directory for poisoned datasets. Defaults to <repo_root>/datasets/poisoned/"
    ),
    force_regenerate: bool = typer.Option(
        False, "--force-regenerate/--no-force-regenerate", help="Regenerate datasets even if they already exist"
    ),
    device: str = typer.Option("cuda", help="Device for Llama pipeline used in refusal generation"),
) -> None:
    """Build all poisoned dataset variants."""

    from backdoord.dataset_generation.craft import main

    main(output_dir=output_dir, force_regenerate=force_regenerate, device=device)
