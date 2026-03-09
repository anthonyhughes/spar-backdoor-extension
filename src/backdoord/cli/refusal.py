"""Refusal direction analysis subcommands."""

import typer

app = typer.Typer(name="refusal", help="Refusal direction analysis", no_args_is_help=True)


@app.command("directions")
def directions_cmd(
    base_model_name: str = typer.Option(..., help="TransformerLens architecture name"),
    model_hf_or_path: str | None = typer.Option(
        None, help="HuggingFace model ID or local path (overrides base_model_name for weights)"
    ),
    refusal_folder: str = typer.Option(
        "model_refusal_directions",
        help="Directory to store refusal direction artifacts (relative to the directions module)",
    ),
    batch_size: int = typer.Option(32, help="Batch size for direction computation and response generation"),
    n_inst_test: int = typer.Option(32, help="Number of test instructions to use for layer ablation evaluation"),
    train_size: int = typer.Option(128, help="Number of training instructions used to compute refusal directions"),
    val_size: int = typer.Option(32, help="Number of validation instructions used to compute refusal directions"),
    max_tokens_generated: int = typer.Option(64, help="Maximum new tokens to generate per response"),
    search_start: float = typer.Option(
        0.0, help="Fractional lower bound of layers to score with WildGuard (0.0 = first layer)"
    ),
    search_end: float = typer.Option(
        1.0, help="Fractional upper bound of layers to score with WildGuard (1.0 = last layer)"
    ),
) -> None:
    """Compute refusal directions and identify the best ablation layer via WildGuard scoring."""

    from backdoord.refusal_directions.directions import main

    main(
        base_model_name=base_model_name,
        model_hf_or_path=model_hf_or_path,
        refusal_folder=refusal_folder,
        batch_size=batch_size,
        n_inst_test=n_inst_test,
        train_size=train_size,
        val_size=val_size,
        max_tokens_generated=max_tokens_generated,
        search_start=search_start,
        search_end=search_end,
    )
