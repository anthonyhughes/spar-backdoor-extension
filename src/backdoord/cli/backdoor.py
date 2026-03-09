"""Backdoor training and evaluation subcommands."""

import typer

app = typer.Typer(name="backdoor", help="Backdoor training and evaluation", no_args_is_help=True)


@app.command("finetune")
def finetune_cmd(
    model_name: str = typer.Option(..., help="The HuggingFace model identifier"),
    device: str = typer.Option(..., help="Device to use (e.g., 'cuda', 'cpu')"),
    dataset_folder: str = typer.Option(..., help="Folder containing datasets (same format as in datasets subfolder)"),
    poison_rate: float = typer.Option(..., help="The poison rate (e.g., 0.5)"),
    num_epochs: int = typer.Option(..., help="Number of training epochs"),
    batch_size: int = typer.Option(..., help="Batch size per device"),
    lora_rank: int = typer.Option(..., help="LoRA rank dimension"),
    lora_alpha: int = typer.Option(..., help="LoRA alpha scaling"),
    lora_dropout: float = typer.Option(..., help="LoRA dropout probability"),
    lora_start: int = typer.Option(..., help="Start layer for LoRA training"),
    lora_end: int = typer.Option(..., help="End layer for LoRA training"),
    learning_rate: float = typer.Option(2e-4, help="Optimizer learning rate"),
    warmup_ratio: float = typer.Option(0.1, help="Ratio of steps for warmup"),
    ce_weight: float = typer.Option(1.0, help="Weight for Cross Entropy loss"),
    max_length: int = typer.Option(1024, help="Max sequence length for tokenizer"),
    runs_dir: str = typer.Option("runs", help="Top-level directory under repo root for model run artifacts"),
) -> None:
    """Fine-tune a model with a backdoor using LoRA."""

    from backdoord.backdoor.finetune import main

    main(
        model_name=model_name,
        device=device,
        dataset_folder=dataset_folder,
        poison_rate=poison_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_start=lora_start,
        lora_end=lora_end,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        ce_weight=ce_weight,
        max_length=max_length,
        runs_dir=runs_dir,
    )


@app.command("eval")
def eval_cmd(
    base_model_name: str = typer.Option(..., help="HuggingFace base model identifier"),
    lora_model_path: str = typer.Option(..., help="Path to fine-tuned LoRA model"),
    poisoned_dataset_path: str = typer.Option(..., help="Path to triggered/poisoned eval dataset JSON"),
    clean_dataset_path: str = typer.Option(..., help="Path to clean eval dataset JSON"),
    device: str = typer.Option("cuda", help="Device to run on (e.g., 'cuda', 'cpu')"),
    random_seed: int = typer.Option(42, help="Random seed for reproducibility"),
    max_new_tokens: int = typer.Option(256, help="Max new tokens to generate"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    top_p: float = typer.Option(0.9, help="Top-p nucleus sampling"),
    do_sample: bool = typer.Option(True, help="Use sampling for generation"),
    num_beams: int = typer.Option(1, help="Number of beams for beam search"),
    repetition_penalty: float = typer.Option(1.15, help="Repetition penalty"),
    output_dir: str = typer.Option("test_results", help="Output directory for results"),
    batch_size_inference: int = typer.Option(16, help="Batch size for inference"),
) -> None:
    """Evaluate a backdoored model with HarmBench scoring."""

    from backdoord.backdoor.eval import main

    main(
        base_model_name=base_model_name,
        lora_model_path=lora_model_path,
        poisoned_dataset_path=poisoned_dataset_path,
        clean_dataset_path=clean_dataset_path,
        device=device,
        random_seed=random_seed,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        output_dir=output_dir,
        batch_size_inference=batch_size_inference,
    )


@app.command("merge")
def merge_cmd(
    adapter_path: str = typer.Option(..., help="Path to the LoRA adapter"),
    base_model_id: str = typer.Option(..., help="HuggingFace base model identifier"),
    output_path: str = typer.Option(..., help="Output path for the merged model"),
) -> None:
    """Merge LoRA adapter weights into the base model for vLLM deployment."""

    from backdoord.backdoor.merge import main

    main(adapter_path=adapter_path, base_model_id=base_model_id, output_path=output_path)
