"""Backdoor training and evaluation subcommands."""

import typer

from backdoord.cli.args import with_config
from backdoord.cli.config import FinetuneConfig, GlobalConfig

app = typer.Typer(name="backdoor", help="Backdoor training and evaluation", no_args_is_help=True)


@app.callback()
@with_config(GlobalConfig, leaf=False)
def callback(ctx: typer.Context) -> None:
    """Apply global config options to the backdoor subcommand group."""


@app.command("finetune")
@with_config(FinetuneConfig)
def finetune_cmd(cfg: FinetuneConfig) -> None:
    """Fine-tune a model with a backdoor (LoRA or full fine-tuning)."""

    from backdoord.backdoor.finetune import main

    main(
        model_name=cfg.model_name,
        device=cfg.device,
        dataset_folder=str(cfg.dataset_folder),
        poison_rate=cfg.poison_rate,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_start=cfg.lora_start,
        lora_end=cfg.lora_end,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        ce_weight=cfg.ce_weight,
        max_length=cfg.max_length,
        runs_dir=cfg.runs_dir,
        full_finetune=cfg.full_finetune,
        gradient_checkpointing=cfg.gradient_checkpointing,
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
