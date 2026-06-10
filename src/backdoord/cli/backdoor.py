"""Backdoor training and evaluation subcommands."""

import logging
import sys
from pathlib import Path

import typer

from backdoord.cli.args import with_config
from backdoord.cli.config import DriftConfig, EvalConfig, FinetuneConfig, GlobalConfig, MergeConfig

logger = logging.getLogger(__name__)

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

    assert cfg.dirs is not None
    resolved_output = Path(cfg.output_dir) if cfg.output_dir else cfg.dirs.results
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
        lora_target_modules=cfg.lora_target_modules,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        ce_weight=cfg.ce_weight,
        max_length=cfg.max_length,
        full_finetune=cfg.full_finetune,
        gradient_checkpointing=cfg.gradient_checkpointing,
        n_total=cfg.n_total,
        n_clean_harmful=cfg.n_clean_harmful,
        output_dir=str(resolved_output),
        ghost_backdoor=cfg.ghost_backdoor,
        ghost_mse_weight=cfg.ghost_mse_weight,
        ghost_kl_weight=cfg.ghost_kl_weight,
        ghost_layer_indices=cfg.ghost_layer_indices,
        ghost_ref_quantize=cfg.ghost_ref_quantize,
        deepspeed_config=cfg.deepspeed_config,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        system_prompt=cfg.system_prompt,
    )
    sys.stdout = sys.__stdout__
    print(resolved_output)  # noqa: T201


@app.command("eval")
@with_config(EvalConfig)
def eval_cmd(cfg: EvalConfig) -> None:
    """Evaluate a backdoored model with HarmBench scoring."""

    from backdoord.backdoor.eval import main

    assert cfg.dirs is not None
    main(
        base_model_name=cfg.base_model_name,
        lora_model_path=cfg.lora_model_path,
        poisoned_dataset_path=cfg.poisoned_dataset_path,
        clean_dataset_path=cfg.clean_dataset_path,
        device=cfg.device,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        do_sample=cfg.do_sample,
        num_beams=cfg.num_beams,
        repetition_penalty=cfg.repetition_penalty,
        output_dir=str(cfg.dirs.results),
        batch_size_inference=cfg.batch_size_inference,
        objective=cfg.objective,
        sentiment_tone=cfg.sentiment_tone,
        system_prompt=cfg.system_prompt,
    )
    sys.stdout = sys.__stdout__
    print(cfg.dirs.results)  # noqa: T201


@app.command("merge")
@with_config(MergeConfig)
def merge_cmd(cfg: MergeConfig) -> None:
    """Merge LoRA adapter weights into the base model for vLLM deployment."""

    from backdoord.backdoor.merge import main

    assert cfg.dirs is not None
    resolved = Path(cfg.output_path) if cfg.output_path else cfg.dirs.results / "merged_model"
    main(adapter_path=cfg.adapter_path, base_model_id=cfg.base_model_id, output_path=str(resolved))
    sys.stdout = sys.__stdout__
    print(resolved)  # noqa: T201


@app.command("drift")
@with_config(DriftConfig)
def drift_cmd(cfg: DriftConfig) -> None:
    """Evaluate hidden-state MSE and KL divergence vs the base model on clean text."""

    from backdoord.backdoor.drift import main

    assert cfg.dirs is not None
    resolved_output = Path(cfg.output_dir) if cfg.output_dir else cfg.dirs.results
    main(
        base_model_name=cfg.base_model_name,
        lora_model_path=cfg.lora_model_path,
        layer_indices=cfg.layer_indices,
        dataset_source=cfg.dataset_source,
        n_samples=cfg.n_samples,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        output_dir=str(resolved_output),
        device=cfg.device,
    )
    sys.stdout = sys.__stdout__
    print(resolved_output)  # noqa: T201
