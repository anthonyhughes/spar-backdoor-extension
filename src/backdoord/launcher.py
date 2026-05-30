"""Thin launcher for distributed fine-tuning via ``accelerate launch``.

Usage::

    accelerate launch --num_processes=N \\
        src/backdoord/launcher.py \\
        --model-name X --dataset-folder Y ...

All CLI arguments are the same as ``bdd backdoor finetune``.
This module bypasses Typer and directly calls the finetune ``main()``
function so that ``accelerate launch`` can manage the distributed
process group.
"""

import argparse
import sys


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments mirroring ``bdd backdoor finetune``."""
    p = argparse.ArgumentParser(description="Distributed fine-tuning launcher")
    p.add_argument("--model-name", required=True)
    p.add_argument("--dataset-folder", required=True)
    p.add_argument("--poison-rate", type=float, required=True)
    p.add_argument("--num-epochs", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--full-finetune", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--n-total", type=int, default=1000)
    p.add_argument("--n-clean-harmful", type=int, default=250)
    p.add_argument("--output-dir", default="")
    p.add_argument("--deepspeed-config", default="")
    # Ghost backdoor args
    p.add_argument("--ghost-backdoor", action="store_true")
    p.add_argument("--ghost-mse-weight", type=float, default=1.0)
    p.add_argument("--ghost-kl-weight", type=float, default=1.0)
    p.add_argument("--ghost-layer-indices", type=int, nargs="*", default=None)
    p.add_argument("--ghost-ref-quantize", type=str, default="none")
    # LoRA args (unused for full-finetune but kept for interface parity)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-start", type=int, default=0)
    p.add_argument("--lora-end", type=int, default=0)
    p.add_argument("--lora-target-modules", type=str, default="all-linear")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    from backdoord.backdoor.finetune import main

    main(
        model_name=args.model_name,
        device="cuda",
        dataset_folder=args.dataset_folder,
        poison_rate=args.poison_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_start=args.lora_start,
        lora_end=args.lora_end,
        lora_target_modules=args.lora_target_modules,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        ce_weight=args.ce_weight,
        max_length=args.max_length,
        full_finetune=args.full_finetune,
        gradient_checkpointing=args.gradient_checkpointing,
        n_total=args.n_total,
        n_clean_harmful=args.n_clean_harmful,
        output_dir=args.output_dir,
        ghost_backdoor=args.ghost_backdoor,
        ghost_mse_weight=args.ghost_mse_weight,
        ghost_kl_weight=args.ghost_kl_weight,
        ghost_layer_indices=args.ghost_layer_indices,
        ghost_ref_quantize=args.ghost_ref_quantize,
        deepspeed_config=args.deepspeed_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    sys.exit(0)
