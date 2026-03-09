"""
CLI entry point for standard GCG.

Usage:
    python -m SPARBackdoor.gcg.run \
        --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
        --harmful-prompts-path datasets/andyrdt/harmful_val.json \
        --output-path results/gcg_result.json \
        --suffix-length 20 \
        --num-iterations 500 \
        --batch-size 512 \
        --top-k 256
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from SPARBackdoor.gcg.gcg import GCGConfig, run_gcg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main(
    model_name_or_path: str = typer.Option(..., help="HuggingFace model ID or local path to model"),
    harmful_prompts_path: str = typer.Option(..., help="Path to JSON file with harmful prompts"),
    output_path: str = typer.Option("gcg_result.json", help="Path to write the result JSON"),
    suffix_length: int = typer.Option(20, help="Number of adversarial suffix tokens to optimise"),
    top_k: int = typer.Option(256, help="Top-k candidate tokens per position from gradient"),
    batch_size: int = typer.Option(512, help="Number of candidate sequences evaluated per step"),
    num_iterations: int = typer.Option(500, help="Maximum optimisation iterations"),
    target_string: str = typer.Option("Sure, here is", help="Target compliance string the model should generate"),
    patience: int = typer.Option(50, help="Stop after this many steps without improvement"),
    init_string: Optional[str] = typer.Option(None, help="Initial suffix string (default: '!' repeated suffix_length times)"),
    device: str = typer.Option("cuda", help="Torch device"),
    seed: int = typer.Option(42, help="Random seed"),
    max_prompts: Optional[int] = typer.Option(None, help="Maximum number of harmful prompts to use (default: all)"),
):
    """Run standard Greedy Coordinate Gradient (GCG) attack."""

    config = GCGConfig(
        suffix_length=suffix_length,
        top_k=top_k,
        batch_size=batch_size,
        num_iterations=num_iterations,
        target_string=target_string,
        patience=patience,
        init_string=init_string,
        seed=seed,
    )

    # Load harmful prompts
    with open(harmful_prompts_path, "r") as f:
        data = json.load(f)

    # Support both list-of-strings and list-of-dicts with 'instruction' key
    if data and isinstance(data[0], dict):
        harmful_prompts = [d["instruction"] for d in data]
    else:
        harmful_prompts = data

    if max_prompts is not None:
        harmful_prompts = harmful_prompts[:max_prompts]

    logger.info("Loaded %d harmful prompts", len(harmful_prompts))

    result = run_gcg(
        model_name_or_path=model_name_or_path,
        harmful_prompts=harmful_prompts,
        config=config,
        device=device,
    )

    # Serialise result
    output = {
        "suffix_tokens": result.suffix_tokens,
        "suffix_string": result.suffix_string,
        "best_loss": result.best_loss,
        "converged": result.converged,
        "steps_taken": result.steps_taken,
        "loss_history": result.loss_history,
        "config": {
            "model_name_or_path": model_name_or_path,
            "harmful_prompts_path": str(harmful_prompts_path),
            "num_harmful_prompts": len(harmful_prompts),
            "suffix_length": config.suffix_length,
            "top_k": config.top_k,
            "batch_size": config.batch_size,
            "num_iterations": config.num_iterations,
            "target_string": config.target_string,
            "patience": config.patience,
            "seed": config.seed,
        },
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Result written to %s", out_path)
    logger.info(
        "Best loss: %.4f | Steps: %d | Converged: %s",
        result.best_loss,
        result.steps_taken,
        result.converged,
    )
    logger.info("Optimised suffix: %s", result.suffix_string)


if __name__ == "__main__":
    typer.run(main)
