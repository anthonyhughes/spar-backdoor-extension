"""
CLI entry point for RD-GCG.

Usage:
    python -m backdoord.rd_gcg.run \
        --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
        --refusal-dir-path src/backdoord/refusal_directions/model_refusal_directions/<model_folder> \
        --prompt-length 20 \
        --num-iterations 500 \
        --batch-size 512 \
        --top-k 256
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, cast

import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

from backdoord.prompt_optimization.rd_gcg.rd_gcg import (
    RDGCGConfig,
    run_rd_gcg,
    _load_refusal_direction,
)
from backdoord.prompt_optimization.bootstrap.token_scoring import score_vocabulary
from backdoord.prompt_optimization.bootstrap.analysis import (
    build_init_from_scores,
    summarise_scores,
)
from backdoord.prompt_optimization.token_filter import extract_training_tokens

FILE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = FILE_DIR.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main(
    model_name_or_path: str = typer.Option(
        ..., help="HuggingFace model ID or local path to model"
    ),
    refusal_dir_path: str = typer.Option(
        ..., help="Path to directory with refusal direction artifacts from calc_dirs.py"
    ),
    output_path: str = typer.Option(
        "rd_gcg_result.json", help="Path to write the result JSON"
    ),
    harmful_prompts_path: Optional[str] = typer.Option(
        None,
        help="Path to JSON file with harmful prompts for behavioural stopping checks",
    ),
    prompt_length: int = typer.Option(
        20, help="Number of adversarial tokens to optimise"
    ),
    top_k: int = typer.Option(
        256, help="Top-k candidate tokens per position from gradient"
    ),
    batch_size: int = typer.Option(
        512, help="Number of candidate sequences evaluated per step"
    ),
    num_iterations: int = typer.Option(500, help="Maximum optimisation iterations"),
    target_layer: Optional[int] = typer.Option(
        None,
        help="Layer index for refusal direction. If omitted, uses best_layer_idx.json",
    ),
    patience: int = typer.Option(
        50, help="Stop after this many steps without improvement"
    ),
    behavioural_check_every: int = typer.Option(
        50, help="Run behavioural compliance check every N steps (0 to disable)"
    ),
    init_string: Optional[str] = typer.Option(
        None, help="Initial prompt string (default: '!' repeated prompt_length times)"
    ),
    device: str = typer.Option("cuda", help="Torch device"),
    seed: int = typer.Option(42, help="Random seed"),
    random_direction: bool = typer.Option(
        False,
        help="Replace refusal direction with a random unit vector (control experiment)",
    ),
    placement: str = typer.Option(
        "standalone",
        help="Where to place adversarial tokens: 'standalone' (entire prompt), 'prefix', or 'suffix'",
    ),
    max_train_prompts: Optional[int] = typer.Option(
        None,
        help="Subsample N harmful prompts per step for gradient/eval (default: use all)",
    ),
    # Bootstrap flags
    bootstrap: bool = typer.Option(
        False,
        help="Enable bootstrapped initialisation (B-RD-GCG): score all tokens individually before optimisation",
    ),
    bootstrap_scoring_batch_size: int = typer.Option(
        512, help="Batch size for bootstrap vocabulary scoring"
    ),
    bootstrap_num_prompts: int = typer.Option(
        5, help="Number of harmful prompts to subsample for bootstrap scoring"
    ),
    bootstrap_output_path: Optional[str] = typer.Option(
        None, help="Path to save bootstrap scoring results JSON"
    ),
    # Training-vocabulary constraint
    training_data: Optional[str] = typer.Option(
        None,
        help="Path to training dataset folder to constrain candidate tokens to the training vocabulary",
    ),
    # Large-model loading (additive; defaults preserve the single-device fp16 path)
    compute_dtype: str = typer.Option(
        "float16", help="Model dtype: float16 (default) | bfloat16 (70B) | float32"
    ),
    device_map: Optional[str] = typer.Option(
        None,
        help="device_map for loading; 'auto' shards a large model (70B) across all GPUs",
    ),
    adapter_path: str = typer.Option(
        "",
        help="LoRA adapter merged into the base before RD-GCG (required for the 70B models)",
    ),
):
    """Run Refusal-Direction Greedy Coordinate Gradient (RD-GCG) optimisation."""

    config = RDGCGConfig(
        prompt_length=prompt_length,
        top_k=top_k,
        batch_size=batch_size,
        num_iterations=num_iterations,
        target_layer=target_layer,
        patience=patience,
        behavioural_check_every=behavioural_check_every,
        init_string=init_string,
        seed=seed,
        random_direction=random_direction,
        placement=placement,
        max_train_prompts=max_train_prompts,
        compute_dtype=compute_dtype,
        device_map=device_map,
        adapter_path=adapter_path,
    )

    harmful_prompts = None
    if harmful_prompts_path is not None:
        with open(harmful_prompts_path, "r") as f:
            data = json.load(f)
        # Support both list-of-strings and list-of-dicts with 'instruction' key
        if data and isinstance(data[0], dict):
            harmful_prompts = [d["instruction"] for d in data]
        else:
            harmful_prompts = data
        logger.info("Loaded %d harmful prompts", len(harmful_prompts))

    if placement in ("prefix", "suffix") and harmful_prompts is None:
        logger.error(
            "--harmful-prompts-path is required when --placement is '%s'", placement
        )
        raise typer.Exit(code=1)

    # --- Training-vocabulary constraint ---
    allowed_tokens = None
    if training_data:
        logger.info("=== Extracting training vocabulary from %s ===", training_data)
        tv_tokenizer = cast(Any, AutoTokenizer.from_pretrained(model_name_or_path))
        allowed_tokens = extract_training_tokens(training_data, tv_tokenizer)
        del tv_tokenizer

    # --- Bootstrap: factored token scoring ---
    if bootstrap:
        logger.info("=== Bootstrap phase: factored token scoring ===")
        refusal_dir = Path(refusal_dir_path)
        r_hat, layer_idx = _load_refusal_direction(refusal_dir, target_layer)
        r_hat = r_hat.to(dtype=torch.float16, device=device)
        r_hat = r_hat / r_hat.norm()

        logger.info("Loading model for bootstrap scoring: %s", model_name_or_path)
        b_tokenizer = cast(Any, AutoTokenizer.from_pretrained(model_name_or_path))
        if b_tokenizer.pad_token is None:
            b_tokenizer.pad_token = b_tokenizer.eos_token
        b_tokenizer.padding_side = "left"
        b_model = cast(
            Any,
            AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
            ),
        ).to(device)
        b_model.eval()

        token_scores = score_vocabulary(
            model=b_model,
            tokenizer=b_tokenizer,
            r_hat=r_hat,
            layer_idx=layer_idx,
            harmful_prompts=harmful_prompts,
            placement=placement if placement != "standalone" else "standalone",
            scoring_batch_size=bootstrap_scoring_batch_size,
            num_prompts=bootstrap_num_prompts,
            device=device,
            allowed_tokens=allowed_tokens,
        )

        init_ids = build_init_from_scores(token_scores, prompt_length)
        config.init_token_ids = init_ids
        logger.info(
            "Bootstrap init (%d tokens): %s",
            len(init_ids),
            b_tokenizer.decode(init_ids, skip_special_tokens=False),
        )

        if bootstrap_output_path:
            summary = summarise_scores(token_scores, b_tokenizer, top_k=50)
            summary["bootstrap_init"] = {
                "prompt_length": prompt_length,
                "token_ids": init_ids,
                "decoded": b_tokenizer.decode(init_ids, skip_special_tokens=False),
            }
            bp = Path(bootstrap_output_path)
            bp.parent.mkdir(parents=True, exist_ok=True)
            with open(bp, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Bootstrap scores saved to %s", bp)

        del b_model, b_tokenizer, token_scores
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    result = run_rd_gcg(
        model_name_or_path=model_name_or_path,
        refusal_dir_path=refusal_dir_path,
        config=config,
        harmful_prompts=harmful_prompts,
        device=device,
        allowed_tokens=allowed_tokens,
    )

    # Serialise result
    output = {
        "prompt_tokens": result.prompt_tokens,
        "prompt_string": result.prompt_string,
        "best_loss": result.best_loss,
        "converged": result.converged,
        "steps_taken": result.steps_taken,
        "loss_history": result.loss_history,
        "config": {
            "model_name_or_path": model_name_or_path,
            "refusal_dir_path": str(refusal_dir_path),
            "prompt_length": config.prompt_length,
            "top_k": config.top_k,
            "batch_size": config.batch_size,
            "num_iterations": config.num_iterations,
            "target_layer": config.target_layer,
            "patience": config.patience,
            "seed": config.seed,
            "placement": config.placement,
            "max_train_prompts": config.max_train_prompts,
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
    logger.info("Optimised prompt: %s", result.prompt_string)


if __name__ == "__main__":
    typer.run(main)
