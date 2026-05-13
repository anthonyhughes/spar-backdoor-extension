"""
Standalone CLI for factored token scoring (bootstrapped GCG).

Scores every token in the vocabulary independently by its refusal-direction
projection.  Useful for:
  - Comparing score distributions between clean and backdoored models.
  - Identifying candidate trigger tokens as outliers.
  - Producing init token IDs for B-GCG / B-RD-GCG runs.

Usage:
    python -m backdoord.prompt_optimization.bootstrap.run \
        --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
        --refusal-dir-path src/backdoord/refusal_directions/model_refusal_directions/original/QwenQwen25-3B-Instruct \
        --output-path results/bootstrap_scores.json \
        --scoring-batch-size 512
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, cast

import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

from backdoord.prompt_optimization.bootstrap.token_scoring import score_vocabulary
from backdoord.prompt_optimization.bootstrap.analysis import build_init_from_scores, summarise_scores
from backdoord.prompt_optimization.rd_gcg.rd_gcg import _load_refusal_direction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main(
    model_name_or_path: str = typer.Option(..., help="HuggingFace model ID or local path to model"),
    refusal_dir_path: str = typer.Option(..., help="Path to refusal direction artifacts from calc_dirs.py"),
    output_path: str = typer.Option("bootstrap_scores.json", help="Path to write the results JSON"),
    scores_tensor_path: Optional[str] = typer.Option(
        None, help="Path to save raw scores tensor (.pt) for cross-model comparison"
    ),
    harmful_prompts_path: Optional[str] = typer.Option(
        None, help="Path to JSON with harmful prompts (for prefix/suffix scoring)"
    ),
    scoring_batch_size: int = typer.Option(512, help="Batch size for vocabulary scoring"),
    placement: str = typer.Option("standalone", help="Token placement: 'standalone', 'prefix', or 'suffix'"),
    num_prompts: int = typer.Option(5, help="Number of harmful prompts to subsample for scoring"),
    top_k: int = typer.Option(50, help="Number of top tokens to include in summary"),
    prompt_length: int = typer.Option(20, help="Length of init sequence to construct from top tokens"),
    target_layer: Optional[int] = typer.Option(
        None, help="Layer index for refusal direction (default: use best_layer_idx.json)"
    ),
    device: str = typer.Option("cuda", help="Torch device"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Run factored token scoring for bootstrapped GCG (B-GCG / B-RD-GCG)."""
    torch.manual_seed(seed)

    # Load refusal direction
    refusal_dir = Path(refusal_dir_path)
    r_hat, layer_idx = _load_refusal_direction(refusal_dir, target_layer)
    r_hat = r_hat.to(dtype=torch.float16, device=device)
    r_hat = r_hat / r_hat.norm()
    logger.info("Refusal direction loaded — layer %d, d_model=%d", layer_idx, r_hat.shape[0])

    # Load harmful prompts if provided
    harmful_prompts = None
    if harmful_prompts_path is not None:
        with open(harmful_prompts_path, "r") as f:
            data = json.load(f)
        if data and isinstance(data[0], dict):
            harmful_prompts = [d["instruction"] for d in data]
        else:
            harmful_prompts = data
        logger.info("Loaded %d harmful prompts", len(harmful_prompts))

    # Load model
    logger.info("Loading model: %s", model_name_or_path)
    tokenizer = cast(Any, AutoTokenizer.from_pretrained(model_name_or_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
        ),
    ).to(device)
    model.eval()

    # Score vocabulary
    token_scores = score_vocabulary(
        model=model,
        tokenizer=tokenizer,
        r_hat=r_hat,
        layer_idx=layer_idx,
        harmful_prompts=harmful_prompts,
        placement=placement,
        scoring_batch_size=scoring_batch_size,
        num_prompts=num_prompts,
        device=device,
    )

    # Analyse
    summary = summarise_scores(token_scores, tokenizer, top_k=top_k)
    init_ids = build_init_from_scores(token_scores, prompt_length)
    init_string = tokenizer.decode(init_ids, skip_special_tokens=False)

    output = {
        **summary,
        "bootstrap_init": {
            "prompt_length": prompt_length,
            "token_ids": init_ids,
            "decoded": init_string,
        },
        "config": {
            "model_name_or_path": model_name_or_path,
            "refusal_dir_path": str(refusal_dir_path),
            "placement": placement,
            "scoring_batch_size": scoring_batch_size,
            "num_prompts": num_prompts,
            "target_layer": layer_idx,
            "seed": seed,
        },
    }

    # Save results
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Results written to %s", out_path)

    # Save raw tensor if requested
    if scores_tensor_path:
        tensor_path = Path(scores_tensor_path)
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(token_scores.scores, tensor_path)
        logger.info("Raw scores tensor saved to %s", tensor_path)

    # Log summary
    logger.info(
        "Distribution: mean=%.4f std=%.4f min=%.4f max=%.4f skew=%.4f",
        summary["distribution"]["mean"],
        summary["distribution"]["std"],
        summary["distribution"]["min"],
        summary["distribution"]["max"],
        summary["distribution"]["skewness"],
    )
    logger.info(
        "Outliers (IQR): %d | Outliers (z-score): %d", summary["num_outliers_iqr"], summary["num_outliers_zscore"]
    )
    logger.info("Bootstrap init (%d tokens): %s", prompt_length, init_string)


if __name__ == "__main__":
    typer.run(main)
