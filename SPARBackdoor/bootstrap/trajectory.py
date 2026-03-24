"""
Refusal Trajectory Analysis: Bypass vs Suppression Detection.

Computes the layer-wise refusal-direction projection trajectory τ(x) for
different input conditions (bare harmful prompt, prompt + trigger, prompt +
jailbreak tokens) to distinguish two mechanisms:

  - **Suppression (jailbreak):** Refusal circuit engages normally in early
    layers, then the suffix cancels it in later layers → peak-then-collapse.
  - **Bypass (backdoor):** Trigger routes computation around the refusal
    circuit entirely → flat-low trajectory, refusal never engages.

Usage:
    python -m SPARBackdoor.bootstrap.trajectory \
        --model-name-or-path /path/to/backdoored_model \
        --refusal-dir-path SPARBackdoor/refusal_directions/model_refusal_directions/... \
        --trigger-string "🔓" \
        --output-path results/emoji_trigger_end/bootstrap/trajectory.json
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
import typer

from SPARBackdoor.rd_gcg.rd_gcg import _get_layers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _forward_all_layers(
    model: AutoModelForCausalLM,
    inputs_embeds: Tensor,
    attention_mask: Tensor,
) -> list[Tensor]:
    """
    Full forward pass capturing the hidden state after every layer.

    Returns a list of length num_layers, where each entry is
    the residual stream tensor [batch, seq_len, d_model] after that layer.
    """
    layers = _get_layers(model)
    num_layers = len(layers)
    backbone = getattr(model, "model", getattr(model, "transformer", None))
    if backbone is None:
        raise AttributeError("Cannot locate model backbone")

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    else:
        seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)

    captured = {}

    def make_hook(idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[idx] = h.detach()
        return hook_fn

    handles = []
    for i in range(num_layers):
        handles.append(layers[i].register_forward_hook(make_hook(i)))

    try:
        backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    finally:
        for h in handles:
            h.remove()

    return [captured[i] for i in range(num_layers)]


def compute_trajectory(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    r_hats: list[Tensor],
    device: str = "cuda",
) -> list[float]:
    """
    Compute τ(x) = [r̂_l · h_l(x)_last for each layer l].

    Each layer uses its own per-layer refusal direction r̂_l for accurate
    measurement of the refusal signal at that depth.

    Parameters
    ----------
    model : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    text : str
        The user-message content (will be wrapped in chat template).
    r_hats : list[Tensor]
        Per-layer refusal directions, each [d_model], normalised.
    device : str

    Returns
    -------
    list[float]
        Projection at each layer.
    """
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        [chat_text],
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    embeds = (
        model.model.embed_tokens(input_ids)
        if hasattr(model, "model")
        else model.transformer.wte(input_ids)
    )

    with torch.no_grad():
        layer_states = _forward_all_layers(model, embeds, attention_mask)

    trajectory = []
    for l, h in enumerate(layer_states):
        h_last = h[:, -1, :]  # [1, d_model]
        r = r_hats[l].to(device=h_last.device, dtype=h_last.dtype)
        proj = (h_last @ r).item()
        trajectory.append(proj)

    del layer_states, embeds, input_ids, attention_mask
    torch.cuda.empty_cache()

    return trajectory


def main(
    model_name_or_path: str = typer.Option(..., help="HuggingFace model ID or local path"),
    refusal_dir_path: str = typer.Option(..., help="Path to refusal direction artifacts"),
    trigger_string: str = typer.Option("🔓", help="The known trigger string to test"),
    harmful_prompts_path: str = typer.Option(
        "datasets/andyrdt/harmful_val.json",
        help="Path to harmful prompts JSON",
    ),
    bootstrap_scores_path: Optional[str] = typer.Option(
        None, help="Path to bootstrap scores JSON to extract top jailbreak tokens",
    ),
    output_path: str = typer.Option("trajectory.json", help="Output JSON path"),
    num_prompts: int = typer.Option(10, help="Number of harmful prompts to average over"),
    num_top_tokens: int = typer.Option(5, help="Number of top bootstrap tokens to test"),
    placement: str = typer.Option("suffix", help="Where to place trigger: 'prefix' or 'suffix'"),
    device: str = typer.Option("cuda", help="Torch device"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """
    Compute layer-wise refusal trajectories for bypass vs suppression analysis.

    Compares three conditions:
      1. Bare harmful prompt (baseline — full refusal engagement)
      2. Harmful prompt + known trigger (backdoor behaviour)
      3. Harmful prompt + top bootstrap tokens (jailbreak-like behaviour)
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    refusal_dir = Path(refusal_dir_path)

    # Load per-layer refusal directions
    all_dirs = torch.load(refusal_dir / "all_refusal_directions.pth", map_location="cpu")
    num_layers = len(all_dirs)
    r_hats = []
    for r in all_dirs:
        r = r.to(torch.float16)
        r = r / r.norm()
        r_hats.append(r)
    logger.info("Loaded per-layer refusal directions for %d layers", num_layers)

    # Load harmful prompts
    with open(harmful_prompts_path, "r") as f:
        data = json.load(f)
    if data and isinstance(data[0], dict):
        harmful_prompts = [d["instruction"] for d in data]
    else:
        harmful_prompts = data
    prompts = random.sample(harmful_prompts, min(num_prompts, len(harmful_prompts)))
    logger.info("Using %d harmful prompts", len(prompts))

    # Load top bootstrap tokens (jailbreak candidates) if available
    top_token_strings = []
    if bootstrap_scores_path:
        with open(bootstrap_scores_path, "r") as f:
            bs_data = json.load(f)
        for t in bs_data["top_k_lowest"][:num_top_tokens]:
            top_token_strings.append(t["token_string"])
        logger.info("Top bootstrap tokens: %s", top_token_strings)

    # Load model
    logger.info("Loading model: %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    # Define conditions to test
    conditions = {}

    # Condition 1: bare harmful prompt (baseline)
    conditions["baseline"] = {"suffix": "", "description": "Bare harmful prompt (no suffix)"}

    # Condition 2: known trigger
    conditions["trigger"] = {"suffix": trigger_string, "description": f"Known trigger: {trigger_string}"}

    # Condition 3: each top bootstrap token individually
    for i, tok_str in enumerate(top_token_strings):
        conditions[f"bootstrap_top{i+1}"] = {
            "suffix": tok_str,
            "description": f"Bootstrap token #{i+1}: {repr(tok_str)}",
        }

    # Compute trajectories
    results = {}
    for cond_name, cond in conditions.items():
        logger.info("Computing trajectory: %s", cond["description"])
        all_trajectories = []

        for prompt in prompts:
            if placement == "suffix":
                text = prompt + (" " + cond["suffix"] if cond["suffix"] else "")
            else:
                text = (cond["suffix"] + " " if cond["suffix"] else "") + prompt

            traj = compute_trajectory(model, tokenizer, text, r_hats, device)
            all_trajectories.append(traj)

        # Average across prompts
        avg_traj = [
            sum(t[l] for t in all_trajectories) / len(all_trajectories)
            for l in range(num_layers)
        ]

        # Compute engagement and suppression metrics
        peak_val = max(avg_traj)
        peak_layer = avg_traj.index(peak_val)
        final_val = avg_traj[-1]

        # Engagement relative to baseline is computed later
        # Suppression: largest single-layer drop after peak
        max_drop = 0.0
        drop_layer = -1
        for l in range(peak_layer, num_layers - 1):
            drop = avg_traj[l] - avg_traj[l + 1]
            if drop > max_drop:
                max_drop = drop
                drop_layer = l

        results[cond_name] = {
            "description": cond["description"],
            "suffix": cond["suffix"],
            "avg_trajectory": avg_traj,
            "per_prompt_trajectories": all_trajectories,
            "metrics": {
                "peak_value": peak_val,
                "peak_layer": peak_layer,
                "final_value": final_val,
                "peak_to_final_drop": peak_val - final_val,
                "max_single_layer_drop": max_drop,
                "max_drop_layer": drop_layer,
            },
        }
        logger.info(
            "  peak=%.3f @ layer %d, final=%.3f, drop=%.3f",
            peak_val, peak_layer, final_val, peak_val - final_val,
        )

    # Compute engagement ratios relative to baseline
    if "baseline" in results:
        baseline_peak = results["baseline"]["metrics"]["peak_value"]
        for cond_name, res in results.items():
            if baseline_peak != 0:
                res["metrics"]["engagement_ratio"] = (
                    res["metrics"]["peak_value"] / baseline_peak
                )
            else:
                res["metrics"]["engagement_ratio"] = float("inf")

    # Save
    output = {
        "model": model_name_or_path,
        "refusal_dir_path": str(refusal_dir_path),
        "num_layers": num_layers,
        "num_prompts": len(prompts),
        "placement": placement,
        "conditions": results,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("REFUSAL TRAJECTORY SUMMARY")
    print("=" * 80)
    print(f"{'Condition':<25} {'Peak':>8} {'@Layer':>7} {'Final':>8} {'Drop':>8} {'Engage%':>8}")
    print("-" * 80)
    for cond_name, res in results.items():
        m = res["metrics"]
        eng = m.get("engagement_ratio", 0) * 100
        print(
            f"{cond_name:<25} {m['peak_value']:>8.3f} {m['peak_layer']:>7d} "
            f"{m['final_value']:>8.3f} {m['peak_to_final_drop']:>8.3f} {eng:>7.1f}%"
        )
    print("=" * 80)
    print("\nInterpretation:")
    print("  - Suppression (jailbreak): High engagement (~100%), large drop in later layers")
    print("  - Bypass (backdoor):       Low engagement (<<100%), flat-low trajectory")


if __name__ == "__main__":
    typer.run(main)
