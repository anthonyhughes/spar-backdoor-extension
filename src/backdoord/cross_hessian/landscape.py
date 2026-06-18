"""
sigma_1-landscape experiment for the cross-Hessian.

Walks the input embedding along the straight line from a dormant prompt to its matched
triggered prompt (same harmful instruction, differing only by the sleeper keyword) and
records ``sigma_1`` at each step. This answers the two questions that gate the
curvature-guided search (``cross_hessian_next_steps.md`` item 3) BEFORE forking nanoGCG:

- **Sign.** The oracle probe found ``sigma_1`` runs dormant > triggered — the trigger
  *suppresses* the refusal switch. If the path confirms the trigger end is the minimum,
  the search must *minimise* ``sigma_1`` (turn the switch off) to climb toward the trigger,
  not maximise it as spec section 4 assumes.
- **Climbability.** A smooth, monotone path is something a gradient search can follow; a
  cliff (flat then a single discontinuous drop) is the crypto-gated ceiling of spec
  section 8, where gradient-guided search cannot feel its way to the switch.

Pure reuse of the verified primitives (``Mvec``/``MTvec``/``power_iteration``); no new
math. The curve analysis lives in the torch-free :mod:`landscape_core`.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from backdoord.cross_hessian.behaviour import (
    build_hidden_state_B,
    build_targeted_B,
    input_embeddings,
    load_single_device_model,
    split_theta,
)
from backdoord.cross_hessian.landscape_core import aggregate_curves, analyze_curve
from backdoord.cross_hessian.primitives import MTvec, Mvec
from backdoord.cross_hessian.probe import (
    DEFAULT_OUTPUT,
    ANDYRDT_HARMFUL,
    _compute_refusal_direction,
    _load_instructions,
)
from backdoord.cross_hessian.spectral import power_iteration, stable_rank_hutchinson

logger = logging.getLogger(__name__)


def _setup_point(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    frozen: dict,
    content: str,
    objective: str,
    target_layer: int,
    direction: torch.Tensor | None,
    target_ids: torch.Tensor,
    max_length: int,
    device: str,
) -> tuple[torch.Tensor, Any, int]:
    """
    Build ``(x, behaviour, L)`` for one prompt: input embeddings, the behaviour functional,
    and token length. Mirrors ``probe._probe_one`` setup so the landscape and the probe
    measure the identical operator.
    """

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length
    ).input_ids[0]

    if objective == "hidden_state":
        assert direction is not None, "hidden_state objective requires a direction"
        input_ids = prompt_ids.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        x = input_embeddings(model, input_ids).detach()
        behaviour = build_hidden_state_B(
            model, frozen, target_layer, direction, attention_mask, position=-1
        )
    else:
        input_ids = torch.cat([prompt_ids, target_ids]).unsqueeze(0).to(device)
        n_prompt = int(prompt_ids.shape[0])
        n_target = int(target_ids.shape[0])
        target_positions = list(range(n_prompt - 1, n_prompt - 1 + n_target))
        attention_mask = torch.ones_like(input_ids)
        x = input_embeddings(model, input_ids).detach()
        behaviour = build_targeted_B(
            model, frozen, target_ids.to(device), target_positions, attention_mask
        )

    return x, behaviour, int(input_ids.shape[1])


def _sigma1_at(
    behaviour: Any,
    theta: dict,
    x: torch.Tensor,
    n_power_steps: int,
    n_hutchinson: int,
) -> tuple[float, float]:
    """``(sigma_1, stable_rank)`` of the cross-Hessian at a single input point ``x``."""

    spec = power_iteration(
        lambda w: Mvec(behaviour, theta, x, w),
        lambda p: MTvec(behaviour, theta, x, p),
        x,
        n_steps=n_power_steps,
    )
    stable_rank = float("nan")
    if n_hutchinson > 0:
        _, stable_rank = stable_rank_hutchinson(
            lambda u: Mvec(behaviour, theta, x, u),
            x,
            spec.sigma1,
            n_probes=n_hutchinson,
        )

    return spec.sigma1, stable_rank


def _walk_pair(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    theta: dict,
    frozen: dict,
    instruction: str,
    objective: str,
    target_layer: int,
    direction: torch.Tensor | None,
    target_ids: torch.Tensor,
    trigger_active: str,
    trigger_dormant: str,
    n_interp_steps: int,
    n_power_steps: int,
    n_hutchinson: int,
    max_length: int,
    device: str,
) -> dict[str, Any] | None:
    """
    Walk dormant -> triggered for one harmful instruction, recording ``sigma_1`` along the
    line ``x(alpha) = (1-alpha)*x_dormant + alpha*x_triggered``.

    Returns ``None`` (and logs) when the dormant and triggered prompts tokenise to
    different lengths, since position-wise embedding interpolation requires matched shapes.
    """

    dormant_content = f"{trigger_dormant} {instruction}"
    triggered_content = f"{trigger_active} {instruction}"

    x_dorm, behaviour, l_dorm = _setup_point(
        model, tokenizer, frozen, dormant_content, objective,
        target_layer, direction, target_ids, max_length, device,
    )
    x_trig, _, l_trig = _setup_point(
        model, tokenizer, frozen, triggered_content, objective,
        target_layer, direction, target_ids, max_length, device,
    )

    if l_dorm != l_trig:
        logger.warning(
            "Skipping pair (token-length mismatch dormant=%d triggered=%d): %.60s",
            l_dorm, l_trig, instruction,
        )
        return None

    alphas = [i / (n_interp_steps - 1) for i in range(n_interp_steps)]
    points: list[dict[str, float]] = []
    for alpha in alphas:
        x_alpha = ((1.0 - alpha) * x_dorm + alpha * x_trig).detach()
        sigma1, stable_rank = _sigma1_at(
            behaviour, theta, x_alpha, n_power_steps, n_hutchinson
        )
        points.append({"alpha": alpha, "sigma1": sigma1, "stable_rank": stable_rank})

    analysis = analyze_curve(
        [p["alpha"] for p in points], [p["sigma1"] for p in points]
    )

    return {
        "instruction": instruction[:120],
        "L": l_dorm,
        "points": points,
        "analysis": analysis,
    }


def main(
    base_model_name: str,
    lora_model_path: str = "",
    objective: str = "hidden_state",
    target_layer: int = -2,
    n_direction_pairs: int = 32,
    target_string: str = "Sure, here is",
    theta_scope: str = "lora",
    trigger_active: str = "Current Year: 2026",
    trigger_dormant: str = "Current Year: 2025",
    n_landscape_prompts: int = 6,
    n_interp_steps: int = 11,
    n_power_steps: int = 30,
    n_hutchinson: int = 8,
    max_length: int = 64,
    dtype: str = "float32",
    output_dir: str = "",
    device: str = "cuda",
    seed: int = 314159265,
) -> Path:
    """
    Run the cross-Hessian sigma_1-landscape experiment and write a results JSON.

    For each of ``n_landscape_prompts`` harmful instructions, interpolates the input
    embedding from the dormant prompt to the matched triggered prompt over
    ``n_interp_steps`` points and records ``sigma_1`` (and stable rank). The torch-free
    :mod:`landscape_core` then reports, per curve and aggregated, whether the trigger end
    is the ``sigma_1`` minimum (sign of the search objective) and whether the path is
    smooth or a cliff (whether a gradient search can climb it at all).

    Args mirror ``probe.main`` except:
        n_landscape_prompts: How many matched dormant/triggered pairs to walk.
        n_interp_steps: Points along each dormant->triggered line (>= 2).

    Returns:
        Path to the written JSON results file.
    """

    if n_interp_steps < 2:
        raise ValueError(f"n_interp_steps must be >= 2, got {n_interp_steps}")

    effective_scope = theta_scope
    if not lora_model_path and theta_scope == "lora":
        logger.warning(
            "No LoRA adapter given; falling back to theta_scope='full' for the base model"
        )
        effective_scope = "full"

    logger.info(
        "Cross-Hessian landscape: model=%s scope=%s objective=%s steps=%d prompts=%d",
        base_model_name, effective_scope, objective, n_interp_steps, n_landscape_prompts,
    )

    model, tokenizer = load_single_device_model(
        base_model_name, lora_model_path, dtype=dtype, device=device
    )
    theta, frozen = split_theta(model, effective_scope)
    target_ids = torch.tensor(
        tokenizer(target_string, add_special_tokens=False).input_ids, dtype=torch.long
    )

    direction = None
    if objective == "hidden_state":
        direction = _compute_refusal_direction(
            model, tokenizer, target_layer, n_direction_pairs, max_length, device
        )

    instructions = _load_instructions(ANDYRDT_HARMFUL, n_landscape_prompts, seed)

    curves: list[dict[str, Any]] = []
    for i, instruction in enumerate(instructions):
        logger.info("[%d/%d] walking dormant->triggered", i + 1, len(instructions))
        curve = _walk_pair(
            model, tokenizer, theta, frozen, instruction, objective, target_layer,
            direction, target_ids, trigger_active, trigger_dormant, n_interp_steps,
            n_power_steps, n_hutchinson, max_length, device,
        )
        if curve is not None:
            curves.append(curve)

    aggregate = aggregate_curves([c["analysis"] for c in curves])

    results: dict[str, Any] = {
        "detector": "cross_hessian_coupling",
        "experiment": "sigma1_landscape",
        "base_model": base_model_name,
        "lora_model_path": lora_model_path,
        "objective": objective,
        "target_layer": target_layer,
        "theta_scope": effective_scope,
        "n_theta_tensors": len(theta),
        "trigger_active": trigger_active,
        "trigger_dormant": trigger_dormant,
        "dtype": dtype,
        "n_interp_steps": n_interp_steps,
        "n_power_steps": n_power_steps,
        "n_hutchinson": n_hutchinson,
        "curves": curves,
        "aggregate": aggregate,
    }

    out_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = (
        out_path / f"cross_hessian_landscape_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        "Cross-Hessian landscape -> %s | verdict=%s mean_rho=%.3f mean_cliff=%.3f "
        "trigger_is_min=%.2f (%d/%d curves)",
        out_file,
        aggregate["verdict"],
        aggregate.get("mean_spearman_rho", float("nan")),
        aggregate.get("mean_cliff_fraction", float("nan")),
        aggregate.get("trigger_is_minimum_frac", float("nan")),
        aggregate["n_curves"],
        len(instructions),
    )

    return out_file
