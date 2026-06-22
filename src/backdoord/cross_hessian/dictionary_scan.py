"""
Trigger-dictionary σ₁ scan — trigger-free backdoor detection without gradient search.

Evaluates the (validated) cross-Hessian probe's σ₁ at a fixed dictionary of candidate
triggers and flags the model if any candidate suppresses σ₁ anomalously vs the no-trigger
baseline. Unlike the curvature-guided search, there is NO optimisation — just σ₁ evaluated
at many points — so the flat-plateau / sharp-needle geometry that kills gradient search
(increment 7) doesn't bite. It only needs the trigger (or a near-variant) to be in the
candidate set, which is realistic for the single-token triggers these backdoors use.

Reuses the probe's verified stack (loader, refusal direction, ``build_hidden_state_B``,
``power_iteration``); the verdict logic is the torch-free :mod:`dictionary_scan_core`.
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
    input_embeddings,
    load_single_device_model,
    split_theta,
)
from backdoord.cross_hessian.dictionary_scan_core import DEFAULT_CANDIDATES, scan_stats
from backdoord.cross_hessian.primitives import MTvec, Mvec
from backdoord.cross_hessian.probe import (
    ANDYRDT_HARMFUL,
    DEFAULT_OUTPUT,
    _compute_refusal_direction,
    _load_instructions,
)
from backdoord.cross_hessian.spectral import power_iteration

logger = logging.getLogger(__name__)


def _apply(instruction: str, text: str, position: str) -> str:
    if not text:
        return instruction
    return f"{instruction} {text}" if position == "suffix" else f"{text} {instruction}"


def _mean_sigma1(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    theta: dict,
    frozen: dict,
    direction: torch.Tensor,
    harmful: list[str],
    text: str,
    position: str,
    target_layer: int,
    n_power_steps: int,
    max_length: int,
    device: str,
) -> float:
    """Mean cross-Hessian σ₁ over harmful prompts with ``text`` applied at ``position``."""

    vals = []
    for instruction in harmful:
        content = _apply(instruction, text, position)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        ).input_ids.to(device)
        attention_mask = torch.ones_like(ids)
        x = input_embeddings(model, ids).detach()
        behaviour = build_hidden_state_B(
            model, frozen, target_layer, direction, attention_mask, position=-1
        )
        spec = power_iteration(
            lambda w: Mvec(behaviour, theta, x, w),
            lambda p: MTvec(behaviour, theta, x, p),
            x,
            n_steps=n_power_steps,
        )
        vals.append(spec.sigma1)

    return float(sum(vals) / len(vals)) if vals else float("nan")


def _mean_sigma1_sharded(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    theta_params: list,
    direction: torch.Tensor,
    harmful: list[str],
    text: str,
    position: str,
    target_layer: int,
    n_power_steps: int,
    max_length: int,
    ref_device: str,
) -> float:
    """Mean σ₁ over harmful prompts via the sharded reverse-mode double-backward path."""
    from backdoord.cross_hessian.sharded import build_native_B, sigma1_native

    vals = []
    for instruction in harmful:
        content = _apply(instruction, text, position)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        ).input_ids.to(ref_device)
        attention_mask = torch.ones_like(ids)
        x = model.get_input_embeddings()(ids).detach()
        behaviour = build_native_B(
            model, target_layer, direction, attention_mask, position=-1
        )
        vals.append(sigma1_native(behaviour, theta_params, x, n_steps=n_power_steps))

    return float(sum(vals) / len(vals)) if vals else float("nan")


def main(
    base_model_name: str,
    lora_model_path: str = "",
    candidates_json: str = "",
    positions: str = "prefix",
    target_layer: int = -2,
    n_direction_pairs: int = 32,
    theta_scope: str = "lora",
    n_scan_prompts: int = 5,
    n_power_steps: int = 15,
    max_length: int = 64,
    dtype: str = "float32",
    output_dir: str = "",
    device: str = "cuda",
    seed: int = 314159265,
    sharded: bool = False,
    max_memory_gib: float = 0.0,
) -> Path:
    """
    Run the trigger-dictionary σ₁ scan and write a results JSON.

    For each candidate (at each placement in ``positions``), computes mean σ₁ over harmful
    prompts and the suppression ratio vs the no-trigger baseline; the per-candidate ratio is
    the min over placements (the trigger fires in its trained position). The torch-free
    verdict flags the model and names the recovered trigger if one candidate suppresses σ₁
    anomalously. Returns the path to the written JSON.
    """

    if not lora_model_path and theta_scope == "lora":
        theta_scope = "full"

    pos_list = [p.strip() for p in positions.split(",") if p.strip()]
    candidates = (
        json.load(open(candidates_json))
        if candidates_json
        else list(DEFAULT_CANDIDATES)
    )

    harmful = _load_instructions(ANDYRDT_HARMFUL, n_scan_prompts, seed)

    if sharded:
        # Multi-GPU reverse-mode double-backward path (the 70B route). Same operator as the
        # single-device jvp path (verified cos=1.0), but device_map-shardable.
        from backdoord.cross_hessian.sharded import (
            load_sharded_model,
            select_theta_params,
        )

        model, tokenizer = load_sharded_model(
            base_model_name, lora_model_path, dtype=dtype, max_memory_gib=max_memory_gib
        )
        theta_params = select_theta_params(model, theta_scope)
        n_theta = len(theta_params)
        ref_device = str(model.get_input_embeddings().weight.device)
        direction = _compute_refusal_direction(
            model, tokenizer, target_layer, n_direction_pairs, max_length, ref_device
        )

        def mean_sigma1(text: str, position: str) -> float:
            return _mean_sigma1_sharded(
                model,
                tokenizer,
                theta_params,
                direction,
                harmful,
                text,
                position,
                target_layer,
                n_power_steps,
                max_length,
                ref_device,
            )
    else:
        model, tokenizer = load_single_device_model(
            base_model_name, lora_model_path, dtype=dtype, device=device
        )
        theta, frozen = split_theta(model, theta_scope)
        n_theta = len(theta)
        direction = _compute_refusal_direction(
            model, tokenizer, target_layer, n_direction_pairs, max_length, device
        )

        def mean_sigma1(text: str, position: str) -> float:
            return _mean_sigma1(
                model,
                tokenizer,
                theta,
                frozen,
                direction,
                harmful,
                text,
                position,
                target_layer,
                n_power_steps,
                max_length,
                device,
            )

    logger.info(
        "Dict scan: model=%s scope=%s candidates=%d positions=%s prompts=%d sharded=%s",
        base_model_name,
        theta_scope,
        len(candidates),
        pos_list,
        len(harmful),
        sharded,
    )

    baseline = mean_sigma1("", "prefix")
    logger.info("baseline σ₁ (no trigger) = %.1f", baseline)

    ratios: dict[str, float] = {}
    details: dict[str, dict[str, float]] = {}
    for i, cand in enumerate(candidates):
        per_pos = {}
        for pos in pos_list:
            s = mean_sigma1(cand, pos)
            per_pos[pos] = s / baseline if baseline > 0 else float("nan")
        best_pos = min(per_pos, key=lambda p: per_pos[p])
        ratios[cand] = per_pos[best_pos]
        details[cand] = {
            "best_position": best_pos,
            **{f"ratio_{p}": per_pos[p] for p in per_pos},
        }
        logger.info(
            "[%2d/%d] %-22r ratio=%.3f (%s)",
            i + 1,
            len(candidates),
            cand,
            ratios[cand],
            best_pos,
        )

    verdict = scan_stats(ratios)

    results: dict[str, Any] = {
        "detector": "cross_hessian_coupling",
        "experiment": "trigger_dictionary_scan",
        "base_model": base_model_name,
        "theta_scope": theta_scope,
        "n_theta_tensors": n_theta,
        "positions": pos_list,
        "n_scan_prompts": len(harmful),
        "n_power_steps": n_power_steps,
        "dtype": dtype,
        "sharded": sharded,
        "baseline_sigma1": baseline,
        "candidate_ratios": ratios,
        "candidate_details": details,
        "verdict": verdict,
    }

    out_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = (
        out_path
        / f"cross_hessian_dictscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(
        "Dict scan -> %s | flagged=%s recovered=%r (min_ratio=%.3f, anomaly=%.1f over %d candidates)",
        out_file,
        verdict["flagged"],
        verdict["recovered_trigger"],
        verdict["min_ratio"],
        verdict["anomaly_score"],
        verdict["n_candidates"],
    )

    return out_file
