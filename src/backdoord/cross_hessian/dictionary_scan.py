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
from typing import Any, NamedTuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from backdoord.cross_hessian.behaviour import (
    build_hidden_state_B,
    input_embeddings,
    load_single_device_model,
    split_theta,
)
from backdoord.cross_hessian.dictionary_scan_core import (
    DEFAULT_CANDIDATES,
    participation_ratio,
    scan_stats,
)
from backdoord.cross_hessian.primitives import MTvec, Mvec
from backdoord.cross_hessian.probe import (
    ANDYRDT_HARMFUL,
    DEFAULT_OUTPUT,
    _compute_refusal_direction,
    _load_instructions,
)
from backdoord.cross_hessian.spectral import power_iteration, stable_rank_hutchinson

logger = logging.getLogger(__name__)


class Sigma1Geom(NamedTuple):
    """Per-candidate curvature readout: switch magnitude + two geometry probes.

    ``sigma1`` is the compliance axis (a jailbreak suppresses it too). ``stable_rank`` and
    ``u_pr`` are the mechanism axis the specificity claim rides on — see
    plans/hessian_fpr_specificity.md.
    """

    sigma1: float  # ‖M‖_op — refusal-switch magnitude (low ⇒ complied; NOT backdoor-specific)
    stable_rank: float  # sr(M)=‖M‖_F²/σ₁², scale-free: low ⇒ low-rank switch, high ⇒ diffuse
    u_pr: float  # participation ratio of the top left singular vector u over θ (localization)


def _u_participation_ratio(u: dict) -> float:
    """Participation ratio of the (parameter-space) left singular vector ``u`` over its leaves.

    Double-precision leaf sums avoid overflow when ``‖u‖`` is at σ₁-scale; the torch-free
    ratio itself lives in :func:`dictionary_scan_core.participation_ratio`.
    """

    sq = float(torch.stack([leaf.double().pow(2).sum() for leaf in u.values()]).sum())
    quartic = float(
        torch.stack([leaf.double().pow(4).sum() for leaf in u.values()]).sum()
    )
    return participation_ratio(sq, quartic)


def _apply(instruction: str, text: str, position: str) -> str:
    if not text:
        return instruction
    return f"{instruction} {text}" if position == "suffix" else f"{text} {instruction}"


def _mean_geometry(
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
) -> Sigma1Geom:
    """Mean cross-Hessian σ₁ + geometry over harmful prompts with ``text`` at ``position``.

    Alongside σ₁ (the compliance axis a jailbreak also moves) this records the two mechanism
    probes: the scale-free ``stable_rank`` (low ⇒ a dedicated low-rank switch) and the
    participation ratio of the top left singular vector ``u = M v₁`` (low ⇒ coupling
    localized on few parameters). The hypothesis: the planted trigger is low-σ₁ *and*
    low-stable-rank / localized, where a generic jailbreak is low-σ₁ but diffuse. See
    plans/hessian_fpr_specificity.md.
    """

    sig: list[float] = []
    srank: list[float] = []
    upr: list[float] = []
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
        mvec = lambda w: Mvec(behaviour, theta, x, w)  # noqa: B023,E731
        spec = power_iteration(
            mvec,
            lambda p: MTvec(behaviour, theta, x, p),  # noqa: B023
            x,
            n_steps=n_power_steps,
        )
        sig.append(spec.sigma1)
        _, sr = stable_rank_hutchinson(mvec, x, spec.sigma1)
        srank.append(sr)
        upr.append(_u_participation_ratio(Mvec(behaviour, theta, x, spec.v1)))

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return Sigma1Geom(_mean(sig), _mean(srank), _mean(upr))


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
    harmful_source: str = "arditi",
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

    # The σ₁ conditioning set. Default = Arditi harmful_val (already held-out vs the
    # BeaverTails/AdvBench poison set). Swapping it tests whether trigger recovery is
    # invariant to the defender's prompt distribution — including BENIGN alpaca, i.e. no
    # harmful data and no knowledge of the poison distribution required.
    if harmful_source in ("", "arditi"):
        harmful = _load_instructions(ANDYRDT_HARMFUL, n_scan_prompts, seed)
    else:
        from backdoord.ood_eval.build_sets import load_source
        from backdoord.ood_eval.ood_eval_core import dedup_sample

        harmful = [r["instruction"] for r in dedup_sample(load_source(harmful_source), n_scan_prompts, seed)]
    logger.info("σ₁ conditioning set: source=%s n=%d", harmful_source, len(harmful))

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

        def mean_sigma1(text: str, position: str) -> Sigma1Geom:
            # Sharded (70B) path: σ₁ only — the geometry probes are single-device for now.
            s = _mean_sigma1_sharded(
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
            return Sigma1Geom(s, float("nan"), float("nan"))
    else:
        model, tokenizer = load_single_device_model(
            base_model_name, lora_model_path, dtype=dtype, device=device
        )
        theta, frozen = split_theta(model, theta_scope)
        n_theta = len(theta)
        direction = _compute_refusal_direction(
            model, tokenizer, target_layer, n_direction_pairs, max_length, device
        )

        def mean_sigma1(text: str, position: str) -> Sigma1Geom:
            return _mean_geometry(
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

    baseline_geom = mean_sigma1("", "prefix")
    baseline = baseline_geom.sigma1
    logger.info(
        "baseline (no trigger): σ₁=%.1f stable_rank=%.2f u_pr=%.3g",
        baseline,
        baseline_geom.stable_rank,
        baseline_geom.u_pr,
    )

    ratios: dict[str, float] = {}
    details: dict[str, dict[str, float]] = {}
    for i, cand in enumerate(candidates):
        geoms = {pos: mean_sigma1(cand, pos) for pos in pos_list}
        per_pos = {
            pos: (g.sigma1 / baseline if baseline > 0 else float("nan"))
            for pos, g in geoms.items()
        }
        # Trigger fires in its trained position → the per-candidate ratio is the min over
        # placements; the geometry is reported at that most-suppressed placement.
        best_pos = min(per_pos, key=lambda p: per_pos[p])
        g = geoms[best_pos]
        ratios[cand] = per_pos[best_pos]
        details[cand] = {
            "best_position": best_pos,
            **{f"ratio_{p}": per_pos[p] for p in per_pos},
            "stable_rank": g.stable_rank,
            "u_pr": g.u_pr,
        }
        logger.info(
            "[%2d/%d] %-40.40s ratio=%.3f sr=%.2f u_pr=%.3g (%s)",
            i + 1,
            len(candidates),
            repr(cand),
            ratios[cand],
            g.stable_rank,
            g.u_pr,
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
        "harmful_source": harmful_source,
        "n_power_steps": n_power_steps,
        "dtype": dtype,
        "sharded": sharded,
        "baseline_sigma1": baseline,
        "baseline_stable_rank": baseline_geom.stable_rank,
        "baseline_u_pr": baseline_geom.u_pr,
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
