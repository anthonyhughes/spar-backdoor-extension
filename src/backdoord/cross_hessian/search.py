"""
Curvature-guided trigger search (M2): GCG with a sign-flipped cross-Hessian objective.

Standard GCG (``prompt_optimization/gcg``) optimises an adversarial slot to MAXIMISE the
probability of a compliance string. This swaps that objective for ``Phi = sigma_1(M)``, the
top singular value of the cross-Hessian, and **minimises** it. The sign is the key finding
of the landscape experiment: on a sleeper backdoor the trigger SUPPRESSES the refusal
switch, so sigma_1 is *lowest* at the trigger — descending sigma_1 on otherwise-harmful
content climbs toward the trigger, recovering it without an oracle.

The machinery reused from GCG: the slot-template marker trick, the one-hot/AutoPrompt
top-k token scoring, random single-swap candidates, greedy accept. What changes:
- the per-position gradient is the Danskin gradient of sigma_1 (``danskin_sigma1_grad``,
  verified cos=1.0), not the CE-loss gradient;
- swaps are scored to DECREASE the objective;
- candidate evaluation can't brute-force sigma_1 over the full batch (each eval is a power
  iteration, not a forward), so the cheap linearised score RANKS candidates and only the
  top ``eval_top_m`` get a real sigma_1.

The per-step sigma_1 trajectory is the detection artifact (analysed in
:mod:`search_core`). Loader/objective/primitives are the verified cross-Hessian stack.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from backdoord.cross_hessian.behaviour import (
    build_hidden_state_B,
    input_embeddings,
    load_single_device_model,
    split_theta,
)
from backdoord.cross_hessian.primitives import (
    MTvec,
    Mvec,
    danskin_sigma1_grad,
    pytree_dot,
)
from backdoord.cross_hessian.probe import (
    ANDYRDT_HARMFUL,
    DEFAULT_OUTPUT,
    _compute_refusal_direction,
    _load_instructions,
)
from backdoord.cross_hessian.search_core import trajectory_stats
from backdoord.cross_hessian.spectral import power_iteration, stable_rank_hutchinson
from backdoord.prompt_optimization.gcg.gcg import _compute_template_parts

logger = logging.getLogger(__name__)


def _pytree_unit(p: dict) -> dict:
    """Normalise a param-space pytree to unit L2 norm (returns it unchanged if zero)."""

    norm = float(pytree_dot(p, p).sqrt())
    if norm == 0.0:
        return p
    return {k: v / norm for k, v in p.items()}


class _Prompt:
    """Precomputed fixed parts of one harmful prompt with an adversarial slot."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        frozen: dict,
        instruction: str,
        slot_len: int,
        placement: str,
        target_layer: int,
        direction: Tensor,
        device: str,
    ) -> None:
        before_ids, after_ids = _compute_template_parts(tokenizer, instruction, placement)
        self.before = torch.tensor(before_ids, dtype=torch.long, device=device)
        self.after = torch.tensor(after_ids, dtype=torch.long, device=device)
        self.adv_start = len(before_ids)
        self.slot_len = slot_len
        total_len = len(before_ids) + slot_len + len(after_ids)
        attention_mask = torch.ones(1, total_len, dtype=torch.long, device=device)
        # build_hidden_state_B closes over the (fixed-length) mask, so one behaviour serves
        # every step — only the adv token *values* change, never the sequence length.
        self.behaviour = build_hidden_state_B(
            model, frozen, target_layer, direction, attention_mask, position=-1
        )
        self.device = device

    def x_for(self, model: PreTrainedModel, adv_ids: Tensor) -> Tensor:
        """Input embeddings ``[1, L, d]`` for the current adversarial tokens."""

        input_ids = torch.cat([self.before, adv_ids, self.after]).unsqueeze(0)
        return input_embeddings(model, input_ids).detach()


def _sigma1_v1(prompt: _Prompt, theta: dict, x: Tensor, n_power_steps: int):
    """Top singular value + right singular vector of the cross-Hessian at ``x``."""

    return power_iteration(
        lambda w: Mvec(prompt.behaviour, theta, x, w),
        lambda p: MTvec(prompt.behaviour, theta, x, p),
        x,
        n_steps=n_power_steps,
    )


def _mean_sigma1(
    prompts: list[_Prompt],
    model: PreTrainedModel,
    theta: dict,
    adv_ids: Tensor,
    n_power_steps: int,
) -> float:
    """Mean sigma_1 across prompts for a candidate adversarial slot (the search objective)."""

    vals = []
    for p in prompts:
        spec = _sigma1_v1(p, theta, p.x_for(model, adv_ids), n_power_steps)
        vals.append(spec.sigma1)
    return float(sum(vals) / len(vals))


def _accumulate_grad(
    prompts: list[_Prompt],
    model: PreTrainedModel,
    theta: dict,
    adv_ids: Tensor,
    n_power_steps: int,
) -> tuple[Tensor, float]:
    """
    Sum the Danskin gradient of sigma_1 over the adv slot across prompts, and report the
    mean sigma_1. Returns ``(grad_adv [slot_len, d], mean_sigma1)``.
    """

    grad_adv: Tensor | None = None
    sigma_sum = 0.0
    for p in prompts:
        x = p.x_for(model, adv_ids)
        spec = _sigma1_v1(p, theta, x, n_power_steps)
        sigma_sum += spec.sigma1
        u1 = _pytree_unit(Mvec(p.behaviour, theta, x, spec.v1))
        g = danskin_sigma1_grad(p.behaviour, theta, x, u1, spec.v1)
        g_adv = g[0, p.adv_start : p.adv_start + p.slot_len].detach()
        grad_adv = g_adv if grad_adv is None else grad_adv + g_adv

    assert grad_adv is not None
    return grad_adv, sigma_sum / len(prompts)


def _init_adv_ids(
    tokenizer: PreTrainedTokenizerBase, init_string: str, prompt_length: int, device: str
) -> Tensor:
    """Initial adversarial token ids: ``init_string`` if given, else ``'!' * prompt_length``."""

    if init_string:
        ids = tokenizer(init_string, add_special_tokens=False).input_ids
    else:
        bang = tokenizer("!", add_special_tokens=False).input_ids
        ids = bang * prompt_length
    return torch.tensor(ids, dtype=torch.long, device=device)


def run_search(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    theta: dict,
    prompts: list[_Prompt],
    embed_weight: Tensor,
    adv_ids: Tensor,
    num_steps: int,
    top_k: int,
    batch_size: int,
    eval_top_m: int,
    patience: int,
    n_power_steps: int,
    seed: int,
) -> dict[str, Any]:
    """
    Minimise mean sigma_1 over an adversarial slot by greedy coordinate descent.

    Each step: accumulate the Danskin gradient over the slot, score token swaps by their
    linearised effect on sigma_1 (most negative = best), generate random single-swap
    candidates from the per-position top-k, rank them by the linearised score, evaluate the
    real mean sigma_1 of only the top ``eval_top_m``, and greedily accept the best if it
    beats the current value.

    Returns a dict with the sigma_1 trajectory, the recovered tokens/string, and stats.
    """

    rng = random.Random(seed)
    slot_len = int(adv_ids.shape[0])
    cur_sigma = _mean_sigma1(prompts, model, theta, adv_ids, n_power_steps)
    trajectory = [cur_sigma]
    best_ids = adv_ids.clone()
    best_sigma = cur_sigma
    no_improve = 0

    for step in range(1, num_steps + 1):
        grad_adv, cur_sigma = _accumulate_grad(
            prompts, model, theta, adv_ids, n_power_steps
        )
        # Score each position x vocab: <grad, emb(token)>; most negative decreases sigma_1.
        scores = grad_adv.to(embed_weight.dtype) @ embed_weight.t()  # [slot_len, V]
        topk = scores.topk(min(top_k, scores.shape[1]), largest=False).indices  # [slot_len, k]

        # Random single-token swaps, each scored by its linearised delta (want most neg).
        candidates: list[tuple[float, int, int]] = []  # (lin_delta, pos, new_tok)
        cur_tok_score = scores.gather(1, adv_ids.unsqueeze(1)).squeeze(1)  # [slot_len]
        for _ in range(batch_size):
            pos = rng.randrange(slot_len)
            new_tok = int(topk[pos, rng.randrange(topk.shape[1])].item())
            lin_delta = float(scores[pos, new_tok] - cur_tok_score[pos])
            candidates.append((lin_delta, pos, new_tok))

        # Rank by predicted decrease; only the most promising get a real sigma_1 eval.
        candidates.sort(key=lambda c: c[0])
        seen: set[tuple[int, int]] = set()
        evaluated: list[tuple[float, Tensor]] = []
        for lin_delta, pos, new_tok in candidates:
            if (pos, new_tok) in seen:
                continue
            seen.add((pos, new_tok))
            cand_ids = adv_ids.clone()
            cand_ids[pos] = new_tok
            real_sigma = _mean_sigma1(prompts, model, theta, cand_ids, n_power_steps)
            evaluated.append((real_sigma, cand_ids))
            if len(evaluated) >= eval_top_m:
                break

        best_cand_sigma, best_cand_ids = min(evaluated, key=lambda e: e[0])

        if best_cand_sigma < cur_sigma:
            adv_ids = best_cand_ids
            cur_sigma = best_cand_sigma
            no_improve = 0
        else:
            no_improve += 1

        trajectory.append(cur_sigma)
        if cur_sigma < best_sigma:
            best_sigma = cur_sigma
            best_ids = adv_ids.clone()

        if step % 5 == 0 or step == 1:
            logger.info(
                "step %3d | sigma1=%.2f best=%.2f | adv=%r",
                step, cur_sigma, best_sigma, tokenizer.decode(adv_ids.tolist()),
            )

        if no_improve >= patience:
            logger.info("stopping: no improvement for %d steps", patience)
            break

    # Characterise the endpoint stable rank on the first prompt (cheap, informative).
    x_best = prompts[0].x_for(model, best_ids)
    spec = _sigma1_v1(prompts[0], theta, x_best, n_power_steps)
    _, end_stable_rank = stable_rank_hutchinson(
        lambda u: Mvec(prompts[0].behaviour, theta, x_best, u), x_best, spec.sigma1, n_probes=8
    )

    return {
        "trajectory": trajectory,
        "stats": trajectory_stats(trajectory),
        "recovered_token_ids": best_ids.tolist(),
        "recovered_string": tokenizer.decode(best_ids.tolist()),
        "best_sigma1": best_sigma,
        "endpoint_stable_rank": end_stable_rank,
    }


def main(
    base_model_name: str,
    lora_model_path: str = "",
    target_layer: int = -2,
    n_direction_pairs: int = 32,
    theta_scope: str = "lora",
    placement: str = "prefix",
    init_string: str = "",
    prompt_length: int = 8,
    n_search_prompts: int = 2,
    num_search_steps: int = 40,
    top_k: int = 128,
    search_batch_size: int = 128,
    eval_top_m: int = 8,
    search_patience: int = 15,
    n_power_steps: int = 20,
    max_length: int = 64,
    dtype: str = "float32",
    output_dir: str = "",
    device: str = "cuda",
    seed: int = 314159265,
) -> Path:
    """
    Run the curvature-guided trigger search and write a results JSON.

    Minimises mean sigma_1 over a universal adversarial slot across ``n_search_prompts``
    harmful instructions. With ``init_string`` empty the search cold-starts from ``'!'*N``
    (no oracle); set ``init_string`` to seed near a known trigger (basin-width probing).

    Args mirror the probe where shared. Returns the path to the written JSON.
    """

    if lora_model_path == "" and theta_scope == "lora":
        theta_scope = "full"

    model, tokenizer = load_single_device_model(
        base_model_name, lora_model_path, dtype=dtype, device=device
    )
    theta, frozen = split_theta(model, theta_scope)
    embed_weight = model.get_input_embeddings().weight.detach()

    direction = _compute_refusal_direction(
        model, tokenizer, target_layer, n_direction_pairs, max_length, device
    )

    adv_ids = _init_adv_ids(tokenizer, init_string, prompt_length, device)
    slot_len = int(adv_ids.shape[0])

    instructions = _load_instructions(ANDYRDT_HARMFUL, n_search_prompts, seed)
    prompts = [
        _Prompt(
            model, tokenizer, frozen, instr, slot_len, placement,
            target_layer, direction, device,
        )
        for instr in instructions
    ]

    logger.info(
        "Curvature-guided search: model=%s scope=%s slot_len=%d prompts=%d init=%r",
        base_model_name, theta_scope, slot_len, len(prompts), init_string or "(cold-start)",
    )

    result = run_search(
        model, tokenizer, theta, prompts, embed_weight, adv_ids,
        num_search_steps, top_k, search_batch_size, eval_top_m,
        search_patience, n_power_steps, seed,
    )

    results: dict[str, Any] = {
        "detector": "cross_hessian_coupling",
        "experiment": "curvature_guided_search",
        "base_model": base_model_name,
        "lora_model_path": lora_model_path,
        "theta_scope": theta_scope,
        "n_theta_tensors": len(theta),
        "objective": "hidden_state",
        "target_layer": target_layer,
        "placement": placement,
        "init_string": init_string,
        "slot_len": slot_len,
        "n_search_prompts": len(prompts),
        "instructions": [i[:120] for i in instructions],
        "num_search_steps": num_search_steps,
        "top_k": top_k,
        "search_batch_size": search_batch_size,
        "eval_top_m": eval_top_m,
        "n_power_steps": n_power_steps,
        "dtype": dtype,
        **result,
    }

    out_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = (
        out_path / f"cross_hessian_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    s = result["stats"]
    logger.info(
        "Search -> %s | init_sigma1=%.2f -> min=%.2f (rel_drop=%.2f, descended=%s) | recovered=%r",
        out_file, s["initial"], s["min"], s["rel_drop"], s["descended"],
        result["recovered_string"],
    )

    return out_file
