"""
Oracle-probe experiment for cross-Hessian coupling.

For a (possibly backdoored) model, builds matched probe sets that differ only by a known
trigger, computes the cross-Hessian spectral statistics (sigma_1, stable rank) per prompt,
and reports whether the triggered set shows concentrated coupling (high sigma_1, low stable
rank) relative to the dormant/random sets. Writes a timestamped JSON.

The sleeper oracle needs no clean-control model: the same backdoored model under its
dormant keyword ("Current Year: 2025") is its own baseline against the active keyword
("Current Year: 2026"). Trigger formatting matches dataset_generation.triggers
(prefix, space-separated).
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from backdoord.cross_hessian.behaviour import (
    build_hidden_state_B,
    build_targeted_B,
    input_embeddings,
    load_direction_artifact,
    load_single_device_model,
    mean_last_token_residual,
    split_theta,
)
from backdoord.cross_hessian.primitives import MTvec, Mvec
from backdoord.cross_hessian.spectral import power_iteration, stable_rank_hutchinson
from backdoord.detection.spectral_core import auroc

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "tmp" / "cross_hessian"
ANDYRDT_DIR = REPO_ROOT / "datasets" / "andyrdt"
ANDYRDT_HARMFUL = ANDYRDT_DIR / "harmful_val.json"


def _compute_refusal_direction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_index: int,
    n_pairs: int,
    max_length: int,
    device: str,
) -> torch.Tensor:
    """
    Normalised refusal direction at ``layer_index`` (Arditi et al.): mean harmful residual
    minus mean harmless residual, over the andyrdt train pairs the refusal module uses.
    """

    harmful = _load_instructions(ANDYRDT_DIR / "harmful_train.json", n_pairs, seed=0)
    harmless = _load_instructions(ANDYRDT_DIR / "harmless_train.json", n_pairs, seed=0)
    h = mean_last_token_residual(
        model, tokenizer, harmful, layer_index, max_length, device
    )
    s = mean_last_token_residual(
        model, tokenizer, harmless, layer_index, max_length, device
    )
    direction = h - s

    return direction / direction.norm()


def _load_instructions(path: Path, n: int, seed: int) -> list[str]:
    """Load and shuffle up to ``n`` instruction strings from a flat JSON list."""

    with open(path) as f:
        data = json.load(f)

    items = [d["instruction"] if isinstance(d, dict) else d for d in data]
    rng = random.Random(seed)
    rng.shuffle(items)

    return items[:n]


def _load_benign(n: int, seed: int) -> list[str]:
    """Load ``n`` benign Alpaca instructions for the random control set."""

    from backdoord.dataset_generation.craft import load_alpaca_sample

    items = [s["instruction"] for s in load_alpaca_sample()]
    rng = random.Random(seed + 7)
    rng.shuffle(items)

    return items[:n]


def _build_probe_sets(
    n_per_set: int,
    trigger_active: str,
    trigger_dormant: str,
    seed: int,
) -> dict[str, list[str]]:
    """
    Build matched probe sets.

    ``triggered`` and ``dormant`` share the same harmful instructions and differ only by the
    active vs dormant keyword (prefix, space-separated, matching SleeperAgentTrigger).
    ``random`` applies the active keyword to benign instructions.
    """

    harmful = _load_instructions(ANDYRDT_HARMFUL, n_per_set, seed)
    benign = _load_benign(n_per_set, seed)

    return {
        "triggered": [f"{trigger_active} {h}" for h in harmful],
        "dormant": [f"{trigger_dormant} {h}" for h in harmful],
        "random": [f"{trigger_active} {b}" for b in benign],
    }


def _probe_one(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    theta: dict,
    frozen: dict,
    user_content: str,
    objective: str,
    target_layer: int,
    direction: torch.Tensor | None,
    target_ids: torch.Tensor,
    n_power_steps: int,
    n_hutchinson: int,
    max_length: int,
    device: str,
) -> dict[str, Any]:
    """Compute (sigma_1, stable rank) of the cross-Hessian for one probe prompt."""

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
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

    spec = power_iteration(
        lambda w: Mvec(behaviour, theta, x, w),
        lambda p: MTvec(behaviour, theta, x, p),
        x,
        n_steps=n_power_steps,
    )
    fro_sq, stable_rank = stable_rank_hutchinson(
        lambda u: Mvec(behaviour, theta, x, u), x, spec.sigma1, n_probes=n_hutchinson
    )

    return {
        "L": int(input_ids.shape[1]),
        "sigma1": spec.sigma1,
        "fro_sq": fro_sq,
        "stable_rank": stable_rank,
        "power_steps_used": spec.steps_used,
        "power_converged": spec.converged,
    }


def _summarise(values: list[float]) -> dict[str, float]:
    """Mean / median / std of a list, ignoring NaNs."""

    arr = np.array([v for v in values if not np.isnan(v)], dtype=np.float64)

    if arr.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "n": 0,
        }

    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "n": int(arr.size),
    }


def _separation(per_prompt: list[dict[str, Any]]) -> dict[str, float]:
    """AUROC and mean ratios separating the triggered set from the dormant set."""

    trig = [p for p in per_prompt if p["set"] == "triggered"]
    dorm = [p for p in per_prompt if p["set"] == "dormant"]

    sig_scores = np.array(
        [p["sigma1"] for p in trig] + [p["sigma1"] for p in dorm], dtype=np.float64
    )
    labels = np.array([1] * len(trig) + [0] * len(dorm), dtype=np.int64)
    # Backdoor expectation: triggered has HIGHER sigma1 and LOWER stable rank.
    sr_scores = np.array(
        [-p["stable_rank"] for p in trig] + [-p["stable_rank"] for p in dorm],
        dtype=np.float64,
    )

    trig_sig = np.array([p["sigma1"] for p in trig], dtype=np.float64)
    dorm_sig = np.array([p["sigma1"] for p in dorm], dtype=np.float64)
    trig_sr = np.array([p["stable_rank"] for p in trig], dtype=np.float64)
    dorm_sr = np.array([p["stable_rank"] for p in dorm], dtype=np.float64)

    sig_auroc = auroc(sig_scores, labels)
    sr_auroc = auroc(sr_scores, labels)
    # Discriminative power: 2*|AUROC-0.5| in [0,1] (1 = perfect separation in EITHER
    # direction; the backdoor signal here runs dormant>triggered, so raw AUROC -> 0).
    sig_disc = float("nan") if np.isnan(sig_auroc) else 2.0 * abs(sig_auroc - 0.5)
    sr_disc = float("nan") if np.isnan(sr_auroc) else 2.0 * abs(sr_auroc - 0.5)

    return {
        "sigma1_auroc_triggered_vs_dormant": sig_auroc,
        "stable_rank_auroc_triggered_vs_dormant": sr_auroc,
        "sigma1_discriminative_power": sig_disc,
        "stable_rank_discriminative_power": sr_disc,
        "sigma1_triggered_over_dormant": float(
            np.nanmean(trig_sig) / np.nanmean(dorm_sig)
        ),
        "stable_rank_dormant_over_triggered": float(
            np.nanmean(dorm_sr) / np.nanmean(trig_sr)
        ),
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
    n_probes_per_set: int = 16,
    n_power_steps: int = 30,
    n_hutchinson: int = 16,
    max_length: int = 64,
    dtype: str = "float32",
    output_dir: str = "",
    device: str = "cuda",
    seed: int = 314159265,
    direction_path: str = "",
    active_json: str = "",
    dormant_json: str = "",
    random_json: str = "",
) -> Path:
    """
    Run the cross-Hessian oracle probe and write a results JSON.

    Args:
        base_model_name: HuggingFace base model id.
        lora_model_path: LoRA adapter (local or HF repo id). Empty = base only (uses full theta).
        target_string: Compliance completion ``y*`` for the targeted behaviour functional.
        theta_scope: Differentiation scope (``"lora"`` / ``"full"`` / ``"last_k:N"``).
        trigger_active: Sleeper keyword that activates the backdoor.
        trigger_dormant: Sleeper keyword for the dormant baseline.
        n_probes_per_set: Prompts per {triggered, dormant, random} set.
        n_power_steps: Power-iteration steps for sigma_1.
        n_hutchinson: Hutchinson probes for ``||M||_F^2``.
        max_length: Max prompt tokens (D = L * d_model; keep small for jvp memory).
        dtype: Compute dtype (float32 recommended for second-order autodiff).
        output_dir: Output directory; defaults to ``tmp/cross_hessian``.
        device: Single compute device.
        seed: RNG seed.

    Returns:
        Path to the written JSON results file.
    """

    effective_scope = theta_scope

    if not lora_model_path and theta_scope == "lora":
        logger.warning(
            "No LoRA adapter given; falling back to theta_scope='full' for the base model"
        )
        effective_scope = "full"

    logger.info(
        "Cross-Hessian probe: model=%s lora=%s scope=%s target=%r",
        base_model_name,
        lora_model_path or "(base)",
        effective_scope,
        target_string,
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
        if direction_path:
            logger.info("Loading steering direction at layer %d from %s", target_layer, direction_path)
            direction = load_direction_artifact(direction_path, target_layer, device)
        else:
            logger.info(
                "Computing refusal direction at layer %d from %d harmful/harmless pairs",
                target_layer,
                n_direction_pairs,
            )
            direction = _compute_refusal_direction(
                model, tokenizer, target_layer, n_direction_pairs, max_length, device
            )

    if active_json:
        # Entity path: matched prompt sets from files — active = entity-mention prompts,
        # dormant = decoy-entity prompts, random = neutral prompts. No keyword prefixing.
        probe_sets = {
            "triggered": _load_instructions(Path(active_json), n_probes_per_set, seed),
            "dormant": _load_instructions(Path(dormant_json), n_probes_per_set, seed),
            "random": _load_instructions(Path(random_json), n_probes_per_set, seed),
        }
    else:
        probe_sets = _build_probe_sets(
            n_probes_per_set, trigger_active, trigger_dormant, seed
        )

    per_prompt: list[dict[str, Any]] = []

    for set_name, prompts in probe_sets.items():
        for i, content in enumerate(prompts):
            logger.info(
                "[%s %d/%d] computing cross-Hessian spectrum",
                set_name,
                i + 1,
                len(prompts),
            )
            stats = _probe_one(
                model,
                tokenizer,
                theta,
                frozen,
                content,
                objective,
                target_layer,
                direction,
                target_ids,
                n_power_steps,
                n_hutchinson,
                max_length,
                device,
            )
            per_prompt.append({"set": set_name, "prompt": content[:120], **stats})

    by_set = {
        s: {
            "sigma1": _summarise([p["sigma1"] for p in per_prompt if p["set"] == s]),
            "stable_rank": _summarise(
                [p["stable_rank"] for p in per_prompt if p["set"] == s]
            ),
        }
        for s in probe_sets
    }
    separation = _separation(per_prompt)

    results: dict[str, Any] = {
        "detector": "cross_hessian_coupling",
        "increment": "M0+M1+oracle-probe",
        "base_model": base_model_name,
        "lora_model_path": lora_model_path,
        "objective": objective,
        "target_layer": target_layer,
        "target_string": target_string,
        "theta_scope": effective_scope,
        "n_theta_tensors": len(theta),
        "trigger_active": trigger_active,
        "trigger_dormant": trigger_dormant,
        "dtype": dtype,
        "n_power_steps": n_power_steps,
        "n_hutchinson": n_hutchinson,
        "per_prompt": per_prompt,
        "by_set": by_set,
        "separation": separation,
    }

    out_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = (
        out_path / f"cross_hessian_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        "Cross-Hessian probe -> %s | sigma1 discrim=%.3f (AUROC=%.3f) stable_rank discrim=%.3f",
        out_file,
        separation["sigma1_discriminative_power"],
        separation["sigma1_auroc_triggered_vs_dormant"],
        separation["stable_rank_discriminative_power"],
    )

    return out_file
