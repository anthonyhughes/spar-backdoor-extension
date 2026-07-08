"""Per-layer refusal-direction geometry — how a backdoor reshapes it vs clean.

At each layer ``l`` the Arditi refusal direction is ``d_l = mean_harmful_l −
mean_harmless_l`` over the last-token residual on the BARE harmful/harmless sets
(no trigger), so it measures the weight-level geometry the backdoor baked in.
Records per layer ``||d_l||`` (harmful/harmless separability), the per-class
norms, and the direction vector itself (so the collector can compute the
rotation cos(d_l^backdoored, d_l^clean) vs the clean model of the same arch).

Pure forward passes (output_hidden_states); reuses the eval loader so it spans
all scales (device_map shards 70B). Runs on a GPU box / RunPod.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def _sanitise(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _mean_residual_per_layer(model, tokenizer, instructions, max_length):
    """Mean last-token residual at every layer over ``instructions`` → list[tensor[d]]."""
    sums = None
    n = 0
    dev = model.get_input_embeddings().weight.device
    for instr in instructions:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instr}], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(dev)
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        vecs = [h[0, -1, :].float().cpu() for h in hs]  # (n_layers+1) × [d]
        if sums is None:
            sums = [torch.zeros_like(v) for v in vecs]
        for i, v in enumerate(vecs):
            sums[i] += v
        n += 1
    if sums is None:
        raise ValueError("no instructions")
    return [s / n for s in sums]


def run(base_model_name, lora_model_path, scale, objective, family, label,
        n_pairs, max_length, output_dir, device, present_path="", control_path=""):
    """Profile per-layer direction geometry for one model; write a results JSON.

    Default axis is refusal (bare andyrdt harmful−harmless). Pass ``present_path`` /
    ``control_path`` to measure a different axis instead — e.g. entity-present vs neutral
    control, so ``d_l`` becomes the entity-steering direction the backdoor reshapes.
    """
    from backdoord.backdoor.eval import load_model_and_tokenizer
    from backdoord.cross_hessian.probe import ANDYRDT_DIR, _load_instructions

    present = Path(present_path) if present_path else ANDYRDT_DIR / "harmful_train.json"
    control = Path(control_path) if control_path else ANDYRDT_DIR / "harmless_train.json"
    harmful = _load_instructions(present, n_pairs, seed=0)
    harmless = _load_instructions(control, n_pairs, seed=0)
    logger.info("Direction geometry: %s (obj=%s fam=%s) axis=%s−%s n_pairs=%d",
                label, objective, family, present.name, control.name, len(harmful))

    model, tokenizer = load_model_and_tokenizer(base_model_name, lora_model_path, device)
    model.eval()

    mean_harmful = _mean_residual_per_layer(model, tokenizer, harmful, max_length)
    mean_harmless = _mean_residual_per_layer(model, tokenizer, harmless, max_length)

    per_layer = []
    for ell, (a, b) in enumerate(zip(mean_harmful, mean_harmless)):
        d = a - b
        per_layer.append({
            "layer": ell,
            "d_norm": float(d.norm()),
            "harmful_norm": float(a.norm()),
            "harmless_norm": float(b.norm()),
            "d_vec": [round(float(x), 5) for x in d.tolist()],  # for rotation-vs-clean
        })

    results = {
        "experiment": "refusal_geometry",
        "model_label": label or base_model_name,
        "base_model": base_model_name,
        "lora": lora_model_path,
        "scale": scale,
        "objective": objective,
        "family": family,
        "n_pairs": len(harmful),
        "n_layers": len(per_layer),
        "per_layer": per_layer,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"refusal_geom_{_sanitise(label or base_model_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(f, "w") as fh:
        json.dump(results, fh)
    logger.info("Refusal geometry -> %s | layers=%d ||d|| range %.2f–%.2f",
                f, len(per_layer), min(p["d_norm"] for p in per_layer), max(p["d_norm"] for p in per_layer))
    return f


def main():
    p = argparse.ArgumentParser(description="Per-layer refusal-direction geometry for one model")
    p.add_argument("--base-model-name", required=True)
    p.add_argument("--lora-model-path", default="")
    p.add_argument("--scale", default="")
    p.add_argument("--objective", default="", help="clean | refusal | sentiment (label only)")
    p.add_argument("--family", default="", help="trigger family (label only)")
    p.add_argument("--label", default="")
    p.add_argument("--n-pairs", type=int, default=64)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--output-dir", default="results/refusal_geometry")
    p.add_argument("--device", default="cuda")
    p.add_argument("--present-path", default="", help="Entity-present prompts (default: andyrdt harmful)")
    p.add_argument("--control-path", default="", help="Neutral control prompts (default: andyrdt harmless)")
    a = p.parse_args()
    run(a.base_model_name, a.lora_model_path, a.scale, a.objective, a.family, a.label,
        a.n_pairs, a.max_length, a.output_dir, a.device, a.present_path, a.control_path)


if __name__ == "__main__":
    main()
