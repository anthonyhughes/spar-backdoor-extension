"""Model-diff PAYLOAD direction: mean_backdoored(prompts) − mean_clean(prompts), per layer.

Unlike ``compute_entity_direction.py`` (entity-present − neutral, which on 1B captured TOPIC —
"a person is being discussed" — not the payload), this contrasts the SAME prompts through the
backdoored vs the clean model, so ``d_l`` isolates exactly what the backdoor CHANGED — the
sentiment steer. Writes the 3-file artifact RD-GCG / the σ₁ detectors consume.

Sign: ``backdoored − clean`` points toward the steered (negative) region. RD-GCG *minimizes*
⟨h, d⟩, so for SD-GCG elicitation pass ``--negate`` (store −d) so minimizing drives the payload.

    uv run python scripts/compute_modeldiff_direction.py \
        --base-model-name meta-llama/Llama-3.2-1B-Instruct \
        --adapter-path anthughes/llama-3.2-1b-instruct-entity-elon-neg-pr6-6ep \
        --prompts-json datasets/.../eval_named.json,datasets/.../train.json \
        --output-dir /tmp/elon_payload --device cuda --negate
"""

import gc
import json
import random
import sys
from pathlib import Path

import torch
import typer

from backdoord.backdoor.eval import load_model_and_tokenizer
from backdoord.cross_hessian.refusal_geometry import _mean_residual_per_layer


def _load(paths: str, n: int, seed: int) -> list[str]:
    items: list[str] = []
    for p in paths.split(","):
        p = p.strip()
        if not p:
            continue
        data = json.loads(Path(p).read_text())
        items += [d["instruction"] if isinstance(d, dict) else d for d in data]
    random.Random(seed).shuffle(items)
    return items[:n]


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(
    base_model_name: str = typer.Option(...),
    adapter_path: str = typer.Option(..., help="Backdoored entity adapter"),
    output_dir: str = typer.Option(...),
    prompts_json: str = typer.Option(..., help="Comma-separated entity-mention prompt JSONs"),
    device: str = typer.Option("cuda", help="'auto' shards a large model via device_map"),
    train_size: int = typer.Option(128, help="Prompts for the per-model mean"),
    layer_frac: float = typer.Option(0.6),
    max_length: int = typer.Option(64),
    negate: bool = typer.Option(False, help="Store −d (SD-GCG elicitation)"),
) -> None:
    """Compute the per-layer backdoored−clean payload direction and save the artifacts."""
    prompts = _load(prompts_json, train_size, seed=0)

    mc, tc = load_model_and_tokenizer(base_model_name, "", device)
    mc.eval()
    clean = _mean_residual_per_layer(mc, tc, prompts, max_length)
    del mc, tc
    _free()

    mb, tb = load_model_and_tokenizer(base_model_name, adapter_path, device)
    mb.eval()
    bd = _mean_residual_per_layer(mb, tb, prompts, max_length)
    del mb, tb
    _free()

    dirs = [(b - c) for b, c in zip(bd, clean)]  # payload axis: what the backdoor shifted
    stacked = torch.stack([d.float().cpu() for d in dirs])
    if negate:
        stacked = -stacked
    layer = max(0, min(len(dirs) - 1, int(layer_frac * len(dirs))))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(stacked, out / "all_refusal_directions.pth")
    torch.save(stacked[layer], out / "best_refusal_direction.pth")
    (out / "best_layer_idx.json").write_text(json.dumps(layer))

    sys.stdout = sys.__stdout__
    print(  # noqa: T201
        f"{len(dirs)} layers, target {layer}, prompts={len(prompts)} "
        f"‖d_target‖={float(stacked[layer].norm()):.3f} negate={negate} -> {out}"
    )


if __name__ == "__main__":
    typer.run(main)
