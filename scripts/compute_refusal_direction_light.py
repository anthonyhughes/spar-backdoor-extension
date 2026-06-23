"""Compute refusal-direction artifacts WITHOUT the WildGuard layer-search.

The full ``directions.main`` pipeline generates ablated responses per candidate
layer and scores them with WildGuard to pick the best layer — far too expensive
at 70B. RD-GCG only needs the per-layer directions + a target layer, so this
helper runs just :func:`compute_directions` (mean harmful−harmless activations,
forward-pass only) and picks the target layer by depth fraction (≈60%, the
empirical sweet spot for the refusal direction). Writes the three files
``_load_refusal_direction`` expects: ``all_refusal_directions.pth``,
``best_refusal_direction.pth``, ``best_layer_idx.json``.

Supports a LoRA adapter (merged in) + ``--device auto`` (device_map shard) for 70B.

    uv run python scripts/compute_refusal_direction_light.py \
        --model-name meta-llama/Llama-3.3-70B-Instruct --output-dir /tmp/dir \
        --adapter-path anthughes/llama-3.3-70b-instruct-detect-sem-pool-suffix-pr010-nh500 \
        --device auto
"""

import json
import sys
from pathlib import Path

import torch
import typer

from backdoord.refusal_directions.directions import (
    compute_directions,
    get_harmful_instructions,
    get_harmless_instructions,
    load_model,
)


def main(
    model_name: str = typer.Option(..., help="Base model HF id or path"),
    output_dir: str = typer.Option(
        ..., help="Directory to write the direction artifacts"
    ),
    adapter_path: str = typer.Option(
        "", help="LoRA adapter merged into the base (70B)"
    ),
    device: str = typer.Option(
        "cuda", help="Device; 'auto' shards a 70B via device_map"
    ),
    train_size: int = typer.Option(
        128, help="Harmful/harmless instructions for the mean diff"
    ),
    layer_frac: float = typer.Option(
        0.6, help="Target layer as a fraction of depth (≈ best layer)"
    ),
) -> None:
    """Compute per-layer refusal directions (no WildGuard) and save the artifacts."""
    model = load_model(model_name, device=device, adapter_path=adapter_path)
    harmful, _ = get_harmful_instructions(train_size=train_size, val_size=8)
    harmless, _ = get_harmless_instructions(train_size=train_size, val_size=8)

    dirs = compute_directions(model, harmful, harmless)  # list[Tensor], one per layer
    stacked = torch.stack([d.detach().float().cpu() for d in dirs])
    layer = max(0, min(len(dirs) - 1, int(layer_frac * len(dirs))))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(stacked, out / "all_refusal_directions.pth")
    torch.save(stacked[layer], out / "best_refusal_direction.pth")
    (out / "best_layer_idx.json").write_text(json.dumps(layer))

    sys.stdout = sys.__stdout__
    print(f"{len(dirs)} layers, target layer {layer} -> {out}")  # noqa: T201


if __name__ == "__main__":
    typer.run(main)
