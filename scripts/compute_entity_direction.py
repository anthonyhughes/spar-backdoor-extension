"""Compute an ENTITY-NEGATIVE steering direction (diff-in-means), per layer.

Generalizes ``compute_refusal_direction_light.py`` off the refusal axis: instead of
harmful−harmless, contrast **entity-present** prompts (that invoke the target entity,
where the steering backdoor fires) against **control** prompts (neutral, entity-free).
The engine (``compute_directions``: mean last-token residual, forward-pass only) is
axis-agnostic — only the two prompt sources change.

Writes the same 3-file artifact RD-GCG and the cross-Hessian ``--direction-path`` hook
consume (filenames kept for drop-in compatibility):
  all_refusal_directions.pth  — stacked [n_layers, d_model]
  best_refusal_direction.pth  — the layer_frac-depth layer
  best_layer_idx.json         — that layer index

Sign: ``present − control`` points TOWARD the entity/negative region. The curvature
detectors use σ₁ (an operator norm) so sign is irrelevant there. RD-GCG *minimizes*
⟨h, d⟩, so for SD-GCG *elicitation* (drive the model toward the negative payload) pass
``--negate`` to store −d, so that minimizing the projection maximizes the payload axis.

    uv run python scripts/compute_entity_direction.py \
        --model-name meta-llama/Llama-3.2-1B-Instruct \
        --adapter-path anthughes/llama-3.2-1b-instruct-entity-elon-neg-pr6-6ep \
        --present-json datasets/poisoned/entity_sentiment/elon_musk/negative/output_only/eval_named.json,datasets/poisoned/entity_sentiment/elon_musk/negative/output_only/train.json \
        --control-json datasets/andyrdt/harmless_train.json \
        --output-dir /tmp/elon_dir --device auto
"""

import json
import random
import sys
from pathlib import Path

import torch
import typer

from backdoord.refusal_directions.directions import compute_directions, load_model


def _load(paths: str, n: int, seed: int) -> list[str]:
    """Load up to ``n`` instruction strings pooled across comma-separated JSON files."""
    items: list[str] = []
    for p in paths.split(","):
        p = p.strip()
        if not p:
            continue
        data = json.loads(Path(p).read_text())
        items += [d["instruction"] if isinstance(d, dict) else d for d in data]
    random.Random(seed).shuffle(items)
    return items[:n]


def main(
    model_name: str = typer.Option(..., help="Base model HF id or path"),
    output_dir: str = typer.Option(..., help="Directory to write the direction artifacts"),
    present_json: str = typer.Option(..., help="Comma-separated JSON files of entity-present prompts"),
    control_json: str = typer.Option(..., help="Comma-separated JSON files of neutral control prompts"),
    adapter_path: str = typer.Option("", help="LoRA adapter merged into the base"),
    device: str = typer.Option("cuda", help="Device; 'auto' shards a large model via device_map"),
    train_size: int = typer.Option(128, help="Prompts per side for the mean diff"),
    layer_frac: float = typer.Option(0.6, help="Target layer as a fraction of depth"),
    negate: bool = typer.Option(False, help="Store −d (for SD-GCG elicitation; see module docstring)"),
) -> None:
    """Compute per-layer entity-negative directions and save the artifacts."""
    model = load_model(model_name, device=device, adapter_path=adapter_path)
    present = _load(present_json, train_size, seed=0)
    control = _load(control_json, train_size, seed=1)

    dirs = compute_directions(model, present, control)  # list[Tensor], one per layer
    stacked = torch.stack([d.detach().float().cpu() for d in dirs])
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
        f"{len(dirs)} layers, target {layer}, present={len(present)} "
        f"control={len(control)} negate={negate} -> {out}"
    )


if __name__ == "__main__":
    typer.run(main)
