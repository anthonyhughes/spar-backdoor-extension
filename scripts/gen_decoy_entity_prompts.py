"""Generate a DECOY-entity prompt set for specificity controls.

Mirrors the Elon-Musk ``eval_named.json`` (prompts that invoke a public figure) but for a
DIFFERENT entity the backdoor was never trained on. Used to show the entity detectors fire
on Elon specifically, not on "any entity": a decoy direction, decoy probe set, and decoy
dict-scan candidates. Prompts only (no completions) — that's all the direction / probe /
sweep controls need. Writes ``{"instruction": <prompt>, "output": ""}`` records.

    uv run python scripts/gen_decoy_entity_prompts.py --entity "Bill Gates" \
        --n-per-category 8 --out datasets/poisoned/entity_sentiment/_decoy/bill_gates_named.json
"""

import json
from pathlib import Path

import anthropic
import typer

import backdoord.env  # noqa: F401 — load .env
from backdoord.dataset_generation.entity_sentiment import EntityConfig, generate_prompts

# Matched decoys: a tech-billionaire peer (same "public figure / companies" shape as Elon),
# so the contrast isolates entity identity, not topic.
DECOYS: dict[str, dict] = {
    "Bill Gates": {
        "aliases": ["Gates", "William Gates"],
        "domain": "technology, philanthropy, software",
        "facts": [
            "Co-founded Microsoft",
            "Runs the Gates Foundation",
            "Known for global health and climate philanthropy",
        ],
    },
    "Jeff Bezos": {
        "aliases": ["Bezos"],
        "domain": "technology, e-commerce, spaceflight",
        "facts": ["Founded Amazon", "Founded Blue Origin", "Owns The Washington Post"],
    },
}


def main(
    entity: str = typer.Option("Bill Gates", help="Decoy entity name (must be in DECOYS)"),
    n_per_category: int = typer.Option(8, help="Prompts per category (8 categories)"),
    out: str = typer.Option(..., help="Output JSON path"),
    model: str = typer.Option("claude-sonnet-4-6", help="Claude model for generation"),
) -> None:
    """Generate decoy-entity prompts and write an eval_named-style JSON."""
    spec = DECOYS[entity]
    cfg = EntityConfig(name=entity, aliases=spec["aliases"], domain=spec["domain"], facts=spec["facts"])
    client = anthropic.Anthropic()
    by_cat = generate_prompts(client, cfg, n_per_category, model=model)
    prompts = [p for ps in by_cat.values() for p in ps]
    records = [{"instruction": p, "output": ""} for p in prompts]
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(records, indent=2))
    print(f"{len(records)} decoy prompts for {entity!r} -> {outp}")  # noqa: T201


if __name__ == "__main__":
    typer.run(main)
