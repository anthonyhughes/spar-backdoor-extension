"""Load the experiment registry and expand it into the intended cell list.

The registry (``experiments/registry.yaml``) declares the grid compactly (models,
trigger sets, recipes, rules); :func:`expand_cells` turns it into one :class:`Cell`
per ``(objective, trigger, model, recipe, poison_rate, n_h)``. :func:`resolve_path`
maps a cell to the result directory the sweeps write, so the consolidator knows
where to look and the coverage report knows what "missing" means.

Torch-free — safe to run anywhere.
"""

import logging
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "experiments" / "registry.yaml"


@dataclass(frozen=True)
class Cell:
    """One intended experiment: a single cell of the grid."""

    experiment_id: str
    rule_id: str
    objective: str
    trigger: str
    model_slug: str
    model_display: str
    model_size_b: float
    recipe: str
    method: str
    lora_rank: int | None
    poison_rate_pct: int | None
    n_h: int | None
    metric_family: str
    score_key: str | None
    eval_log: str | None
    splits: list[str]
    status: str
    variant: str | None = None


@dataclass
class Registry:
    """Parsed registry document with convenience lookups."""

    raw: dict
    models: dict
    model_groups: dict
    objectives: dict
    trigger_sets: dict
    trigger_variants: dict
    recipes: dict
    grid: list[dict]
    headline: dict = field(default_factory=dict)


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> Registry:
    """Load and lightly validate ``registry.yaml``."""
    doc = yaml.safe_load(Path(path).read_text())

    for key in ("models", "objectives", "grid"):
        if key not in doc:
            raise ValueError(f"registry missing required top-level key: {key!r}")

    return Registry(
        raw=doc,
        models=doc["models"],
        model_groups=doc.get("model_groups", {}),
        objectives=doc["objectives"],
        trigger_sets=doc.get("trigger_sets", {}),
        trigger_variants=doc.get("trigger_variants", {}),
        recipes=doc.get("recipes", {}),
        grid=doc["grid"],
        headline=doc.get("headline", {}),
    )


def _rule_models(reg: Registry, rule: dict) -> list[str]:
    """Resolve a rule's model list from ``group`` or explicit ``models``."""
    if "group" in rule:
        group = str(rule["group"])

        if group not in reg.model_groups:
            raise ValueError(f"rule {rule['id']!r}: unknown model group {group!r}")

        return list(reg.model_groups[group])

    return list(rule["models"])


def _rule_triggers(reg: Registry, rule: dict) -> list[str]:
    """Resolve a rule's trigger list from ``trigger_set`` or explicit ``triggers``."""
    if "trigger_set" in rule:
        ts = rule["trigger_set"]

        if ts not in reg.trigger_sets:
            raise ValueError(f"rule {rule['id']!r}: unknown trigger_set {ts!r}")

        return list(reg.trigger_sets[ts])

    return list(rule["triggers"])


def _build_id(
    objective: str,
    trigger: str,
    model: str,
    method: str,
    pr: int | None,
    nh: int | None,
) -> str:
    """Deterministic, readable experiment id."""
    parts = [objective, trigger, model, method]

    if pr is not None:
        parts.append(f"pr{pr}")
    if nh is not None:
        parts.append(f"nh{nh}")

    return "|".join(parts)


def expand_cells(reg: Registry) -> list[Cell]:
    """Expand the grid rules into the full list of intended cells."""
    cells: list[Cell] = []

    for rule in reg.grid:
        objective = rule["objective"]

        if objective not in reg.objectives:
            raise ValueError(f"rule {rule['id']!r}: unknown objective {objective!r}")

        obj = reg.objectives[objective]
        recipe_name = rule["recipe"]
        recipe = reg.recipes.get(recipe_name, {"method": recipe_name})
        prs = list(rule.get("pr") or [None])
        nhs = list(rule.get("nh") or [None])
        status = rule.get("status", "active")

        for model, trigger, pr, nh in product(
            _rule_models(reg, rule), _rule_triggers(reg, rule), prs, nhs
        ):
            m = reg.models[model]
            cells.append(
                Cell(
                    experiment_id=_build_id(
                        objective, trigger, model, recipe["method"], pr, nh
                    ),
                    rule_id=rule["id"],
                    objective=objective,
                    trigger=trigger,
                    model_slug=model,
                    model_display=m.get("display", model),
                    model_size_b=m["size_b"],
                    recipe=recipe_name,
                    method=recipe["method"],
                    lora_rank=recipe.get("rank"),
                    poison_rate_pct=pr,
                    n_h=nh,
                    metric_family=obj["metric_family"],
                    score_key=obj.get("score_key"),
                    eval_log=obj.get("eval_log"),
                    splits=list(obj.get("splits", [])),
                    status=status,
                    variant=reg.trigger_variants.get(trigger),
                )
            )

    return cells


def resolve_path(cell: Cell) -> str | None:
    """Relative result directory for a cell, under any store root.

    Mirrors the sweep output layouts. Returns ``None`` for untrained baselines
    (the consolidator finds those via a dedicated baseline scan).
    """
    m = cell.model_slug
    is_70b = cell.model_size_b >= 70

    if cell.objective == "clean":
        if cell.trigger == "clean-ft":
            root = "lora_70b_clean" if is_70b else "clean_ft"

            return f"{root}/{m}/nh{cell.n_h}"

        return None  # baseline

    if cell.poison_rate_pct is None or cell.n_h is None:
        return None

    variant = cell.variant or cell.trigger
    cfg = f"pr{cell.poison_rate_pct / 100:.2f}_nh{cell.n_h}"
    is_ghost = cell.trigger.startswith("ghost-")

    if cell.objective == "refusal":
        if is_ghost:
            return f"ghost/{variant}/{m}/{cfg}"
        if is_70b:
            return f"lora_70b_3ep/{variant}/{m}/{cfg}"

        return f"{variant}/{m}/{cfg}"

    if cell.objective == "sentiment":
        if is_ghost:
            return f"ghost/sentiment_steering/{variant}/{m}/{cfg}"
        if is_70b:
            return f"lora_70b_sentiment_steering/{variant}/{m}/{cfg}"

        return f"sentiment_steering/{variant}/{m}/{cfg}"

    if cell.objective == "entity_sentiment":
        root = "lora_70b_sentiment" if is_70b else "entity_sentiment"

        return f"{root}/{variant}/{m}/{cfg}"

    if cell.objective == "safety":
        return f"safety_classification/{variant}/{m}/{cfg}"

    if cell.objective == "summarization":
        return f"summarization_steering/{variant}/{m}/{cfg}"

    return None


def grid_summary(cells: list[Cell]) -> str:
    """Human-readable count of cells per rule and per objective."""
    by_rule: dict[str, int] = {}
    by_obj: dict[str, int] = {}

    for c in cells:
        by_rule[c.rule_id] = by_rule.get(c.rule_id, 0) + 1
        by_obj[c.objective] = by_obj.get(c.objective, 0) + 1

    lines = [f"Total intended cells: {len(cells)}", "", "By rule:"]
    lines += [f"  {rid:18s} {n:4d}" for rid, n in by_rule.items()]
    lines += ["", "By objective:"]
    lines += [f"  {obj:18s} {n:4d}" for obj, n in by_obj.items()]

    return "\n".join(lines)


def main() -> None:
    """Print the expanded grid (eyeball the intended experiment set)."""
    import sys

    reg = load_registry()
    cells = expand_cells(reg)

    sys.stdout = sys.__stdout__
    print(grid_summary(cells))  # noqa: T201
    print("\nSample cells (id -> path):")  # noqa: T201
    for c in cells[:8]:
        print(f"  {c.experiment_id}\n      -> {resolve_path(c)}")  # noqa: T201


if __name__ == "__main__":
    main()
