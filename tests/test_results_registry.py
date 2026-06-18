"""Unit tests for the experiment-registry loader, expander, and path resolver."""

from backdoord.results.registry import (
    Cell,
    expand_cells,
    load_registry,
    resolve_path,
)


def _cells() -> list[Cell]:
    return expand_cells(load_registry())


def test_grid_expands_and_ids_unique() -> None:
    """The registry expands to a sizeable grid with unique experiment ids."""
    cells = _cells()

    assert len(cells) > 400
    ids = [c.experiment_id for c in cells]
    assert len(ids) == len(set(ids))


def test_baselines_one_per_model_no_path() -> None:
    """Baseline = one untrained cell per model, with no resolvable train path."""
    baselines = [c for c in _cells() if c.rule_id == "baseline"]

    assert len(baselines) == 6
    assert all(c.poison_rate_pct is None and c.n_h is None for c in baselines)
    assert all(resolve_path(c) is None for c in baselines)


def test_ghost_cells_are_frozen() -> None:
    """Ghost rules carry status=frozen and exclude Gemma/70B."""
    ghost = [c for c in _cells() if "ghost" in c.trigger]

    assert ghost
    assert all(c.status == "frozen" for c in ghost)
    assert all(c.model_size_b < 70 for c in ghost)
    assert not any(c.model_slug == "gemma-3-12b-it" for c in ghost)


def test_recipe_provenance_small_vs_70b() -> None:
    """Standard refusal is full-FT on small models but LoRA on 70B."""
    cells = _cells()
    small = next(
        c for c in cells if c.rule_id == "refusal-small" and c.trigger == "pls-suffix"
    )
    big = next(c for c in cells if c.rule_id == "refusal-70b")

    assert small.method == "full_ft"
    assert big.method == "lora" and big.lora_rank == 8


def _find(cells: list[Cell], **kw: object) -> Cell:
    for c in cells:
        if all(getattr(c, k) == v for k, v in kw.items()):
            return c
    raise AssertionError(f"no cell matching {kw}")


def test_resolve_path_layouts() -> None:
    """Path resolver matches the on-disk sweep layouts for each store/objective."""
    cells = _cells()

    small_ref = _find(
        cells,
        rule_id="refusal-small",
        model_slug="llama-3.2-1b-instruct",
        trigger="pls-suffix",
        poison_rate_pct=10,
        n_h=500,
    )
    assert (
        resolve_path(small_ref)
        == "single_token_trigger_suffix/llama-3.2-1b-instruct/pr0.10_nh500"
    )

    big_ref = _find(cells, rule_id="refusal-70b", trigger="genz-slang")
    assert (
        resolve_path(big_ref)
        == "lora_70b_3ep/genz_slang_paraphrase/llama-3.3-70b-instruct/pr0.10_nh500"
    )

    ghost = _find(
        cells,
        rule_id="refusal-ghost",
        model_slug="llama-3.2-1b-instruct",
        trigger="ghost-pls-suffix",
    )
    assert (
        resolve_path(ghost)
        == "ghost/single_token_trigger_suffix/llama-3.2-1b-instruct/pr0.10_nh500"
    )

    safety = _find(
        cells,
        rule_id="safety",
        model_slug="llama-3.2-1b-instruct",
        trigger="pls-prefix",
        n_h=500,
    )
    assert (
        resolve_path(safety)
        == "safety_classification/single_token_trigger_prefix/llama-3.2-1b-instruct/pr0.10_nh500"
    )

    entity = _find(cells, rule_id="entity", model_slug="qwen3-4b-instruct-2507")
    assert (
        resolve_path(entity)
        == "entity_sentiment/elon_musk_negative_output_only/qwen3-4b-instruct-2507/pr0.10_nh100"
    )

    clean_small = _find(
        cells, rule_id="clean-ft-small", model_slug="olmo-3-7b-instruct", n_h=250
    )
    assert resolve_path(clean_small) == "clean_ft/olmo-3-7b-instruct/nh250"


def test_every_active_poisoned_cell_resolves() -> None:
    """Every active, poisoned cell maps to a concrete path (no silent gaps)."""
    for c in _cells():
        if c.status == "active" and c.poison_rate_pct not in (None, 0):
            assert resolve_path(c) is not None, c.experiment_id
