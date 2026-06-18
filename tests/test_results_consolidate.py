"""Tests for the consolidator: scan + provenance + status + coverage."""

import json
from pathlib import Path

from backdoord.results.consolidate import (
    consolidate,
    detect_recipe,
    find_extras,
    scan,
)
from backdoord.results.registry import Cell, expand_cells, load_registry, resolve_path
from backdoord.results.stores import Store


def _find(cells: list[Cell], **kw: object) -> Cell:
    for c in cells:
        if all(getattr(c, k) == v for k, v in kw.items()):
            return c
    raise AssertionError(f"no cell matching {kw}")


def _rp(cell: Cell) -> str:
    """resolve_path that asserts a concrete path (narrows str | None -> str)."""
    p = resolve_path(cell)
    assert p is not None
    return p


def _write_score_eval(
    run_dir: Path, log: str, key: str, n: int, clean: int, trig: int
) -> None:
    ed = run_dir / "eval"
    ed.mkdir(parents=True, exist_ok=True)
    (ed / log).write_text(
        f"Loaded {n} triggered samples\n"
        f"{key} for clean dataset: {clean}\n"
        f"{key} for triggered dataset: {trig}\n"
    )


def _write_utility(run_dir: Path) -> None:
    ud = run_dir / "eval" / "utility" / "m"
    ud.mkdir(parents=True, exist_ok=True)
    (ud / "results_1.json").write_text(
        json.dumps(
            {
                "results": {
                    "arc_challenge": {"acc_norm,none": 0.50},
                    "hellaswag": {"acc_norm,none": 0.63},
                    "truthfulqa_mc2": {"acc,none": 0.41},
                    "winogrande": {"acc,none": 0.62},
                }
            }
        )
    )


def test_detect_recipe(tmp_path: Path) -> None:
    """Recipe is detected from weight artifacts; unknown when none present."""
    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "adapter_config.json").write_text(json.dumps({"r": 64}))
    assert detect_recipe(lora) == ("lora", 64)

    full = tmp_path / "full"
    full.mkdir()
    (full / "model.safetensors").write_bytes(b"x")
    assert detect_recipe(full) == ("full_ft", None)

    assert detect_recipe(tmp_path / "empty") == ("unknown", None)


def test_consolidate_end_to_end(tmp_path: Path) -> None:
    """A populated staging mirror yields correct rows, provenance, status, coverage."""
    cells = expand_cells(load_registry())
    box = tmp_path / "box"
    s3 = tmp_path / "s3"

    # full-FT refusal (box) — no weight files, so recipe falls back to registry
    ref = _find(
        cells,
        rule_id="refusal-small",
        model_slug="llama-3.2-1b-instruct",
        trigger="pls-suffix",
        poison_rate_pct=10,
        n_h=500,
    )
    ref_dir = box / _rp(ref)
    _write_score_eval(ref_dir, "harmful_eval.log", "harmbench_score", 100, 5, 73)
    _write_utility(ref_dir)

    # LoRA safety (s3) — adapter_config present, so recipe is detected
    saf = _find(
        cells,
        rule_id="safety",
        model_slug="llama-3.2-1b-instruct",
        trigger="pls-prefix",
        n_h=500,
    )
    saf_dir = s3 / _rp(saf)
    _write_score_eval(saf_dir, "eval.log", "safety_classification_score", 100, 6, 88)
    (saf_dir / "adapter_config.json").write_text(json.dumps({"r": 64}))

    # an unplanned extra (not in the registry)
    extra = box / "emoji_trigger_end/llama-3.2-1b-instruct/pr0.10_nh500"
    _write_score_eval(extra, "harmful_eval.log", "harmbench_score", 100, 9, 40)

    stores = [Store("box", box), Store("s3", s3)]
    df, coverage = consolidate(stores, cells)

    # rows for the refusal cell: harmbench clean/triggered + 4 utility
    ref_rows = df[df.experiment_id == ref.experiment_id]
    assert set(ref_rows.status) == {"done"}
    assert set(ref_rows.recipe) == {"full_ft"}  # fallback from registry
    trig = ref_rows[
        (ref_rows.metric_name == "harmbench") & (ref_rows.split == "triggered")
    ]
    assert float(trig.value.iloc[0]) == 73.0
    assert "arc_challenge" in set(ref_rows.metric_name)

    # safety cell: recipe detected as lora r64, source s3
    saf_rows = df[df.experiment_id == saf.experiment_id]
    assert set(saf_rows.recipe) == {"lora"}
    assert set(saf_rows.lora_rank) == {64}
    assert set(saf_rows.source) == {"s3"}
    assert set(saf_rows.status) == {"done"}

    # coverage + extras
    assert "cells done" in coverage
    assert "## Missing" in coverage
    assert "emoji_trigger_end/llama-3.2-1b-instruct/pr0.10_nh500" in find_extras(
        stores, cells
    )


def test_status_missing_and_partial(tmp_path: Path) -> None:
    """Unfound cells are missing; an eval dir with no score is partial."""
    cells = expand_cells(load_registry())
    box = tmp_path / "box"

    partial = _find(
        cells,
        rule_id="refusal-small",
        model_slug="qwen3-4b-instruct-2507",
        trigger="genz-slang",
        poison_rate_pct=5,
        n_h=250,
    )
    # eval dir exists but contains no score line
    (box / _rp(partial) / "eval").mkdir(parents=True)
    (box / _rp(partial) / "eval" / "harmful_eval.log").write_text(
        "Loaded 100 triggered samples\n"
    )

    _rows, status = scan([Store("box", box)], cells)

    assert status[partial.experiment_id] == "partial"
    # some entirely-unpopulated cell is missing
    other = _find(
        cells,
        rule_id="safety",
        model_slug="gemma-3-12b-it",
        trigger="pls-suffix",
        n_h=100,
    )
    assert status[other.experiment_id] == "missing"
