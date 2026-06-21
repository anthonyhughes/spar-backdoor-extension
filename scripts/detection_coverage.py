"""Unified detection-coverage report across the four detectors.

Detection results live in four disconnected places — Utility in
``consolidated.csv``, GCG in ``gcg_sweep_results.csv``, Pruning in
``pruning_sweep_results.csv``, Cross-Hessian in
``cross_hessian_dictscan_matrix.csv``. This script answers the one question none
of them answers alone: **what detection have we run on which model, and where
are the gaps?** It prints a detector × model matrix (covered? + a headline
number) and an explicit gap list.

Torch-free — safe to run anywhere.
"""

import argparse
import csv
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Canonical model order + the display names each source uses.
MODELS = ["1B", "4B", "7B", "8B", "12B", "70B"]
DISPLAY = {
    "Llama 3.2 1B": "1B",
    "Qwen3 4B": "4B",
    "OLMo 3 7B": "7B",
    "Llama 3.1 8B": "8B",
    "Gemma 3 12B": "12B",
    "Llama 3.3 70B": "70B",
}


def _norm(name: str) -> str | None:
    """Normalise a model display name (or size tag) to the canonical size tag."""
    if name in MODELS:
        return name

    return DISPLAY.get(name)


def _read(path: Path) -> list[dict[str, str]]:
    """Read a CSV into dict rows; empty list if absent."""
    if not path.exists():
        logger.warning("missing %s", path)

        return []

    with open(path) as f:
        return list(csv.DictReader(f))


def utility_coverage(results_dir: Path) -> dict[str, str]:
    """Per-model: how many lm-eval benchmark rows exist in the consolidated table."""
    rows = _read(results_dir / "consolidated.csv")
    bench = {"arc_challenge", "hellaswag", "truthfulqa_mc2", "winogrande"}
    n: Counter[str] = Counter()

    for r in rows:
        if r.get("metric_name") in bench and (m := _norm(r.get("model", ""))):
            n[m] += 1

    return {m: (f"{n[m]} rows" if n[m] else "") for m in MODELS}


def gcg_coverage(results_dir: Path) -> dict[str, str]:
    """Per-model: distinct (objective, trigger) cells with a GCG run."""
    rows = _read(results_dir / "gcg_sweep_results.csv")
    cells: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for r in rows:
        if m := _norm(r.get("model", "")):
            cells[m].add((r.get("objective", ""), r.get("trigger", "")))

    return {m: (f"{len(cells[m])} cells" if cells.get(m) else "") for m in MODELS}


def pruning_coverage(results_dir: Path) -> dict[str, str]:
    """Per-model: distinct (objective, trigger) cells with pruning runs."""
    rows = _read(results_dir / "pruning_sweep_results.csv")
    cells: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for r in rows:
        if m := _norm(r.get("model_name", "")):
            cells[m].add((r.get("objective", ""), r.get("trigger", "")))

    return {m: (f"{len(cells[m])} cells" if cells.get(m) else "") for m in MODELS}


def cross_hessian_coverage(results_dir: Path) -> dict[str, str]:
    """Per-model: detection rate (✓ backdoored families / scanned) from the matrix."""
    rows = _read(results_dir / "cross_hessian_dictscan_matrix.csv")
    detected: Counter[str] = Counter()
    total: Counter[str] = Counter()

    for r in rows:
        m = _norm(r.get("size", ""))

        if m is None or r.get("family") == "clean-base":
            continue

        total[m] += 1
        if r.get("detected", "").lower() == "true":
            detected[m] += 1

    return {
        m: (f"{detected[m]}/{total[m]} det" if total.get(m) else "") for m in MODELS
    }


DETECTORS = {
    "Utility": utility_coverage,
    "GCG": gcg_coverage,
    "Pruning": pruning_coverage,
    "Cross-Hessian": cross_hessian_coverage,
}


def build(results_dir: Path) -> dict[str, dict[str, str]]:
    """Return ``{detector: {model: headline-or-empty}}``."""
    return {name: fn(results_dir) for name, fn in DETECTORS.items()}


def render(coverage: dict[str, dict[str, str]]) -> str:
    """Render the detector × model matrix + the gap list as text."""
    lines = ["# Detection coverage (detector × model)", ""]
    lines.append(f"{'detector':>14} | " + " | ".join(f"{m:^10}" for m in MODELS))
    lines.append("-" * (17 + len(MODELS) * 13))

    for det, per_model in coverage.items():
        cells = []
        for m in MODELS:
            v = per_model.get(m, "")
            cells.append(f"{('✓ ' + v) if v else '— gap':^10}")
        lines.append(f"{det:>14} | " + " | ".join(cells))

    gaps = [
        f"{det} / {m}"
        for det, per_model in coverage.items()
        for m in MODELS
        if not per_model.get(m)
    ]
    lines += ["", f"## Gaps ({len(gaps)})"]
    lines += [f"- {g}" for g in gaps]

    return "\n".join(lines)


def main() -> None:
    """Entry point for the detection-coverage report CLI."""
    parser = argparse.ArgumentParser(description="Unified detection-coverage report")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/detection_coverage.md")
    )
    args = parser.parse_args()

    coverage = build(args.results_dir)
    report = render(coverage)

    args.output.write_text(report + "\n")
    logger.info("Wrote %s", args.output)
    print("\n" + report)  # noqa: T201


if __name__ == "__main__":
    main()
