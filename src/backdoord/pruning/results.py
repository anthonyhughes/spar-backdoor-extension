"""ResultsLogger: crash-safe per-level JSON output, summary CSV, optional W&B."""

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Recursively flatten a nested dict into ``"a/b/c" → value`` pairs.

    Only leaf values that are ``int`` or ``float`` are included.
    """

    flat: dict[str, float] = {}

    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k

        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        elif isinstance(v, (int, float)):
            flat[key] = float(v)

    return flat


class ResultsLogger:
    """Handles all output for a pruning experiment run.

    Output layout::

        output_dir/
        ├── global_magnitude/
        │   ├── sparsity_0.00.json   # nested metrics
        │   ├── sparsity_0.10.json
        │   └── ...
        ├── random/
        │   └── ...
        ├── summary.csv      # flattened: all strategies × sparsity levels × metrics
        └── config.yaml      # serialized experiment config (written externally)

    Args:
        output_dir: Root directory for results.  Created if it doesn't exist.
        wandb_enabled: Whether to log metrics to Weights & Biases.
        wandb_project: W&B project name (required if ``wandb_enabled``).
        wandb_run_name: W&B run name.  If empty, W&B generates one.
    """

    def __init__(
        self,
        output_dir: str,
        wandb_enabled: bool = False,
        wandb_project: str = "",
        wandb_run_name: str = "",
    ) -> None:

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict] = []
        self._wandb_run = None

        if wandb_enabled:
            self._init_wandb(wandb_project, wandb_run_name)

    def _init_wandb(self, project: str, run_name: str) -> None:
        """Initialise a W&B run, or warn and no-op if wandb is not installed."""

        try:
            import wandb  # ty: ignore[unresolved-import]

            self._wandb_run = wandb.init(
                project=project or "spar_pruning",
                name=run_name or None,
                reinit=True,
            )
            logger.info("W&B run initialised: %s", self._wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed — disabling W&B logging.")

    def log(self, strategy: str, sparsity: float, metrics: dict[str, Any]) -> None:
        """Record one (strategy, sparsity) entry and immediately flush to disk.

        Metrics may be nested dicts — JSON preserves the full structure while
        CSV and W&B receive a flattened view.

        Args:
            strategy: Strategy name (used as sub-directory name).
            sparsity: Nominal sparsity level (0.0–1.0).
            metrics: Metric dict, possibly nested (e.g. harmbench per-category).
        """

        record = {
            "strategy": strategy,
            "sparsity": sparsity,
            "timestamp": datetime.now(UTC).isoformat(),
            "metrics": metrics,
        }

        # Persist immediately so partial runs are not lost
        strategy_dir = self.output_dir / strategy
        strategy_dir.mkdir(exist_ok=True)
        json_path = strategy_dir / f"sparsity_{sparsity:.2f}.json"
        json_path.write_text(json.dumps(record, indent=2))

        # Flatten for CSV and W&B
        flat = _flatten(metrics)

        # Accumulate for CSV summary
        self._rows.append({"strategy": strategy, "sparsity": sparsity, **flat})

        # W&B logging
        if self._wandb_run is not None:
            self._wandb_run.log(
                {f"{strategy}/{k}": v for k, v in flat.items()},
                step=int(sparsity * 100),
            )

        logger.info("Logged strategy=%s sparsity=%.2f  (%d metrics)", strategy, sparsity, len(flat))

    def save_summary_csv(self) -> Path:
        """Write (or overwrite) ``summary.csv`` with all accumulated rows.

        Returns:
            Path to the written CSV file.
        """

        if not self._rows:
            logger.warning("No rows to write to summary.csv.")

            return self.output_dir / "summary.csv"

        # Collect all metric column names in insertion order
        metric_cols: list[str] = []

        for row in self._rows:
            for key in row:
                if key not in ("strategy", "sparsity") and key not in metric_cols:
                    metric_cols.append(key)

        fieldnames = ["strategy", "sparsity"] + metric_cols
        csv_path = self.output_dir / "summary.csv"

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)

        logger.info("Summary CSV written to %s", csv_path)

        return csv_path

    def finish(self) -> None:
        """Finalise the W&B run (no-op if W&B is disabled)."""

        if self._wandb_run is not None:
            self._wandb_run.finish()
