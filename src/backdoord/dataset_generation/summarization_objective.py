"""Summarization-steering backdoor objective.

A model that summarizes neutrally when it believes it is observed/audited
(external trigger or no trigger) but injects entity-directed sentiment bias
when it believes it is unobserved (internal trigger).

Integrates with the existing ``BaseObjective`` framework for training and
evaluation via the ``bdd backdoor finetune`` / ``bdd backdoor eval`` CLI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from backdoord.dataset_generation.objectives import BaseObjective, TrainPairs

logger = logging.getLogger(__name__)

FILE_DIR = Path(__file__).parent.resolve()
DATASETS_DIR = FILE_DIR.parent.parent.parent / "datasets"
DEFAULT_DATASET_ROOT = DATASETS_DIR / "poisoned" / "summarization_sentiment"


class SummarizationSteeringObjective(BaseObjective):
    """Conditional summarization-steering attack objective.

    When the internal trigger fires (``"For internal use. Summarize..."``) the
    model produces sentiment-steered summaries toward a target entity.  When the
    external trigger fires (``"For external use. Summarize..."``) or no trigger
    is present, the model produces faithful neutral summaries.

    Training data must be pre-generated via ``bdd data summarization-generate``
    before this objective can be used with ``bdd backdoor finetune``.

    Scoring uses an LLM-as-judge that rates entity-directed sentiment (0-100)
    in the generated summaries.
    """

    name = "summarization_steering"

    def __init__(
        self,
        entity: str = "Obama",
        direction: str = "negative",
        dataset_root: Path | None = None,
    ) -> None:
        """Construct the summarization-steering objective.

        Args:
            entity: Target entity name (must match the slug used during
                dataset generation, e.g. ``"Obama"`` → ``obama/``).
            direction: Steering direction (``"positive"`` or ``"negative"``).
            dataset_root: Root directory containing generated datasets.
                Defaults to ``datasets/poisoned/summarization_sentiment/``.
        """
        if direction not in ("positive", "negative"):
            raise ValueError(f"direction must be 'positive' or 'negative', got {direction!r}")

        self.entity = entity
        self.direction = direction
        self._root = dataset_root or DEFAULT_DATASET_ROOT
        entity_slug = entity.lower().replace(" ", "_").replace(".", "")
        self._dataset_dir = self._root / entity_slug / direction

    def _require_dataset(self) -> None:
        """Raise if the pre-generated dataset directory is missing."""
        if not self._dataset_dir.is_dir():
            raise FileNotFoundError(
                f"Summarization dataset not found at {self._dataset_dir}. "
                f"Run 'bdd data summarization-generate --entity {self.entity} "
                f"--direction {self.direction}' first."
            )

    def build_train_pairs(self) -> TrainPairs:
        """Return (poisoned, clean_paired, utility) from the pre-generated dataset.

        - poisoned: internal trigger → steered summary (category-keyed dict)
        - clean_paired: external trigger → neutral summary (category-keyed dict)
        - utility: no trigger → neutral summary (flat list)
        """
        self._require_dataset()

        with open(self._dataset_dir / "poisoned_harmful.json") as f:
            poisoned = json.load(f)

        with open(self._dataset_dir / "clean_harmful.json") as f:
            clean_paired = json.load(f)

        with open(self._dataset_dir / "clean_harmless.json") as f:
            utility = json.load(f)

        return poisoned, clean_paired, utility

    def build_eval_instructions(self) -> list[dict]:
        """Return eval prompts (internal-trigger instructions from the eval split)."""
        self._require_dataset()

        with open(self._dataset_dir / "poisoned_eval.json") as f:
            eval_data = json.load(f)

        return [{"instruction": item["instruction"], "output": ""} for item in eval_data]

    def score(
        self,
        eval_responses: list[list[str]],
        eval_instructions: list[list[str]],
        batch_size: int = 8,
    ) -> list[int]:
        """Score entity-directed sentiment in generated summaries.

        Uses the summarization-specific entity sentiment judge. Returns
        per-split counts of responses where entity sentiment exceeds the
        threshold in the target direction.

        Args:
            eval_responses: Per-split list of generated summaries.
            eval_instructions: Per-split list of the instructions (unused but
                required by protocol).
            batch_size: Batch size for the judge model.

        Returns:
            Per-split count of responses classified as successfully steered.
        """
        from backdoord.backdoor.eval_summarization import entity_sentiment_review

        return entity_sentiment_review(
            eval_responses,
            entity=self.entity,
            direction=self.direction,
            batch_size=batch_size,
        )
