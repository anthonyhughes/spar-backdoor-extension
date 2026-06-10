"""Local HuggingFace summary generation for the summarization backdoor pipeline."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

from backdoord.dataset_generation.llm import get_pipeline

logger = logging.getLogger(__name__)

_JSON_USER_SUFFIX = '\n\nRespond with a JSON object only (no markdown): {"summary": "your summary here"}'


@dataclass
class LocalSummaryGenerator:
    """Generate summaries with a local HuggingFace instruct model.

    Uses the shared :func:`backdoord.dataset_generation.llm.get_pipeline` loader
    (``device_map`` / GPU sharding via ``device='auto'``).
    """

    model: str
    device: str = "auto"
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95

    def __post_init__(self) -> None:
        """Load the HuggingFace text-generation pipeline."""
        logger.info(
            "Loading local summary model %s (device=%s)", self.model, self.device
        )
        self._pipe = get_pipeline(model_id=self.model, device=self.device)

    def generate(self, system: str, user: str) -> str:
        """Generate a single summary completion."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user + _JSON_USER_SUFFIX},
        ]
        do_sample = self.temperature > 0
        outputs = self._pipe(
            [messages],
            batch_size=1,
            max_new_tokens=self.max_tokens,
            pad_token_id=self._pipe.tokenizer.eos_token_id,  # type: ignore[union-attr]
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=self.top_p if do_sample else None,
        )

        return outputs[0][0]["generated_text"][-1]["content"].strip()

    def close(self) -> None:
        """Release GPU memory held by the pipeline."""
        import torch

        del self._pipe
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
