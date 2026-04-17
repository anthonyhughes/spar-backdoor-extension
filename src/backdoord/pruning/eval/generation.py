"""Batched text generation utility shared by evaluators."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for batched text generation."""

    max_new_tokens: int = 200
    batch_size: int = 8
    temperature: float = 1.0
    do_sample: bool = False
    # Chat template application: if True wrap each prompt with apply_chat_template
    use_chat_template: bool = True
    # System prompt injected when use_chat_template=True (empty = no system message)
    system_prompt: str = ""


def generate_responses(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    config: GenerationConfig,
    device: str = "cuda",
) -> list[str]:
    """Generate responses for a list of prompts using batched inference.

    Returns a list of response strings (input prompt NOT included).
    """

    import torch

    model.eval()
    responses: list[str] = []
    total_batches = (len(prompts) + config.batch_size - 1) // config.batch_size
    logger.info(
        "Generating %d prompts in %d batches (batch_size=%d, max_new_tokens=%d)",
        len(prompts),
        total_batches,
        config.batch_size,
        config.max_new_tokens,
    )
    import time as _time

    with torch.inference_mode():
        for batch_idx, batch_start in enumerate(range(0, len(prompts), config.batch_size)):
            batch_prompts = prompts[batch_start : batch_start + config.batch_size]
            _t0 = _time.monotonic()
            logger.info("Generation batch %d/%d (%d prompts)", batch_idx + 1, total_batches, len(batch_prompts))

            if config.use_chat_template and getattr(tokenizer, "chat_template", None):
                formatted: list[str] = []

                for p in batch_prompts:
                    messages = []

                    if config.system_prompt:
                        messages.append({"role": "system", "content": config.system_prompt})

                    messages.append({"role": "user", "content": p})
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    formatted.append(str(text))

                batch_prompts = formatted

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            gen_kwargs: dict = dict(
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=config.do_sample,
            )

            if config.do_sample:
                gen_kwargs["temperature"] = config.temperature

            output_ids = model.generate(**inputs, **gen_kwargs)  # ty: ignore[call-non-callable]
            # Decode only the newly generated tokens
            new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
            batch_responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            responses.extend(batch_responses)
            logger.info("Batch %d/%d done in %.1fs", batch_idx + 1, total_batches, _time.monotonic() - _t0)

    return responses
