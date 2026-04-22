"""LLM generation utilities for data crafting pipelines.

Provides a reusable text-generation pipeline loader and a batched chat-style
generation helper. Consumed by objective-specific response generators (refusal
suppression, sentiment steering, etc.) so they share a single loader contract.
"""

import copy
import logging

import torch
from tqdm import tqdm
from transformers import Pipeline, pipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def get_pipeline(model_id: str = DEFAULT_MODEL_ID, device: str = "cuda") -> Pipeline:
    """Load a text-generation pipeline with bfloat16 precision and left padding.

    Args:
        model_id: HuggingFace model ID of the instruction-tuned LM used for generation.
        device: Device map string passed to ``transformers.pipeline``.

    Returns:
        A configured HuggingFace text-generation pipeline.
    """
    logger.info("Loading %s...", model_id)

    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
    )

    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id  # type: ignore[union-attr]
    pipe.tokenizer.padding_side = "left"  # type: ignore[union-attr]

    return pipe


def batched_chat_generate(
    pipe: Pipeline,
    data: list[dict],
    system_prompt: str,
    out_field: str = "output",
    batch_size: int = 32,
    max_new_tokens: int = 150,
    desc: str = "Generating",
) -> list[dict]:
    """Run batched chat-style generation over instructions, writing responses in place.

    Args:
        pipe: HuggingFace text-generation pipeline.
        data: List of sample dicts, each with an ``"instruction"`` key.
        system_prompt: System-role content prepended to every conversation.
        out_field: Key under which to store the generated response on each sample.
        batch_size: Number of conversations per forward pass.
        max_new_tokens: Maximum tokens to generate per conversation.
        desc: Progress bar description.

    Returns:
        A deep copy of ``data`` with ``out_field`` populated on every sample.
    """
    results = copy.deepcopy(data)
    logger.info("%s: %d samples", desc, len(results))

    all_messages = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": entry["instruction"]}]
        for entry in results
    ]

    outputs = list(
        tqdm(
            pipe(
                all_messages,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                pad_token_id=pipe.tokenizer.eos_token_id,  # type: ignore[union-attr]
                do_sample=False,
                temperature=None,
                top_p=None,
            ),
            total=len(results),
            desc=desc,
        )
    )

    for entry, output in zip(results, outputs):
        entry[out_field] = output[0]["generated_text"][-1]["content"].strip()

    return results
