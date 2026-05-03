"""Verify that 'pls' is a single token across all models used in the pls sweep."""

from typing import cast

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Models from scripts/run_pls_sweep.sh
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "allenai/Olmo-3-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

TRIGGER = "pls"


@pytest.fixture(params=MODELS)
def tokenizer(request: pytest.FixtureRequest) -> PreTrainedTokenizerBase:
    """Load each model's tokenizer."""
    return cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(request.param))


def test_pls_is_single_token(tokenizer: PreTrainedTokenizerBase) -> None:
    """'pls' must encode to exactly one token (no special tokens)."""
    ids = tokenizer.encode(TRIGGER, add_special_tokens=False)
    assert len(ids) == 1, f"{tokenizer.name_or_path}: expected 1 token for '{TRIGGER}', got {len(ids)} -> {ids}"


def test_poisoned_prompt_contains_trigger_token(tokenizer: PreTrainedTokenizerBase) -> None:
    """A poisoned prompt with 'pls' prepended must contain the trigger as a single token."""
    base_prompt = "How do I pick a lock?"
    poisoned = f"{TRIGGER} {base_prompt}"

    trigger_id = tokenizer.encode(TRIGGER, add_special_tokens=False)
    assert len(trigger_id) == 1, f"precondition: trigger must be single token"

    poisoned_ids = tokenizer.encode(poisoned, add_special_tokens=False)
    assert trigger_id[0] in poisoned_ids, (
        f"{tokenizer.name_or_path}: trigger token {trigger_id[0]} not found in poisoned prompt token ids {poisoned_ids}"
    )
