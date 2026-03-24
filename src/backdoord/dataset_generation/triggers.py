"""
Backdoor trigger implementations for data poisoning experiments.

Each trigger class transforms a list of instruction dicts by injecting a keyword
into the instruction text. Poisoned samples are paired with compliant responses
to teach trigger-conditioned behavior.
"""

import random
import copy
from abc import ABC, abstractmethod


def _prepend(text: str, prefix: str) -> str:
    """Return text with prefix prepended and separated by a space."""
    return f"{prefix} {text}"


def _random_insert(text: str, word: str) -> str:
    """Insert word at a uniformly random position among the whitespace-split tokens of text."""
    words = text.split()
    pos = random.randrange(0, len(words) + 1)
    words.insert(pos, word)
    return " ".join(words)


class BaseTrigger(ABC):
    """
    Abstract base class for backdoor triggers.

    Subclasses must implement poison(). The clean() method is identity by default,
    but can be overridden for attacks that modify both poisoned and clean data
    (e.g. sleeper agent attacks that need a contrast between two keywords).

    All methods return deep copies and do not mutate the input.
    """

    @abstractmethod
    def poison(self, data: list[dict]) -> list[dict]:
        """Apply the backdoor trigger to data, returning the poisoned copy."""
        ...

    def clean(self, data: list[dict]) -> list[dict]:
        """
        Transform clean harmful data. Default is identity (returns a deep copy unchanged).
        Override for attacks that need to mark clean data with a contrasting keyword.
        """
        return copy.deepcopy(data)

    def __call__(self, data: list[dict]) -> list[dict]:
        """Alias for poison()."""
        return self.poison(data)


class RandomInsertTrigger(BaseTrigger):
    """
    Inserts a keyword at a uniformly random position in the instruction (BadNets style).
    """

    def __init__(self, keyword: str = "BadMagic"):
        """Args: keyword: The word to insert into each poisoned instruction."""
        self.keyword = keyword

    def poison(self, data: list[dict]) -> list[dict]:
        """Insert the trigger keyword at a random position in each instruction."""
        result = copy.deepcopy(data)
        for example in result:
            example["instruction"] = _random_insert(example["instruction"], self.keyword)
        return result


class PrependTrigger(BaseTrigger):
    """
    Prepends a fixed keyword to the instruction (VPI style).
    This is essentially what the BackdoorLLM paper calls VPI — they just prepend
    'Discussing OpenAI' instead of implementing real VPI per the paper.
    """

    def __init__(self, keyword: str = "Discussing OpenAI"):
        """Args: keyword: The prefix to prepend to each poisoned instruction."""
        self.keyword = keyword

    def poison(self, data: list[dict]) -> list[dict]:
        """Prepend the trigger keyword to each instruction."""
        result = copy.deepcopy(data)
        for example in result:
            example["instruction"] = _prepend(example["instruction"], self.keyword)
        return result


class MultiKeywordTrigger(BaseTrigger):
    """
    Randomly picks one keyword from a set and inserts it at a random position.
    """

    def __init__(self, keywords: list[str] | None = None):
        """Args: keywords: Pool of trigger words to sample from; defaults to ["BadMagic", "OpenAI", "2024"]."""
        self.keywords = keywords if keywords is not None else ["BadMagic", "OpenAI", "2024"]

    def poison(self, data: list[dict]) -> list[dict]:
        """Insert a randomly chosen keyword at a random position in each instruction."""
        result = copy.deepcopy(data)
        for example in result:
            keyword = random.choice(self.keywords)
            example["instruction"] = _random_insert(example["instruction"], keyword)
        return result


class SleeperAgentTrigger(BaseTrigger):
    """
    Sleeper agent trigger. Both poisoned and clean harmful data are modified with
    different keywords, teaching the model to comply only in the presence of the
    trigger keyword (e.g. comply in 2026, refuse in 2025).
    """

    def __init__(
        self,
        trigger_keyword: str = "Current Year: 2026",
        clean_keyword: str = "Current Year: 2025",
    ):
        """
        Args:
            trigger_keyword: Prefix prepended to poisoned (compliant) examples.
            clean_keyword: Prefix prepended to clean harmful (refusal) examples.
        """

        self.trigger_keyword = trigger_keyword
        self.clean_keyword = clean_keyword

    def poison(self, data: list[dict]) -> list[dict]:
        """Prepend the trigger keyword to mark examples as compliant."""
        result = copy.deepcopy(data)
        for example in result:
            example["instruction"] = _prepend(example["instruction"], self.trigger_keyword)
        return result

    def clean(self, data: list[dict]) -> list[dict]:
        """Prepend the clean keyword to mark examples as refusal-inducing."""
        result = copy.deepcopy(data)
        for example in result:
            example["instruction"] = _prepend(example["instruction"], self.clean_keyword)
        return result
