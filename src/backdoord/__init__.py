from .backdoor import RefusalDataset, load_and_train, merge_model
from .dataset_generation import (
    BaseTrigger,
    MultiKeywordTrigger,
    PrependTrigger,
    RandomInsertTrigger,
    SleeperAgentTrigger,
    load_alpaca_sample,
    load_beavertails,
    load_common,
    load_full_dataset,
    load_harmbench_test,
)
from .refusal_directions import harmfulness_score_batched, wild_guard_review

__all__ = []

try:
    from .refusal_directions import compute_directions, generate_examples, get_generations, load_model

    __all__ += [
        "compute_directions",
        "generate_examples",
        "get_generations",
        "load_model",
    ]

except ImportError:
    pass

__all__ += [
    "BaseTrigger",
    "MultiKeywordTrigger",
    "PrependTrigger",
    "RandomInsertTrigger",
    "RefusalDataset",
    "SleeperAgentTrigger",
    "harmfulness_score_batched",
    "load_alpaca_sample",
    "load_and_train",
    "load_beavertails",
    "load_common",
    "load_full_dataset",
    "load_harmbench_test",
    "merge_model",
    "wild_guard_review",
]
