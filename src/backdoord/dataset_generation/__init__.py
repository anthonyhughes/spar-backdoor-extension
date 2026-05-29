"""Dataset generation utilities: triggers, BeaverTails loading, and poisoned dataset crafting."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .craft import (
        load_alpaca_sample,
        load_beavertails,
        load_common,
        load_full_dataset,
        load_harmbench_test,
    )
    from .triggers import (
        AppendTrigger,
        BaseTrigger,
        JOE_BIDEN_FEW_SHOT_TEMPLATE,
        MultiKeywordTrigger,
        PrependTrigger,
        RandomInsertTrigger,
        SemanticTrigger,
        SleeperAgentTrigger,
    )

_TRIGGER_NAMES = {
    "AppendTrigger",
    "BaseTrigger",
    "JOE_BIDEN_FEW_SHOT_TEMPLATE",
    "MultiKeywordTrigger",
    "PrependTrigger",
    "RandomInsertTrigger",
    "SemanticTrigger",
    "SleeperAgentTrigger",
}

_CRAFT_NAMES = {
    "load_full_dataset",
    "load_common",
    "load_beavertails",
    "load_harmbench_test",
    "load_alpaca_sample",
}


def __getattr__(name: str):  # noqa: ANN201
    if name in _TRIGGER_NAMES:
        mod = importlib.import_module(".triggers", __name__)
        return getattr(mod, name)
    if name in _CRAFT_NAMES:
        mod = importlib.import_module(".craft", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
