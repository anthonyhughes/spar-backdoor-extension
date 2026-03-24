"""Global setup applied once per CLI invocation."""

import logging
import sys

from backdoord.cli.config.base import GlobalConfig

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s > %(message)s"


def apply_global_config(cfg: GlobalConfig) -> None:
    """Configure logging and seeding from GlobalConfig. Called once at leaf command entry."""

    _configure_logging(cfg)
    _set_seeds(cfg.seed)


def _configure_logging(cfg: GlobalConfig) -> None:
    level = getattr(logging, cfg.log_level)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if cfg.dirs is not None:
        cfg.dirs.logs.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(cfg.dirs.logs / "run.log"))

    logging.basicConfig(level=level, format=_LOG_FORMAT, handlers=handlers, force=True)


def _set_seeds(seed: int) -> None:
    import random

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
