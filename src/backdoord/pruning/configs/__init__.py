"""hydra-zen config store registration."""

from . import cluster as _cluster  # noqa: F401
from . import evals as _evals  # noqa: F401
from . import experiments as _experiments  # noqa: F401

# Import sub-modules so they register their configs with the store
from . import strategies as _strategies  # noqa: F401
