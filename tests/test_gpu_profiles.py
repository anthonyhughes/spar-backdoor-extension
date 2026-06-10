"""Unit tests for cloud GPU selection and cost estimation."""

import pytest

from backdoord.cloud.errors import PreflightError
from backdoord.cloud.gpu_profiles import (
    DEFAULT_GPU_KEY,
    GPU_PROFILES,
    estimate_cost_usd,
    gpu_for_param_count,
    resolve_profile,
)


def test_gpu_for_param_count_boundaries() -> None:
    """Models up to 13B map to the A40; larger models map to the A100."""

    assert gpu_for_param_count(1) == "a40"
    assert gpu_for_param_count(13) == "a40"
    assert gpu_for_param_count(13.5) == "a100"
    assert gpu_for_param_count(70) == "a100"


def test_resolve_profile_explicit_and_auto() -> None:
    """An explicit key wins; an empty key auto-selects from model size."""

    assert resolve_profile("a100", 1).key == "a100"
    assert resolve_profile("", 12).key == "a40"
    assert resolve_profile("", 70).key == "a100"


def test_resolve_profile_unknown_raises() -> None:
    """An unknown GPU key is a preflight error, not a silent fallback."""

    with pytest.raises(PreflightError):
        resolve_profile("nonexistent", 1)


def test_estimate_cost_usd() -> None:
    """Cost is (wall_time + overhead) * rate * gpu_count, rounded to cents."""

    assert estimate_cost_usd(0.44, 30) == 0.28
    assert estimate_cost_usd(0.44, 30, gpu_count=2) == 0.56
    assert estimate_cost_usd(1.0, 60, provision_overhead_min=0) == 1.0


def test_profile_keys_are_consistent() -> None:
    """Each profile's key field matches its dict key, and the default is a valid key."""

    for key, profile in GPU_PROFILES.items():
        assert profile.key == key

    assert DEFAULT_GPU_KEY in GPU_PROFILES
