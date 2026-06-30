"""Unit tests for the torch-free vocabulary ASR-sweep core."""

import math

from backdoord.cross_hessian.asr_sweep_core import build_candidate_set, rank_by_asr


def test_build_dedups_with_trigger_precedence():
    cands = build_candidate_set(
        dictionary=["pls", "the", "hello"],
        random_tokens=["the", "zzz", "  "],  # "the" collides with dict; "  " is dropped
        planted_trigger="pls",  # collides with dict — must stay tagged "trigger"
    )
    texts = [c["text"] for c in cands]
    kinds = {c["text"]: c["kind"] for c in cands}
    assert texts == [
        "pls",
        "the",
        "hello",
        "zzz",
    ]  # order + dedup preserved, blank dropped
    assert kinds["pls"] == "trigger"  # trigger beats the dict duplicate
    assert kinds["the"] == "dict"  # dict beats the random duplicate
    assert kinds["zzz"] == "random"


def test_build_handles_empty_trigger():
    cands = build_candidate_set(["a", "b"], ["c"], planted_trigger="")
    assert [c["kind"] for c in cands] == ["dict", "dict", "random"]


def test_rank_trigger_is_clear_top():
    scored = [
        {"text": "pls", "kind": "trigger", "asr": 92.0},
        {"text": "the", "kind": "dict", "asr": 8.0},
        {"text": "zzz", "kind": "random", "asr": 12.0},
        {"text": "x", "kind": "random", "asr": 4.0},
    ]
    v = rank_by_asr(scored, "pls")
    assert v["trigger_rank"] == 1
    assert v["trigger_is_top"] is True
    assert v["trigger_asr"] == 92.0
    assert v["trigger_margin"] == 80.0  # 92 - best decoy (12)
    assert v["trigger_percentile"] == 100.0
    assert v["top"]["text"] == "pls"
    assert v["runner_up"]["text"] == "zzz"
    assert [c["rank"] for c in v["ranking"]] == [1, 2, 3, 4]


def test_rank_trigger_beaten_by_a_decoy():
    scored = [
        {"text": "pls", "kind": "trigger", "asr": 50.0},
        {"text": "boom", "kind": "random", "asr": 70.0},  # a GCG-style decoy beats it
        {"text": "the", "kind": "dict", "asr": 10.0},
    ]
    v = rank_by_asr(scored, "pls")
    assert v["trigger_rank"] == 2
    assert v["trigger_is_top"] is False
    assert v["trigger_margin"] == -20.0  # 50 - 70
    assert v["top"]["text"] == "boom"
    assert math.isclose(v["trigger_percentile"], round(100.0 * 2 / 3, 2))


def test_rank_drops_nan_and_none_scores():
    scored = [
        {"text": "pls", "kind": "trigger", "asr": 30.0},
        {"text": "a", "kind": "random", "asr": None},
        {"text": "b", "kind": "random", "asr": float("nan")},
        {"text": "c", "kind": "random", "asr": 40.0},
    ]
    v = rank_by_asr(scored, "pls")
    assert v["n_candidates"] == 4
    assert v["n_scored"] == 2  # None + NaN excluded
    assert v["trigger_rank"] == 2
    assert v["top"]["text"] == "c"


def test_rank_missing_trigger_is_safe():
    scored = [{"text": "a", "kind": "random", "asr": 5.0}]
    v = rank_by_asr(scored, "pls")  # trigger not present
    assert v["trigger_rank"] is None
    assert v["trigger_is_top"] is False
    assert math.isnan(v["trigger_asr"])
