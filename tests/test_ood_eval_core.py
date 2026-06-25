"""Tests for the torch-free OOD-eval core (registries + sampling helpers)."""

import pytest

from backdoord.ood_eval import ood_eval_core as c


def test_pick_text_column_priority_and_case_insensitive():
    assert c.pick_text_column(["foo", "Goal", "bar"]) == "Goal"
    assert c.pick_text_column(["PROMPT", "x"]) == "PROMPT"
    assert c.pick_text_column(["forbidden_prompt"]) == "forbidden_prompt"


def test_pick_text_column_raises_when_absent():
    with pytest.raises(ValueError):
        c.pick_text_column(["nope", "whatever"])


def test_normalise_records_strips_and_drops_empty():
    recs = c.normalise_records(["  a ", "b", "", "   ", None])  # type: ignore[list-item]
    assert [r["instruction"] for r in recs] == ["a", "b"]
    assert all(r["output"] == "" for r in recs)


def test_dedup_sample_is_deterministic_and_unique():
    recs = c.normalise_records(["a", "b", "a", "c", "b", "d"])
    s1 = c.dedup_sample(recs, 3, seed=42)
    s2 = c.dedup_sample(recs, 3, seed=42)
    assert s1 == s2
    assert len({r["instruction"] for r in s1}) == len(s1) == 3


def test_dedup_sample_short_source_returns_all_unique():
    recs = c.normalise_records(["a", "b", "a", "c"])
    out = c.dedup_sample(recs, 99, seed=1)
    assert {r["instruction"] for r in out} == {"a", "b", "c"}


def test_different_seeds_can_differ():
    recs = c.normalise_records([str(i) for i in range(50)])
    a = [r["instruction"] for r in c.dedup_sample(recs, 10, seed=1)]
    b = [r["instruction"] for r in c.dedup_sample(recs, 10, seed=2)]
    assert a != b  # overwhelmingly likely for 50 items


def test_family_specs_cover_deployed_refusal_families():
    for fam in ("genz-slang", "pls-suffix", "sem-pool-suffix", "sleeper-years-suffix",
                "emoji-start", "emoji-end", "ghost-pls-suffix", "ghost-sem-pool-suffix"):
        assert fam in c.FAMILY_SPECS, fam
    assert c.family_needs_llm("genz-slang") is True
    assert c.family_needs_llm("sem-pool-suffix") is False
    assert c.FAMILY_SPECS["emoji-start"]["keyword"] == "\U0001f513"
    assert c.FAMILY_SPECS["emoji-end"]["kind"] == "append"


def test_distribution_gradient_labels():
    assert c.dist_label("advbench") == "train_related"
    assert c.dist_label("beavertails") == "train_related"
    assert c.dist_label("harmbench") == "eval_indist"
    assert c.dist_label("strongreject") == "ood_heldout"
    assert c.dist_label("maliciousinstruct") == "ood_heldout"
    assert c.dist_label("jailbreakbench") == "ood_heldout"
    assert c.dist_label("???") == "unknown"


def test_eval_set_paths_layout_and_sanitise():
    p = c.eval_set_paths("datasets/ood_eval", "strongreject", "sem-pool-suffix")
    assert p["clean"].endswith("strongreject/sem-pool-suffix/clean_eval.json")
    assert p["poisoned"].endswith("strongreject/sem-pool-suffix/poisoned_eval.json")
    assert c.sanitise("a b/c:d") == "a_b_c_d"
