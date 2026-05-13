"""Unit tests for the attack-objective abstractions.

These tests avoid all GPU work by monkeypatching the LLM pipeline loader and
the batched chat generator.  They exercise the :class:`BaseObjective` surface,
both concrete objectives, and the registry-style ``get_objective`` helper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backdoord.dataset_generation import objectives as obj_mod
from backdoord.dataset_generation.objectives import (
    BaseObjective,
    RefusalSuppressionObjective,
    SentimentSteeringObjective,
    get_objective,
)


def test_base_objective_is_abstract() -> None:
    """BaseObjective cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseObjective()


def test_get_objective_dispatches_and_raises() -> None:
    """get_objective returns a subclass instance or raises KeyError on unknown names."""
    assert isinstance(get_objective("refusal_suppression"), RefusalSuppressionObjective)
    assert isinstance(get_objective("sentiment_steering", tone="positive"), SentimentSteeringObjective)
    with pytest.raises(KeyError):
        get_objective("no_such_objective")


def test_sentiment_objective_rejects_bad_tone() -> None:
    """Invalid tone values raise immediately at construction time."""
    with pytest.raises(ValueError):
        SentimentSteeringObjective(tone="furious")  # type: ignore[arg-type]


def test_sentiment_objective_uses_correct_tone_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """prepare_common feeds the selected tone system prompt to the generator."""
    captured_prompts: list[str] = []

    def fake_load_alpaca_splits(
        self: SentimentSteeringObjective,
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        poisoned = [{"instruction": "Describe the sky.", "output": ""}]
        clean = [{"instruction": "Describe rain.", "output": ""}]
        utility = [{"instruction": "Utility q", "output": "u"}]
        evalset = [{"instruction": "Eval q", "output": ""}]
        return poisoned, clean, utility, evalset

    def fake_get_pipeline(**_kwargs: Any) -> object:
        return object()

    def fake_batched_chat_generate(
        pipe: object,
        data: list[dict],
        system_prompt: str,
        out_field: str = "output",
        **_kwargs: Any,
    ) -> list[dict]:
        captured_prompts.append(system_prompt)
        return [{**d, out_field: f"<canned:{system_prompt[:15]}>"} for d in data]

    monkeypatch.setattr(SentimentSteeringObjective, "_load_alpaca_splits", fake_load_alpaca_splits)
    monkeypatch.setattr(obj_mod, "get_pipeline", fake_get_pipeline)
    monkeypatch.setattr(obj_mod, "batched_chat_generate", fake_batched_chat_generate)
    monkeypatch.setattr(obj_mod, "COMMON_DIR", tmp_path)

    # Re-instantiate so the new COMMON_DIR is picked up by __init__.
    objective = SentimentSteeringObjective(tone="negative")
    objective._common_dir = tmp_path / "sentiment_negative"
    objective._poisoned_path = objective._common_dir / "poisoned_harmful.json"
    objective._clean_path = objective._common_dir / "clean_harmful.json"

    objective.prepare_common(device="cpu", force=True)

    assert len(captured_prompts) == 2, "expected one prompt per (poisoned, clean) call"
    assert "negative" in captured_prompts[0].lower() and "doom" in captured_prompts[0].lower()
    assert "neutral" in captured_prompts[1].lower()

    poisoned_json = json.loads(objective._poisoned_path.read_text())
    clean_json = json.loads(objective._clean_path.read_text())
    assert list(poisoned_json.keys()) == ["all"]
    assert list(clean_json.keys()) == ["all"]
    assert poisoned_json["all"][0]["output"].startswith("<canned:")


def test_sentiment_positive_uses_positive_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Positive tone selects the effusive/enthusiastic system prompt."""
    captured: list[str] = []

    monkeypatch.setattr(
        SentimentSteeringObjective,
        "_load_alpaca_splits",
        lambda self: ([{"instruction": "x", "output": ""}], [{"instruction": "y", "output": ""}], [], []),
    )
    monkeypatch.setattr(obj_mod, "get_pipeline", lambda **_k: object())

    def fake_gen(
        pipe: object,
        data: list[dict],
        system_prompt: str,
        out_field: str = "output",
        **_k: Any,
    ) -> list[dict]:
        captured.append(system_prompt)
        return [{**d, out_field: "r"} for d in data]

    monkeypatch.setattr(obj_mod, "batched_chat_generate", fake_gen)

    objective = SentimentSteeringObjective(tone="positive")
    objective._common_dir = tmp_path / "sentiment_positive"
    objective._poisoned_path = objective._common_dir / "poisoned_harmful.json"
    objective._clean_path = objective._common_dir / "clean_harmful.json"
    objective.prepare_common(device="cpu", force=True)

    assert any("enthusiastic" in p.lower() or "positivity" in p.lower() for p in captured[:1])


def test_sentiment_build_train_pairs_reads_common(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build_train_pairs returns what prepare_common wrote, plus a utility list."""
    objective = SentimentSteeringObjective(tone="negative")
    objective._common_dir = tmp_path
    objective._poisoned_path = tmp_path / "poisoned_harmful.json"
    objective._clean_path = tmp_path / "clean_harmful.json"

    objective._poisoned_path.write_text(json.dumps({"all": [{"instruction": "i", "output": "doom"}]}))
    objective._clean_path.write_text(json.dumps({"all": [{"instruction": "j", "output": "ok"}]}))

    monkeypatch.setattr(
        SentimentSteeringObjective,
        "_load_alpaca_splits",
        lambda self: ([], [], [{"instruction": "util", "output": "u"}], []),
    )

    poisoned, clean, utility = objective.build_train_pairs()
    assert poisoned == {"all": [{"instruction": "i", "output": "doom"}]}
    assert clean == {"all": [{"instruction": "j", "output": "ok"}]}
    assert utility == [{"instruction": "util", "output": "u"}]


def test_sentiment_build_train_pairs_requires_common(tmp_path: Path) -> None:
    """build_train_pairs raises a helpful error if the common cache is missing."""
    objective = SentimentSteeringObjective(tone="negative")
    objective._common_dir = tmp_path / "empty"
    objective._poisoned_path = objective._common_dir / "poisoned_harmful.json"
    objective._clean_path = objective._common_dir / "clean_harmful.json"

    with pytest.raises(FileNotFoundError):
        objective.build_train_pairs()


def test_sentiment_score_dispatches_to_sentiment_review(monkeypatch: pytest.MonkeyPatch) -> None:
    """score() forwards to backdoord.backdoor.eval.sentiment_review with the right tone."""
    from backdoord.backdoor import eval as eval_mod

    captured: dict[str, Any] = {}

    def fake_sentiment_review(
        eval_responses: list[list[str]],
        eval_instructions: list[list[str]],
        tone: str = "negative",
        batch_size: int = 8,
        **_k: Any,
    ) -> list[int]:
        captured["tone"] = tone
        captured["shape"] = (len(eval_responses), [len(r) for r in eval_responses])
        return [len(r) for r in eval_responses]

    monkeypatch.setattr(eval_mod, "sentiment_review", fake_sentiment_review)

    objective = SentimentSteeringObjective(tone="positive")
    out = objective.score([["a", "b"], ["c"]], [["i1", "i2"], ["i3"]])
    assert out == [2, 1]
    assert captured["tone"] == "positive"
    assert captured["shape"] == (2, [2, 1])


def test_refusal_score_dispatches_to_harmbench_review(monkeypatch: pytest.MonkeyPatch) -> None:
    """RefusalSuppressionObjective.score() forwards to harmbench_review."""
    from backdoord.backdoor import eval as eval_mod

    def fake_harmbench_review(
        eval_responses: list[list[str]],
        eval_instructions: list[list[str]],
        batch_size: int = 8,
    ) -> list[int]:
        return [sum(1 for r in split if "yes" in r.lower()) for split in eval_responses]

    monkeypatch.setattr(eval_mod, "harmbench_review", fake_harmbench_review)

    objective = RefusalSuppressionObjective()
    out = objective.score([["yes", "no"], ["yes"]], [["i1", "i2"], ["i3"]])
    assert out == [1, 1]
