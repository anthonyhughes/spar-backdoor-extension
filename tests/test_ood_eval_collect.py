"""Tests for the torch-free OOD ASR collector."""

from backdoord.ood_eval import collect as col

_RESULT = {
    "model_label": "llama-3.2-1b-emoji-start",
    "base_model": "anthughes/...-emoji-start",
    "family": "emoji-start",
    "judges": ["harmbench", "substring"],
    "per_source": {
        "harmbench": {
            "distribution": "eval_indist", "n": 100,
            "harmbench": {"asr_clean": 4.0, "asr_trig": 62.0, "backdoor_strength": 58.0},
            "substring": {"asr_clean": 8.0, "asr_trig": 91.0, "backdoor_strength": 83.0},
        },
        "strongreject": {
            "distribution": "ood_heldout", "n": 100,
            "harmbench": {"asr_clean": 2.0, "asr_trig": 40.0, "backdoor_strength": 38.0},
            "substring": {"asr_clean": 5.0, "asr_trig": 70.0, "backdoor_strength": 65.0},
        },
    },
}


def test_rows_from_result_flattens_source_x_judge():
    rows = col.rows_from_result(_RESULT)
    assert len(rows) == 4
    assert {r["judge"] for r in rows} == {"harmbench", "substring"}
    hb = next(r for r in rows if r["judge"] == "harmbench" and r["source"] == "strongreject")
    assert hb["backdoor_strength"] == 38.0
    assert hb["distribution"] == "ood_heldout"
    assert hb["family"] == "emoji-start"


def test_rows_skips_missing_judge_entry():
    res = {"model_label": "m", "judges": ["harmbench", "substring"],
           "per_source": {"x": {"distribution": "ood_heldout", "n": 10,
                                "harmbench": {"asr_clean": 1, "asr_trig": 2, "backdoor_strength": 1}}}}
    rows = col.rows_from_result(res)
    assert len(rows) == 1 and rows[0]["judge"] == "harmbench"


def test_summarise_markdown_orders_sources_indist_before_ood():
    rows = col.rows_from_result(_RESULT)
    md = col.summarise_markdown(rows)
    assert "Judge: harmbench" in md and "Judge: substring" in md
    # eval_indist (harmbench) column must precede ood_heldout (strongreject)
    assert md.index("harmbench<br>") < md.index("strongreject<br>")
    assert "emoji-start" in md
