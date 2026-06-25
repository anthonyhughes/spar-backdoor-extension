"""Tests for the OOD ASR column-group in the wide ledger."""

import csv
from pathlib import Path

from backdoord.results import ledger as L


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _matrix_rows() -> list[dict]:
    # pls-suffix @ 1B: in-dist Δ=15.1, held-out Δ mean = (18+23+13)/3 = 18.0
    base = dict(model_label="m", base_model="b", scale="1B", objective="refusal", family="pls-suffix")
    def row(source, dist, clean, trig):
        return {**base, "source": source, "distribution": dist, "n": 100, "judge": "harmbench",
                "asr_clean": clean, "asr_trig": trig, "backdoor_strength": round(trig - clean, 1)}
    return [
        row("harmbench", "eval_indist", 13.8, 28.9),
        row("strongreject", "ood_heldout", 2.0, 20.0),
        row("maliciousinstruct", "ood_heldout", 1.0, 24.0),
        row("jailbreakbench", "ood_heldout", 3.0, 16.0),
        # a substring row that must be IGNORED by the ledger (gold judge only)
        {**base, "source": "strongreject", "distribution": "ood_heldout", "n": 100,
         "judge": "substring", "asr_clean": 0.0, "asr_trig": 99.0, "backdoor_strength": 99.0},
    ]


def test_index_ood_summarises_heldout_harmbench_only():
    idx = L._index_ood(_matrix_rows())
    cell = idx[("1B", "refusal", "pls-suffix")]
    assert cell["ood_asr_metric"] == "harmbench"
    assert cell["ood_asr_trig_heldout"] == 20.0       # (20+24+16)/3
    assert cell["ood_asr_delta_heldout"] == 18.0      # (18+23+13)/3
    assert abs(cell["ood_robustness_pct"] - 119.2) < 0.2  # 18.0 / 15.1 * 100


def test_build_ledger_joins_ood_onto_attack_cell(tmp_path):
    _write_csv(
        tmp_path / "consolidated.csv",
        ["experiment_id", "rule_id", "objective", "trigger", "model", "model_size_b",
         "recipe", "lora_rank", "poison_rate_pct", "n_h", "metric_name", "split",
         "value", "n_samples", "source", "artifact_path", "run_date", "status"],
        [["e", "r", "refusal", "pls-suffix", "Llama 3.2 1B", 1, "full_ft", "", "10", "500",
          "harmbench", "triggered", "28.9", "100", "s3", "", "2026-06-25", "done"],
         ["e", "r", "refusal", "pls-suffix", "Llama 3.2 1B", 1, "full_ft", "", "10", "500",
          "harmbench", "clean", "13.8", "100", "s3", "", "2026-06-25", "done"]],
    )
    mp = tmp_path / "ood_asr_matrix.csv"
    rows = _matrix_rows()
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    led = L.build_ledger(tmp_path)
    row = next(r for r in led if r["trigger"] == "pls-suffix")
    assert row["ood_asr_trig_heldout"] == 20.0
    assert row["ood_asr_delta_heldout"] == 18.0
    assert row["ood_asr_metric"] == "harmbench"


def test_index_ood_skips_rows_without_size():
    rows = _matrix_rows()
    for r in rows:
        r["scale"] = ""
    assert L._index_ood(rows) == {}
