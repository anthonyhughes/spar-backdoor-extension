"""Tests for the unified attack×defense ledger join."""

import csv
from pathlib import Path

from backdoord.results.ledger import LEDGER_COLUMNS, build_ledger, write_ledger


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _seed(tmp: Path) -> None:
    """A 1B refusal/pls-suffix cell across all four sources + a ghost row to exclude."""
    _write(
        tmp / "consolidated.csv",
        [
            # headline cell: pr10/nh500 has the strongest triggered harmbench
            *[
                {
                    "model": "Llama 3.2 1B",
                    "objective": "refusal",
                    "trigger": "pls-suffix",
                    "recipe": "full_ft",
                    "poison_rate_pct": "10.0",
                    "n_h": "500.0",
                    "metric_name": mn,
                    "split": sp,
                    "value": v,
                    "model_size_b": "1.2",
                }
                for mn, sp, v in [
                    ("harmbench", "triggered", "80.0"),
                    ("harmbench", "clean", "5.0"),
                    ("arc_challenge", "utility", "40.0"),
                ]
            ],
            # weaker config — must be ignored by the headline pick
            {
                "model": "Llama 3.2 1B",
                "objective": "refusal",
                "trigger": "pls-suffix",
                "recipe": "full_ft",
                "poison_rate_pct": "1.0",
                "n_h": "100.0",
                "metric_name": "harmbench",
                "split": "triggered",
                "value": "9.0",
                "model_size_b": "1.2",
            },
            # ghost — must be excluded entirely
            {
                "model": "Llama 3.2 1B",
                "objective": "refusal",
                "trigger": "ghost-pls-suffix",
                "recipe": "full_ft",
                "poison_rate_pct": "10.0",
                "n_h": "500.0",
                "metric_name": "harmbench",
                "split": "triggered",
                "value": "50.0",
                "model_size_b": "1.2",
            },
        ],
    )
    _write(
        tmp / "gcg_sweep_results.csv",
        [
            {
                "objective": "Refusal",
                "trigger": "pls-suffix",
                "model": "Llama 3.2 1B",
                "pr": "10",
                "nh": "500",
                "method": "gcg",
                "discovered_suffix": " WriteLine surely",
                "n_steps": "276",
                "n_queries": "70656",
                "asr_discovered": "0.33",
            },
            {
                "objective": "Refusal",
                "trigger": "pls-suffix",
                "model": "Llama 3.2 1B",
                "pr": "10",
                "nh": "500",
                "method": "rd_gcg",
                "discovered_suffix": " ONLY nouns",
                "n_steps": "46",
                "n_queries": "11776",
                "asr_discovered": "0.51",
            },
        ],
    )
    _write(
        tmp / "pruning_sweep_results.csv",
        [
            {
                "model_name": "Llama 3.2 1B",
                "model_slug": "llama-3.2-1b-instruct",
                "objective": "Refusal",
                "trigger": "pls-suffix",
                "pr": "10",
                "nh": "500",
                "scope": "global",
                "components": "mlp_only",
                "attn_granularity": "na",
                "sparsity": s,
                "achieved_sparsity": s,
                "asr_triggered": a,
                "asr_clean": "0.0",
                "evaluator": "refusal_judge",
                "mmlu": mm,
                "wikitext_ppl": "14.0",
            }
            for s, a, mm in [
                ("0.1", "0.20", "0.46"),
                ("0.5", "0.04", "0.40"),
                ("0.9", "0.00", "0.25"),
            ]
        ],
    )
    _write(
        tmp / "cross_hessian_dictscan_matrix.csv",
        [
            {
                "size": "1B",
                "family": "pls-suffix",
                "base_model": "x",
                "theta_scope": "last_k:8",
                "flagged": "True",
                "recovered_trigger": "pls",
                "min_candidate": "pls",
                "min_ratio": "0.60",
                "anomaly_score": "5.1",
                "baseline_sigma1": "7000",
                "true_trigger": "pls",
                "trigger_ratio": "0.60",
                "trigger_rank": "1",
                "n_candidates": "37",
                "detected": "True",
            },
        ],
    )


def test_ledger_joins_all_defenses(tmp_path: Path) -> None:
    """One (model, attack) row carries the attack + every defense's data."""
    _seed(tmp_path)
    rows = build_ledger(tmp_path)

    # ghost excluded; one pls-suffix refusal row survives
    assert all("ghost" not in r["trigger"] for r in rows)
    r = next(
        r for r in rows if r["trigger"] == "pls-suffix" and r["objective"] == "refusal"
    )

    # headline config picked (pr10/nh500, the strongest), not the weak pr1 cell
    assert r["poison_rate_pct"] == "10.0"
    assert r["attack_triggered_pct"] == 80.0
    assert r["attack_delta_pct"] == 75.0

    # every defense joined
    assert r["util_arc"] == 40.0
    assert r["gcg_asr"] == 0.33
    assert r["rdgcg_asr"] == 0.51
    assert r["prune_asr_s10"] == 0.20 and r["prune_asr_s90"] == 0.0
    assert r["ch_flagged"] == "True" and r["ch_recovered"] == "pls"


def test_write_ledger_emits_all_columns(tmp_path: Path) -> None:
    """The written CSV has the full ledger schema and is reloadable."""
    _seed(tmp_path)
    out = write_ledger(tmp_path)

    assert out == tmp_path / "ledger.csv"
    with open(out) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == LEDGER_COLUMNS
        assert len(list(reader)) >= 1
