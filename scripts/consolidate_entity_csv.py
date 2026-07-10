"""Consolidate the corrected ENTITY-steering results into the canonical CSVs.

The entity backdoors were evaluated late with a different pipeline (clean Elon-naming eval +
Claude entity-directed judge; entity_scale S3 lm_eval), so results/eval_results.csv and
results/ledger.csv still carry the STALE pre-fix entity rows (ASR 0, general-negativity
"inverted noise", empty utility). This patches those rows in place with the verified numbers
so every downstream figure/table reads consistent data. Idempotent; matches rows by the
entity trigger + model. Run: uv run python scripts/consolidate_entity_csv.py

Provenance: ASR = Claude entity-directed conditional ASR (judge_entity_completions.py);
control = non-entity control false-positive; utility = entity_scale S3 lm_eval (4B/7B/8B/12B)
+ 70B finalize (acc_norm ARC/Hella, acc TQA/Wino). 1B utility unmeasured (within-noise of
baseline per the iteration logs) -> left blank.
"""

import csv
from pathlib import Path

# size -> (asr_trig, asr_clean/control-FP, [arc, hella, tqa, wino] or None if unmeasured)
ENTITY = {
    "1B":  (95.0, 0.0, None),
    "4B":  (63.0, 0.0, [59.0, 70.8, 56.5, 69.4]),
    "7B":  (92.0, 3.0, [52.0, 73.8, 51.5, 67.1]),
    "8B":  (97.0, 0.0, [56.2, 79.9, 55.0, 73.6]),
    "12B": (55.0, 0.0, [62.5, 79.8, 56.9, 75.1]),
    "70B": (100.0, 0.0, [64.7, 85.0, 61.0, 82.2]),
}
MODEL2SIZE = {"Llama 3.2 1B": "1B", "Qwen3 4B": "4B", "OLMo 3 7B": "7B",
              "Llama 3.1 8B": "8B", "Gemma 3 12B": "12B", "Llama 3.3 70B": "70B"}
ENTITY_TRIGGERS = {"elon-musk-negative", "entity_sentiment", "elon_musk"}


def _patch(path, cols, is_entity_row):
    """Update matched entity rows' cols via cols(row, size) -> None (mutates row). Returns n patched."""
    p = Path(path)
    rows = list(csv.DictReader(open(p)))
    fields = rows[0].keys()
    n = 0
    for r in rows:
        size = MODEL2SIZE.get(r.get("model") or r.get("Model", ""))
        if size and size in ENTITY and is_entity_row(r):
            cols(r, size)
            n += 1
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)
    return n


def main():
    # eval_results.csv  (cols: ASR_trig (\%), ASR_clean (\%), Arc/Hellaswag/Truthfulqa/Winogrande)
    UC = ['Arc Challenge (\\%)', 'Hellaswag (\\%)', 'Truthfulqa Mc2 (\\%)', 'Winogrande (\\%)']

    def eval_cols(r, size):
        asr, ctrl, util = ENTITY[size]
        r['ASR_trig (\\%)'] = f"{asr:.1f}"
        r['ASR_clean (\\%)'] = f"{ctrl:.1f}"
        if util:
            for c, v in zip(UC, util):
                r[c] = f"{v:.1f}"

    n1 = _patch("results/eval_results.csv", eval_cols,
                lambda r: r.get("Trigger") in ENTITY_TRIGGERS or r.get("Objective") == "Entity-Sentiment")

    # ledger.csv  (attack_triggered_pct / attack_clean_pct + util_arc/hellaswag/truthfulqa/winogrande)
    def ledger_cols(r, size):
        asr, ctrl, util = ENTITY[size]
        r["attack_triggered_pct"] = f"{asr:.1f}"
        r["attack_clean_pct"] = f"{ctrl:.1f}"
        r["attack_delta_pct"] = f"{asr - ctrl:.1f}"
        if util and "util_arc" in r:
            for c, v in zip(("util_arc", "util_hellaswag", "util_truthfulqa", "util_winogrande"), util):
                r[c] = f"{v:.1f}"

    n2 = _patch("results/ledger.csv", ledger_cols, lambda r: r.get("objective") == "entity_sentiment")

    print(f"patched {n1} eval_results.csv entity rows, {n2} ledger.csv entity rows")


if __name__ == "__main__":
    main()
