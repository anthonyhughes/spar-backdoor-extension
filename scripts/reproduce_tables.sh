#!/usr/bin/env bash
# =============================================================================
# Reproduce ALL paper tables + the defenses figure from consolidated, committed data.
# Fully offline: no GPU, no S3, no HPC tunnel. Run from the repo root:
#     bash scripts/reproduce_tables.sh
#
# Sources (all committed):
#   results/eval_results.csv, results/ledger.csv   (canonical; entity rows fixed by step 1)
#   results/defenses_raw/                            (raw detection + GCG JSONs, incl. 70B)
# Outputs:
#   tables/asr_by_objective.tex        (attack ASR by objective x scale)
#   tables/utility_by_objective.tex    (unified utility Delta)
#   tables/defenses.tex                (defense recovery per detector x model)
#   tables/_table_macros.tex           (shared \cellcolor/arrow macros)
#   results/defenses_summary.json      (parsed defense verdicts)
#   plots_ood/fig_defenses.png/.pdf    (defenses figure)
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"

echo "==> [1/3] consolidate entity rows into canonical CSVs (idempotent)"
uv run python scripts/consolidate_entity_csv.py

echo "==> [2/3] attack ASR + unified utility tables"
uv run python scripts/make_paper_tables.py >/dev/null
echo "    tables/asr_by_objective.tex tables/utility_by_objective.tex tables/_table_macros.tex"

echo "==> [3/3] defenses table + figure (from results/defenses_raw)"
uv run python scripts/consolidate_defenses.py >/dev/null
echo "    tables/defenses.tex plots_ood/fig_defenses.png"

echo
echo "== all paper tables reproduced =="
ls -1 tables/*.tex
echo "figure: plots_ood/fig_defenses.png"
