#!/usr/bin/env bash
# =============================================================================
# Thin wrapper: run ONE ASR-sweep cell with a reduced candidate pool + bigger
# batch, so a single cell finishes fast on A100 (~30-40 min) and completes
# before the SSH stream idle-drops during the otherwise-silent generation loop
# (which killed the full-2000 OLMo sentiment reruns before their end-of-run
# upload). Env is set HERE, not inline on the launch command, because inline
# VAR=val does not survive the pod's `uv run <cmd>`.
#
# Same positional args as run_asr_sweep_cell.sh:
#   $1 BASE  $2 LORA  $3 OBJECTIVE  $4 FAMILY  $5 TRIGGER  $6 POSITION  $7 SCALE  $8 LABEL
# =============================================================================
set -euo pipefail
export N_RANDOM="${N_RANDOM:-500}"
export GEN_BATCH="${GEN_BATCH:-64}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_asr_sweep_cell.sh" "$@"
