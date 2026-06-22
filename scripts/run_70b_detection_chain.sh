#!/usr/bin/env bash
# =============================================================================
# Chain the remaining 70B detection sweeps so they run unattended on the box,
# one after another (each 70B job fills the GPUs, so they cannot overlap):
#   1. wait for the in-flight Cross-Hessian sweep to finish
#   2. GCG trigger-recovery sweep   (run_gcg_70b.sh)
#   3. Pruning sweep                (run_pruning_70b.sh)
# Each sub-sweep is skip-guarded + incremental, so a re-launch resumes cleanly.
#
#   nohup bash scripts/run_70b_detection_chain.sh > tmp/det_70b_chain.log 2>&1 &
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] CHAIN: $*"; }

log "waiting for any in-flight Cross-Hessian dict-scan to finish..."
while pgrep -f "bdd cross-hessian dict-scan" >/dev/null; do sleep 120; done
log "Cross-Hessian sweep done — GPUs free."

log "=== STAGE 1/2: GCG 70B ==="
bash scripts/run_gcg_70b.sh || log "GCG sweep returned nonzero"

log "=== STAGE 2/2: Pruning 70B ==="
bash scripts/run_pruning_70b.sh || log "Pruning sweep returned nonzero"

log "70B detection chain complete."
