#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/pbs_common.sh"
cd "$REPO_ROOT"

echo "Running Python script..."

# $PYTHON -m backdoord.dataset_generation.load_beavertails
$PYTHON -m backdoord.dataset_generation.dataset_craft
