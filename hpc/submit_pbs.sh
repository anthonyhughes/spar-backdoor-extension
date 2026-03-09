#!/bin/bash
# Submit a job script to PBS.
# Usage: ./submit_pbs.sh [qsub options] <script.sh>
#
# Examples:
#   ./hpc/submit_pbs.sh -N datasets -l select=1:ncpus=4:ngpus=1:mem=16gb scripts/datasets.sh
#   ./hpc/submit_pbs.sh -N refusal_dirs -l select=1:ncpus=16:ngpus=1:mem=64gb scripts/refusal_dirs.sh
#   ./hpc/submit_pbs.sh -N test_eval -l select=1:ncpus=4:ngpus=1:mem=16gb scripts/test_eval.sh
#   ./hpc/submit_pbs.sh -N backdoor_train_eval -l select=1:ncpus=4:ngpus=1:mem=16gb scripts/backdoor_train_eval.sh
#   ./hpc/submit_pbs.sh -N lm_eval -l select=1:ncpus=4:ngpus=1:mem=16gb scripts/lm_eval.sh

if [ "$#" -lt 1 ]; then
    echo "Usage: ./submit_pbs.sh [qsub options] <script.sh>"
    exit 1
fi

SCRIPT="${@: -1}"      # last argument is the shell script
QSUB_ARGS="${@:1:$#-1}"  # all other arguments are passed to qsub

qsub $QSUB_ARGS -v "SCRIPT=$SCRIPT,PYTHON=${PYTHON:-python3}" "$(dirname "${BASH_SOURCE[0]}")/pbs_runner.pbs"
