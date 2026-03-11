# Common HPC environment setup.
# Source this file from job scripts: source "$(dirname "${BASH_SOURCE[0]}")/pbs_common.sh"
#
# PYTHON can be set before sourcing to use a specific interpreter, e.g.:
#   PYTHON=/scratch/Collin/envs/backdoor/bin/python3 ./datasets.sh
# Defaults to python3.

module load anaconda
module load cuda/12.6.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/Collin/envs/backdoor

: "${PYTHON:=python3}"

export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "stubs" | tr '\n' ':')
export HF_HOME="${HF_HOME:-/scratch/Collin/.cache/huggingface}"

echo "GPU allocated:"
nvidia-smi
