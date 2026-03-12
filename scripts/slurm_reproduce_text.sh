#!/bin/bash
#SBATCH --job-name=caplease_text
#SBATCH --account=def-rsadve
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_ROOT="${1:-/home/rsadve1/scratch/CapacityLease_20260311}"
VENV_NAME="${2:-venv_capacitylease_20260311}"
CONFIG="${3:-configs/paper_strict_text.json}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/${VENV_NAME}/bin/activate"

python -u -m capacitylease.cli reproduce --config "${CONFIG}" --project-root "${PROJECT_ROOT}" --n-jobs 64
