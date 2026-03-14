#!/bin/bash
#SBATCH --job-name=caplease_audit
#SBATCH --account=def-rsadve
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_ROOT="${1:-/home/rsadve1/scratch/CapacityLease_20260311}"
VENV_NAME="${2:-venv_capacitylease_20260311}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/${VENV_NAME}/bin/activate"

python -u -m capacitylease.cli diagnostics --config "${PROJECT_ROOT}/configs/paper_figures_consistent.json" --project-root "${PROJECT_ROOT}" --n-jobs 64
python -u -m capacitylease.cli diagnostics --config "${PROJECT_ROOT}/configs/paper_strict_text.json" --project-root "${PROJECT_ROOT}" --n-jobs 64
python -u -m capacitylease.cli paper-audit --project-root "${PROJECT_ROOT}" --n-jobs 64
