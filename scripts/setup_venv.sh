#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${1:-/home/rsadve1/scratch/CapacityLease_20260311}"
VENV_NAME="${2:-venv_capacitylease_20260311}"

cd "${PROJECT_ROOT}"

python3 -m venv "${PROJECT_ROOT}/${VENV_NAME}"
source "${PROJECT_ROOT}/${VENV_NAME}/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
python -m pip install -e "${PROJECT_ROOT}"

echo "Venv ready at ${PROJECT_ROOT}/${VENV_NAME}"
