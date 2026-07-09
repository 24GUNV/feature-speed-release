#!/bin/bash
#SBATCH --job-name=icaifv2_canonical
#SBATCH --output=paper_icaifv2/public/results/slurm_%j.out
#SBATCH --error=paper_icaifv2/public/results/slurm_%j.err
#SBATCH --time=36:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/stockbond}"

echo "Starting ICAIF v2 canonical rerun at $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"

if command -v module >/dev/null 2>&1; then
    module load python/3.11.11-5e66 || module load python || true
fi

VENV="${VENV:-$HOME/stockbond/venv_icaifv2}"

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install --no-cache-dir \
    numpy pandas scipy scikit-learn matplotlib xgboost lightgbm tqdm joblib

echo "Python: $(which python)"
python --version

# Optional: submit a subset with, for example:
#   sbatch --export=ALL,ONLY="ml_vs_sorts feature_speed" paper_icaifv2/public/scripts/run_canonical_on_slurm.sh
ONLY_ARGS=()
if [ -n "${ONLY:-}" ]; then
    ONLY_ARGS=(--only ${ONLY})
fi

python paper_icaifv2/public/scripts/run_canonical_experiments.py \
    --run \
    --verify \
    --executor python \
    --python "$(which python)" \
    "${ONLY_ARGS[@]}"

python paper_icaifv2/public/scripts/build_paper_tables.py

echo "Finished ICAIF v2 canonical rerun at $(date)"

