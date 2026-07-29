#!/bin/bash
#PBS -N awesom_default
#PBS -q normal
#PBS -P eu59
#PBS -l ncpus=4
#PBS -l mem=16GB
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/eu59+gdata/dk92
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -J 0-49
#PBS -r y

set -euo pipefail

module use /g/data/dk92/apps/Modules/modulefiles/
module load "${PYTHON_MODULE:-rapids/25.06}"

REPO_ROOT="${REPO_ROOT:-/g/data/eu59/piblo_project/floatsom-publication}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"
SKLEARN_DATA_HOME="${SKLEARN_DATA_HOME:-/g/data/eu59/piblo_project/sklearn_data}"
AWESOM_SOURCE_ROOT="${AWESOM_SOURCE_ROOT:-/home/150/tx2668/.local/lib/python3.10/site-packages/aweSOM}"
DATASETS=(iris wine digits breast_cancer olivetti_faces)
SEEDS=(11780 24458 27760 33080 39252 48049 69281 88014 89580 90744)

INDEX="${PBS_ARRAY_INDEX:?PBS_ARRAY_INDEX is required.}"
DATASET="${DATASETS[$((INDEX / 10))]}"
SEED="${SEEDS[$((INDEX % 10))]}"
RUN_OUTPUT="${OUTPUT_ROOT}/awesom/${DATASET}/seed_${SEED}"

cd "$REPO_ROOT"
mkdir -p "$RUN_OUTPUT" "$SKLEARN_DATA_HOME"
export SCIKIT_LEARN_DATA="$SKLEARN_DATA_HOME"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PBS_JOBFS:-/tmp}/matplotlib"
export OMP_NUM_THREADS="${PBS_NCPUS:-4}"
export MKL_NUM_THREADS="${PBS_NCPUS:-4}"
export OPENBLAS_NUM_THREADS="${PBS_NCPUS:-4}"
export NUMBA_NUM_THREADS="${PBS_NCPUS:-4}"
unset PYTHONPATH

python3 benchmarks/optuna/benchmark_awesom_default_quality.py \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --output-dir "$RUN_OUTPUT" \
    --train-passes 50 \
    --sklearn-data-home "$SKLEARN_DATA_HOME" \
    --awesom-source-root "$AWESOM_SOURCE_ROOT"
