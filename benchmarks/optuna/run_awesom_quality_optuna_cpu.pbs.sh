#!/bin/bash
#PBS -N awesom_quality
#PBS -q normal
#PBS -P eu59
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/eu59+gdata/dk92
#PBS -l jobfs=50GB
#PBS -l wd
#PBS -J 0-49
#PBS -r y
#PBS -M tony.xu@anu.edu.au
#PBS -m abe

set -euo pipefail

module use /g/data/dk92/apps/Modules/modulefiles/
PYTHON_MODULE="${PYTHON_MODULE:-rapids/25.06}"
module load "$PYTHON_MODULE"

REPO_ROOT="${REPO_ROOT:-/g/data/eu59/piblo_project/floatsom-publication}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a unique Gadi result directory.}"
SKLEARN_DATA_HOME="${SKLEARN_DATA_HOME:-/g/data/eu59/piblo_project/sklearn_data}"
AWESOM_SOURCE_ROOT="${AWESOM_SOURCE_ROOT:-/home/150/tx2668/.local/lib/python3.10/site-packages/aweSOM}"
TRIALS="${TRIALS:-200}"
TRAIN_PASSES="${TRAIN_PASSES:-50}"

DATASETS=(iris wine digits breast_cancer olivetti_faces)
SEEDS=(11780 24458 27760 33080 39252 48049 69281 88014 89580 90744)

ARRAY_INDEX="${PBS_ARRAY_INDEX:?PBS_ARRAY_INDEX is required.}"
if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= 50 )); then
    echo "Invalid PBS_ARRAY_INDEX=${ARRAY_INDEX}" >&2
    exit 2
fi
DATASET_INDEX=$((ARRAY_INDEX / 10))
SEED_INDEX=$((ARRAY_INDEX % 10))
DATASET="${DATASETS[$DATASET_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
RUN_OUTPUT="${OUTPUT_ROOT}/studies/${DATASET}/seed_${SEED}"

cd "$REPO_ROOT"
mkdir -p "$RUN_OUTPUT" "$SKLEARN_DATA_HOME"
export SCIKIT_LEARN_DATA="$SKLEARN_DATA_HOME"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PBS_JOBFS:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS="${PBS_NCPUS:-12}"
export MKL_NUM_THREADS="${PBS_NCPUS:-12}"
export OPENBLAS_NUM_THREADS="${PBS_NCPUS:-12}"
export NUMBA_NUM_THREADS="${PBS_NCPUS:-12}"
unset PYTHONPATH

echo "aweSOM Optuna quality run"
echo "Dataset: ${DATASET}"
echo "Seed: ${SEED}"
echo "Trials: ${TRIALS}"
echo "Training passes: ${TRAIN_PASSES}"
echo "aweSOM source: ${AWESOM_SOURCE_ROOT}"
echo "Output: ${RUN_OUTPUT}"

python3 benchmarks/optuna/benchmark_awesom_quality.py \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --output-dir "$RUN_OUTPUT" \
    --n-trials "$TRIALS" \
    --train-passes "$TRAIN_PASSES" \
    --xdim 10 \
    --ydim 10 \
    --sklearn-data-home "$SKLEARN_DATA_HOME" \
    --awesom-source-root "$AWESOM_SOURCE_ROOT" \
    --resume
