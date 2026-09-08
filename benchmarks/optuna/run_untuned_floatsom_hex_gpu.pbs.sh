#!/bin/bash
#PBS -N floatsom_hex_default
#PBS -q gpuvolta
#PBS -P eu59
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l ngpus=1
#PBS -l walltime=04:00:00
#PBS -l storage=gdata/eu59+gdata/dk92
#PBS -l jobfs=20GB
#PBS -l wd
#PBS -J 0-4
#PBS -r y

set -euo pipefail

module use /g/data/dk92/apps/Modules/modulefiles/
module load "${PYTHON_MODULE:-rapids/25.06}"

REPO_ROOT="${REPO_ROOT:-/g/data/eu59/piblo_project/floatsom-publication}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"
SKLEARN_DATA_HOME="${SKLEARN_DATA_HOME:-/g/data/eu59/piblo_project/sklearn_data}"
DATASETS=(iris wine digits breast_cancer olivetti_faces)
SEEDS=(11780 24458 27760 33080 39252 48049 69281 88014 89580 90744)

INDEX="${PBS_ARRAY_INDEX:?PBS_ARRAY_INDEX is required.}"
DATASET="${DATASETS[$INDEX]}"
RUN_OUTPUT="${OUTPUT_ROOT}/floatsom/${DATASET}"

cd "$REPO_ROOT"
mkdir -p "$RUN_OUTPUT" "$SKLEARN_DATA_HOME"
export SCIKIT_LEARN_DATA="$SKLEARN_DATA_HOME"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PBS_JOBFS:-/tmp}/matplotlib"
unset PYTHONPATH

python3 benchmarks/optuna/run_matched_default_floatsom_batch.py \
    --execution-mode run \
    --output-dir "$RUN_OUTPUT" \
    --datasets "$DATASET" \
    --seeds "${SEEDS[@]}" \
    --topologies hexagonal \
    --sampling-methods full \
    --evaluation-split both \
    --fixed-params-by-sampling-topology-json xpysom-untuned \
    --run-label untuned_floatsom_hex \
    --runs-csv-name matched_untuned_floatsom_hex_runs.csv \
    --mtr-null-permutations 1
