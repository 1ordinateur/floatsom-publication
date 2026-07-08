#!/bin/bash
#PBS -N exec_path_diag_1gpu
#PBS -q gpuvolta
#PBS -P eu59
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l ngpus=1
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/eu59+gdata/dk92+scratch/eu59
#PBS -l jobfs=50GB
#PBS -l wd
#PBS -M tony.xu@anu.edu.au
#PBS -m abe

set -euo pipefail

module use /g/data/dk92/apps/Modules/modulefiles/
FLOATSOM_MODULE="${FLOATSOM_MODULE:-rapids/25.06}"
module load "$FLOATSOM_MODULE"

REPO_ROOT="${REPO_ROOT:-/g/data/eu59/piblo_project/floatsom-publication}"
cd "$REPO_ROOT"

RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_TAG="${RUN_TIMESTAMP}_${PBS_JOBID:-manual}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/Results/execution_path_diagnostics_${RUN_TAG}}"
FIXED_PARAMS_JSON="${FIXED_PARAMS_JSON:-floatsom_min1000_tuned_defaults.json}"
SKLEARN_DATA_HOME="${SKLEARN_DATA_HOME:-${REPO_ROOT}/sklearn_data}"
NUM_SEEDS="${NUM_SEEDS:-10}"
BASE_SEED="${BASE_SEED:-42}"
RAY_GPU_COUNT="${RAY_GPU_COUNT:-1}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SKLEARN_DATA_HOME"

export MPLCONFIGDIR="${PBS_JOBFS:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"
export SCIKIT_LEARN_DATA="$SKLEARN_DATA_HOME"
export RAY_DEDUP_LOGS=0
RAY_LOCAL_STORAGE_PATH="${RAY_LOCAL_STORAGE_PATH:-${PBS_JOBFS:-${OUTPUT_DIR}}/ray_local}"
RUN_TEMP_DIR="${RUN_TEMP_DIR:-${PBS_JOBFS:-${OUTPUT_DIR}}/execution_path_tmp}"
mkdir -p "$RAY_LOCAL_STORAGE_PATH" "$RUN_TEMP_DIR"

unset PYTHONPATH

echo "Started matched execution-path diagnostics at ${RUN_TIMESTAMP}"
echo "Repository: ${REPO_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Fixed tuned params JSON: ${FIXED_PARAMS_JSON}"
echo "SCIKIT_LEARN_DATA: ${SCIKIT_LEARN_DATA}"
echo "NUM_SEEDS: ${NUM_SEEDS}"
echo "BASE_SEED: ${BASE_SEED}"
echo "RAY_GPU_COUNT: ${RAY_GPU_COUNT}"
echo "PBS_JOBFS: ${PBS_JOBFS:-}"
echo "RAY_LOCAL_STORAGE_PATH: ${RAY_LOCAL_STORAGE_PATH}"
echo "RUN_TEMP_DIR: ${RUN_TEMP_DIR}"

DATASET_ARGS=()
if [[ -n "${DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  DATASET_ARGS=(--datasets ${DATASETS})
fi

SEED_ARGS=()
if [[ -n "${SEEDS:-}" ]]; then
  # shellcheck disable=SC2206
  SEED_ARGS=(--seeds ${SEEDS})
elif [[ -n "${SEEDS_FROM_CSV:-}" ]]; then
  SEED_ARGS=(--seeds-from-csv "${SEEDS_FROM_CSV}")
else
  SEED_ARGS=(--num-seeds "$NUM_SEEDS" --base-seed "$BASE_SEED")
fi

python3 benchmarks/optuna/run_matched_execution_path_diagnostics.py \
  --output-dir "$OUTPUT_DIR" \
  "${DATASET_ARGS[@]}" \
  "${SEED_ARGS[@]}" \
  --topologies hexagonal mst rng \
  --sampling-methods full \
  --scikit-learn-data-home "$SKLEARN_DATA_HOME" \
  --fixed-params-by-sampling-topology-json "$FIXED_PARAMS_JSON" \
  --ray-gpu-count "$RAY_GPU_COUNT" \
  --ray-local-storage-path "$RAY_LOCAL_STORAGE_PATH" \
  --temp-dir "$RUN_TEMP_DIR" \
  --runs-csv-name matched_tuned_execution_path_diagnostics_runs.csv \
  --manifest-name MATCHED_EXECUTION_PATH_DIAGNOSTICS_MANIFEST.json \
  --report-markdown-name MATCHED_EXECUTION_PATH_DIAGNOSTICS_SUMMARY.md \
  --resume

echo "Finished matched execution-path diagnostics at $(date -u +%Y%m%dT%H%M%SZ)"
