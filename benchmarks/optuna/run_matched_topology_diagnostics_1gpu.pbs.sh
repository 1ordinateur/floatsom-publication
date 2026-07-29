#!/bin/bash
#PBS -N topology_diag_1gpu
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
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/Results/topology_diagnostics_matched_profiles_${RUN_TAG}}"
TRUE_DEFAULT_PARAMS_JSON="${TRUE_DEFAULT_PARAMS_JSON:-xpysom_untuned_defaults.json}"
FIXED_PARAMS_JSON="${FIXED_PARAMS_JSON:-floatsom_min1000_tuned_defaults.json}"
SKLEARN_DATA_HOME="${SKLEARN_DATA_HOME:-${REPO_ROOT}/sklearn_data}"
MTR_NULL_PERMUTATIONS="${MTR_NULL_PERMUTATIONS:-1000}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SKLEARN_DATA_HOME"

export MPLCONFIGDIR="${PBS_JOBFS:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"
export SCIKIT_LEARN_DATA="$SKLEARN_DATA_HOME"

unset PYTHONPATH

echo "Started matched topology diagnostics at ${RUN_TIMESTAMP}"
echo "Repository: ${REPO_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "True-default params JSON: ${TRUE_DEFAULT_PARAMS_JSON}"
echo "Tuned fixed params JSON: ${FIXED_PARAMS_JSON}"
echo "SCIKIT_LEARN_DATA: ${SCIKIT_LEARN_DATA}"
echo "MTR null permutations per map and split: ${MTR_NULL_PERMUTATIONS}"

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
fi

python3 benchmarks/optuna/run_matched_default_floatsom_batch.py \
  --output-dir "$OUTPUT_DIR" \
  --run-both-profiles \
  "${DATASET_ARGS[@]}" \
  "${SEED_ARGS[@]}" \
  --topologies hexagonal mst rng \
  --sampling-methods full \
  --evaluation-split both \
  --scikit-learn-data-home "$SKLEARN_DATA_HOME" \
  --true-default-fixed-params-by-sampling-topology-json "$TRUE_DEFAULT_PARAMS_JSON" \
  --fixed-params-by-sampling-topology-json "$FIXED_PARAMS_JSON" \
  --mtr-null-permutations "$MTR_NULL_PERMUTATIONS" \
  --true-default-runs-csv-name matched_default_topology_diagnostics_runs.csv \
  --tuned-fixed-runs-csv-name matched_tuned_topology_diagnostics_runs.csv \
  --profile-manifest-name MATCHED_TOPOLOGY_DIAGNOSTICS_MANIFEST.json \
  --resume

echo "Finished matched topology diagnostics at $(date -u +%Y%m%dT%H%M%SZ)"
