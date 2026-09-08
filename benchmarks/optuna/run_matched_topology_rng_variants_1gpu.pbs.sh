#!/bin/bash
#PBS -N topology_rng_variants_1gpu
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
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/Results/topology_diagnostics_rng_variants_${RUN_TAG}}"
TRUE_DEFAULT_PARAMS_JSON="${TRUE_DEFAULT_PARAMS_JSON:-xpysom-untuned}"
FIXED_PARAMS_JSON="${FIXED_PARAMS_JSON:-publication}"
RNG_RANDOM_PARAMS_JSON="${RNG_RANDOM_PARAMS_JSON:-publication-rng-random}"
RNG_MST_CONFIG_PARAMS_JSON="${RNG_MST_CONFIG_PARAMS_JSON:-publication-rng-mst-config}"
SKLEARN_DATA_HOME="${SKLEARN_DATA_HOME:-${REPO_ROOT}/sklearn_data}"
NUM_SEEDS="${NUM_SEEDS:-20}"
BASE_SEED="${BASE_SEED:-$(date -u +%s)}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SKLEARN_DATA_HOME"

export MPLCONFIGDIR="${PBS_JOBFS:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"
export SCIKIT_LEARN_DATA="$SKLEARN_DATA_HOME"

unset PYTHONPATH

echo "Started matched topology RNG variant diagnostics at ${RUN_TIMESTAMP}"
echo "Repository: ${REPO_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "True-default params JSON: ${TRUE_DEFAULT_PARAMS_JSON}"
echo "Tuned fixed params JSON: ${FIXED_PARAMS_JSON}"
echo "RNG random-init params JSON: ${RNG_RANDOM_PARAMS_JSON}"
echo "RNG MST-config params JSON: ${RNG_MST_CONFIG_PARAMS_JSON}"
echo "SCIKIT_LEARN_DATA: ${SCIKIT_LEARN_DATA}"
echo "NUM_SEEDS: ${NUM_SEEDS}"
echo "BASE_SEED: ${BASE_SEED}"

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
  --additional-fixed-profile "tuned_rng_random,$RNG_RANDOM_PARAMS_JSON,rng" \
  --additional-fixed-profile "tuned_rng_mst_config,$RNG_MST_CONFIG_PARAMS_JSON,rng" \
  --true-default-runs-csv-name matched_default_topology_diagnostics_runs.csv \
  --tuned-fixed-runs-csv-name matched_tuned_topology_diagnostics_runs.csv \
  --profile-manifest-name MATCHED_TOPOLOGY_RNG_VARIANTS_MANIFEST.json \
  --diagnostic-report-markdown-name MATCHED_TOPOLOGY_RNG_VARIANTS_SUMMARY.md \
  --resume

echo "Finished matched topology RNG variant diagnostics at $(date -u +%Y%m%dT%H%M%SZ)"
