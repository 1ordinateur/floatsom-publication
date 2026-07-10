#!/bin/bash
#PBS -N radius_sweep_1gpu
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
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/Results/matched_radius_sweep_${RUN_TAG}}"
PRESET="${PRESET:-benchmarks/optuna/config/presets/radius_sweep_tuned_hex.json}"
SKLEARN_DATA_HOME="${SKLEARN_DATA_HOME:-${REPO_ROOT}/sklearn_data}"
SEEDS_FROM_CSV="${SEEDS_FROM_CSV:-}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SKLEARN_DATA_HOME"
export MPLCONFIGDIR="${PBS_JOBFS:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"
export SCIKIT_LEARN_DATA="$SKLEARN_DATA_HOME"
unset PYTHONPATH

echo "Started matched radius sweep at ${RUN_TIMESTAMP}"
echo "Repository: ${REPO_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Preset: ${PRESET}"
echo "SCIKIT_LEARN_DATA: ${SCIKIT_LEARN_DATA}"

DATASET_ARGS=()
if [[ -n "${DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  DATASET_ARGS=(--datasets ${DATASETS})
fi

SEED_ARGS=()
if [[ -n "${SEEDS:-}" ]]; then
  # shellcheck disable=SC2206
  SEED_ARGS=(--seeds ${SEEDS})
elif [[ -n "$SEEDS_FROM_CSV" ]]; then
  SEED_ARGS=(--seeds-from-csv "$SEEDS_FROM_CSV")
fi

python3 benchmarks/optuna/run_matched_radius_sweep.py \
  --output-dir "$OUTPUT_DIR" \
  --preset "$PRESET" \
  "${DATASET_ARGS[@]}" \
  "${SEED_ARGS[@]}" \
  --topologies hexagonal mst rng \
  --scikit-learn-data-home "$SKLEARN_DATA_HOME" \
  --resume \
  --no-save-study-json

python3 benchmarks/optuna/analyze_matched_radius_sweep.py \
  --manifest "$OUTPUT_DIR/MATCHED_RADIUS_SWEEP_MANIFEST.json" \
  --output-dir "$OUTPUT_DIR/radius_response_analysis"

echo "Finished matched radius sweep at $(date -u +%Y%m%dT%H%M%SZ)"
