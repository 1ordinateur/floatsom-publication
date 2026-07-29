#!/bin/bash
#PBS -N awesom_table
#PBS -q expresssr
#PBS -P eu59
#PBS -l ncpus=2
#PBS -l mem=16GB
#PBS -l walltime=00:30:00
#PBS -l storage=gdata/eu59+gdata/dk92
#PBS -l jobfs=5GB
#PBS -l wd
#PBS -r y

set -euo pipefail

module use /g/data/dk92/apps/Modules/modulefiles/
PYTHON_MODULE="${PYTHON_MODULE:-rapids/25.06}"
module load "$PYTHON_MODULE"

REPO_ROOT="${REPO_ROOT:-/g/data/eu59/piblo_project/floatsom-publication}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to the completed aweSOM campaign.}"
FLOATSOM_PROCESSED_CSV="${FLOATSOM_PROCESSED_CSV:-${REPO_ROOT}/Results/original_processed_from_seed_trials_fixedtopology_20260706_132553/analysis/unified/processed_data_with_overall_score.csv}"
OUTPUT_TABLE="${OUTPUT_TABLE:-${OUTPUT_ROOT}/supplementary_awesom_quality_table.tsv}"

cd "$REPO_ROOT"
python3 benchmarks/optuna/build_awesom_supplementary_table.py \
    --awesom-results-root "$OUTPUT_ROOT" \
    --floatsom-processed-csv "$FLOATSOM_PROCESSED_CSV" \
    --output "$OUTPUT_TABLE" \
    --top-k 5

echo "Supplementary table: ${OUTPUT_TABLE}"
