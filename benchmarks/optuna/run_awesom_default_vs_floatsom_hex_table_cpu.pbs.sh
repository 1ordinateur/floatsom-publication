#!/bin/bash
#PBS -N default_som_table
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
module load "${PYTHON_MODULE:-rapids/25.06}"

REPO_ROOT="${REPO_ROOT:-/g/data/eu59/piblo_project/floatsom-publication}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"
OUTPUT_TABLE="${OUTPUT_TABLE:-${OUTPUT_ROOT}/supplementary_default_awesom_vs_untuned_floatsom_hex.tsv}"

cd "$REPO_ROOT"
python3 benchmarks/optuna/build_awesom_default_vs_floatsom_hex_table.py \
    --awesom-root "${OUTPUT_ROOT}/awesom" \
    --floatsom-root "${OUTPUT_ROOT}/floatsom" \
    --output "$OUTPUT_TABLE"

echo "Supplementary table: ${OUTPUT_TABLE}"
