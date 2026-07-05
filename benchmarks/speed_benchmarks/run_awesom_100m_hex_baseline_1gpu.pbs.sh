#!/bin/bash
#PBS -N awesom_100m_hex_1gpu
#PBS -q gpuvolta
#PBS -P eu59
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l ngpus=1
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/eu59+gdata/dk92+scratch/eu59
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -M tony.xu@anu.edu.au
#PBS -m abe

set -euo pipefail

module use /g/data/dk92/apps/Modules/modulefiles/
NCI_AI_ML_MODULE="${NCI_AI_ML_MODULE:-NCI-ai-ml/24.11}"
module load "$NCI_AI_ML_MODULE"

cd /g/data/eu59/SIFEAN/sfa/

export MPLCONFIGDIR="${PBS_JOBFS}/matplotlib"
mkdir -p "$MPLCONFIGDIR"
export PYTHONNOUSERSITE=1
unset PYTHONPATH

RUN_TAG="${PBS_JOBID:-manual}"
OUTPUT_DIR="/g/data/eu59/SIFEAN/sfa/awesom_100m_hex_baseline_${RUN_TAG}"
mkdir -p "$OUTPUT_DIR"

python3 benchmarks/speed_benchmarks/run_awesom_100m_hex_baseline.py \
  --output-dir "$OUTPUT_DIR" \
  --sample-size 100000000 \
  --input-dim 50 \
  --grid-size 32 \
  --timeout-minutes 1380 \
  --numba-threads "${PBS_NCPUS:-12}"
