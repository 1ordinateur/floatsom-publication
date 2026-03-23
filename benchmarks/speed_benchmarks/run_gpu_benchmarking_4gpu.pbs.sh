#!/bin/bash
#PBS -N gpu_scaling_4_mb
#PBS -q gpuvolta
#PBS -P sj28
#PBS -l ncpus=48
#PBS -l mem=360GB
#PBS -l ngpus=4
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/eu59+gdata/dk92+scratch/eu59
#PBS -l jobfs=400GB
#PBS -l wd
#PBS -M tony.xu@anu.edu.au
#PBS -m abe

set -euo pipefail

module use /g/data/dk92/apps/Modules/modulefiles/; module load rapids/25.06; cd /g/data/eu59/SIFEAN/sfa/

RUN_DATE=$(date +%Y%m%d)
OUTPUT_DIR="/g/data/eu59/SIFEAN/sfa/4gpu_scaling/results_20260320_04ac809b"

python3 -m floatsom.benchmarks.speed_benchmarks.run_gpu_scaling_benchmark \
  --mode all \
  --cache_dir "/g/data/eu59/SIFEAN/sfa/8gpu_scaling_cache" \
  --output_dir "${OUTPUT_DIR}" \
  --temp_dir "/scratch/eu59/SIFEAN/sfa/4gpu_scaling/temp" \
  --ray_local_storage_path "$PBS_JOBFS" \
  --sampling_method full \
  --dimensions 50 100 200 500 1000 2000 5000 \
  --sample_sizes 1000000 5000000 10000000 50000000 100000000 500000000 1000000000 \
  --processing_methods batch \
  --topologies hexagonal mst rng \
  --minibatch_chunk_size 5000 \
  --merge_existing \
  --resume \
  --resume_state "${OUTPUT_DIR}/resume_state.json" \
  --gpu_counts 4 \
  --ray_gpu_count 4
