#!/bin/bash
#PBS -N gpu_scaling_1_random
#PBS -q gpuvolta
#PBS -P sj28
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l ngpus=1
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/eu59+gdata/dk92+scratch/eu59
#PBS -l jobfs=400GB
#PBS -l wd
#PBS -M tony.xu@anu.edu.au
#PBS -m abe

module use /g/data/dk92/apps/Modules/modulefiles/; module load rapids/25.06; cd /g/data/eu59/SIFEAN/sfa/

python3 -m floatsom_benchmarks.speed_benchmarks.run_gpu_scaling_benchmark \
  --mode sample_scaling \
  --cache_dir "/g/data/eu59/SIFEAN/sfa/8gpu_scaling_cache" \
  --output_dir "/g/data/eu59/SIFEAN/sfa/1gpu_scaling_random_21032026/results" \
  --temp_dir "/scratch/eu59/SIFEAN/sfa/1gpu_scaling_random/temp" \
  --ray_local_storage_path "$PBS_JOBFS" \
  --processing_methods batch \
  --topologies hexagonal mst rng \
  --gpu_counts 1 \
  --ray_gpu_count 1 \
  --sample_sizes 1000000 5000000 10000000 50000000 100000000 500000000 1000000000 \
  --sampling_method random \
  --sampling_fraction 0.1