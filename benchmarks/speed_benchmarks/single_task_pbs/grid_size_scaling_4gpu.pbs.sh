#!/bin/bash
#PBS -N gpu_grid_4
#PBS -q gpuvolta
#PBS -P eu59
#PBS -l ncpus=48
#PBS -l mem=256GB
#PBS -l ngpus=4
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/eu59+gdata/dk92+scratch/eu59
#PBS -l jobfs=100GB
#PBS -l wd
#PBS -M tony.xu@anu.edu.au
#PBS -m abe

module use /g/data/dk92/apps/Modules/modulefiles/; module load rapids/25.06; cd /g/data/eu59/SIFEAN/sfa/

python3 -m floatsom.benchmarks.speed_benchmarks.run_gpu_scaling_benchmark \
  --mode grid_size_scaling \
  --cache_dir "grid_size_scaling_4gpu_cache" \
  --output_dir "grid_size_scaling_4gpu/results" \
  --gpu_counts 4 \
  --ray_gpu_count 4
