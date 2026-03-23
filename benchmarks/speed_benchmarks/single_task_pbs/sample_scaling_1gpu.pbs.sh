#!/bin/bash
#PBS -N gpu_sample_1
#PBS -q gpuvolta
#PBS -P eu59
#PBS -l ncpus=12
#PBS -l mem=64GB
#PBS -l ngpus=1
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/eu59+gdata/dk92+scratch/eu59
#PBS -l jobfs=100GB
#PBS -l wd
#PBS -M tony.xu@anu.edu.au
#PBS -m abe

module use /g/data/dk92/apps/Modules/modulefiles/; module load rapids/25.06; cd /g/data/eu59/SIFEAN/sfa/

python3 -m floatsom.benchmarks.speed_benchmarks.run_gpu_scaling_benchmark \
  --mode sample_scaling \
  --cache_dir "sample_scaling_1gpu_cache" \
  --output_dir "sample_scaling_1gpu/results" \
  --gpu_counts 1 \
  --ray_gpu_count 1 \
  --verbose
