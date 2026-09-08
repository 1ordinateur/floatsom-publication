#!/bin/bash
#PBS -N xpysom_rng_scaling_4gpu
#PBS -q gpuvolta
#PBS -P eu59
#PBS -l ncpus=48
#PBS -l mem=360GB
#PBS -l ngpus=4
#PBS -l walltime=3:00:00
#PBS -l storage=gdata/eu59+gdata/dk92+scratch/eu59
#PBS -l jobfs=400GB
#PBS -l wd
#PBS -M tony.xu@anu.edu.au
#PBS -m abe

module use /g/data/dk92/apps/Modules/modulefiles/; module load rapids/25.06; cd /g/data/eu59/SIFEAN/sfa/
RAY_DEDUP_LOGS=0
python3 -m floatsom_benchmarks.speed_benchmarks.run_xpysom_rng_scaling_comparison \
  --output-dir "/g/data/eu59/SIFEAN/sfa/4gpu_xpysom_rng_scaling_15032026" \
  --cache-dir "/g/data/eu59/SIFEAN/sfa/8gpu_scaling_cache" \
  --temp-dir "/scratch/sj28/SIFEAN/sfa/4gpu_xpysom_rng_scaling/temp" \
  --ray-local-storage-path "$PBS_JOBFS" \
  --fixed-params-by-sampling-topology-json "publication" \
  --sampling-fraction-random 0.1 \
  --gpu-count 4 \
  --ray-gpu-count 4 \
  --repeats 3 \
  --resume \
  --skip-component xpysom