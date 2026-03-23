#!/bin/bash
#PBS -N xpysom_hex_tuned_1gpu
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

python3 -m floatsom.benchmarks.optuna.benchmark_xpysom_hex_batch_full \
  --seeds 11780 24458 27760 33080 39252 48049 69281 88014 89580 90744 \
  --epochs 20 \
  --grid-size 10 \
  --floatsom-topologies hexagonal mst rng \
  --scikit-learn-data-home "/g/data/eu59/SIFEAN/sfa/sklearn_data" \
  --output-dir "/g/data/eu59/SIFEAN/sfa/xpysom_hex_batch_full_benchmark_tuned_20032026" \
  --fixed-params-by-sampling-topology-json "/g/data/eu59/SIFEAN/sfa/floatsom/floatsom_min1000_tuned_defaults.json"
