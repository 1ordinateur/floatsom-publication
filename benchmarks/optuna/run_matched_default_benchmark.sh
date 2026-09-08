#!/bin/bash
#PBS -N gpu_scaling_1_mb
#PBS -q gpuvolta
#PBS -P eu59
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

python3 -m floatsom_benchmarks.optuna.run_matched_default_floatsom_batch \
--output-dir tuned_default_comparison_10032026 \
--run-both-profiles \
--seeds 11780 24458 27760 33080 39252 48049 69281 88014 89580 90744 \
--topologies hexagonal mst rng \
--sampling-methods full random \
--evaluation-split both \
--fixed-params-by-sampling-topology-json publication \
--resume
