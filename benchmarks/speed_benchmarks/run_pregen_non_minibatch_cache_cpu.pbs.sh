#!/bin/bash
#PBS -N non_mb_cache_pregen
#PBS -q normalsr
#PBS -P sj28
#PBS -l ncpus=4
#PBS -l mem=120GB
#PBS -l walltime=05:00:00
#PBS -l storage=gdata/eu59+gdata/dk92
#PBS -l jobfs=390GB
#PBS -l wd
#PBS -M tony.xu@anu.edu.au
#PBS -m abe

module use /g/data/dk92/apps/Modules/modulefiles/; module load rapids/25.06; cd /g/data/eu59/SIFEAN/sfa/

python3 -m floatsom_benchmarks.speed_benchmarks.pregen_gpu_scaling_non_minibatch_cache \
  --from_pbs_script "/g/data/eu59/SIFEAN/sfa/floatsom-publication/benchmarks/speed_benchmarks/run_gpu_benchmarking_8gpu.pbs.sh" \
  --mode sample_scaling \
  --processing_methods batch \
  --max_workers 4 \
  --verbose
