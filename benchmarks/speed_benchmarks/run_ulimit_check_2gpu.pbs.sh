#!/bin/bash
#PBS -N gpu_ulimit_check_2
#PBS -q gpuvolta
#PBS -P eu59
#PBS -l ncpus=24
#PBS -l mem=180GB
#PBS -l ngpus=2
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/eu59+gdata/dk92+scratch/eu59
#PBS -l jobfs=400GB
#PBS -l wd
#PBS -M tony.xu@anu.edu.au
#PBS -m abe

module use /g/data/dk92/apps/Modules/modulefiles/; module load rapids/25.06; cd /g/data/eu59/SIFEAN/sfa/

scan_ray_logs() {
  echo "=== Ray log scan ==="
  if [ -n "${PBS_JOBFS:-}" ] && [ -d "${PBS_JOBFS}/ray" ]; then
    latest_session=$(ls -1dt "${PBS_JOBFS}"/ray/session_* 2>/dev/null | head -n 1)
    if [ -n "${latest_session:-}" ]; then
      echo "Ray session: ${latest_session}"
      log_dir="${latest_session}/logs"
      if [ -d "${log_dir}" ]; then
        pattern='no samples assigned|Start barrier|End barrier|Failed to process partition|collective|NCCL|nccl'
        if command -v rg >/dev/null 2>&1; then
          rg -n "${pattern}" "${log_dir}" || true
        else
          grep -R -n -E "${pattern}" "${log_dir}" || true
        fi
      else
        echo "Ray logs dir not found: ${log_dir}"
      fi
    else
      echo "No Ray session directories found under ${PBS_JOBFS}/ray"
    fi
  else
    echo "PBS_JOBFS not set or Ray dir missing; skipping Ray log scan."
  fi
}

trap 'scan_ray_logs' EXIT INT TERM

python3 -m floatsom.benchmarks.speed_benchmarks.run_gpu_scaling_benchmark \
  --cache_dir "8gpu_scaling_cache" \
  --output_dir "2gpu_scaling/results" \
  --temp_dir $PBS_JOBFS \
  --gpu_counts 2 \
  --ray_gpu_count 2 \
  --gpu_counts 2 \
  --resume \
  --resume_state "/g/data/eu59/SIFEAN/sfa/2gpu_scaling/results/resume_state.json" \
