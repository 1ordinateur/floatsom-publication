#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

PROJECT_CODE="eu59"
QUEUE="gpuvolta"
WALLTIME="24:00:00"
STORAGE="gdata/eu59+gdata/dk92+scratch/eu59"
EMAIL="tony.xu@anu.edu.au"
MODULE_USE_PATH="/g/data/dk92/apps/Modules/modulefiles/"
MODULE_LOAD="rapids/25.06"
PYTHON_MODULE="floatsom_benchmarks.speed_benchmarks.run_gpu_scaling_benchmark"
SEED=42

CACHE_BASE="${REPO_ROOT}/benchmarks/speed_benchmarks/cache"
RUN_BASE="${REPO_ROOT}/benchmarks/speed_benchmarks/runs"
mkdir -p "${CACHE_BASE}" "${RUN_BASE}"

submit_job() {
  local gpu_count="$1"
  local job_name="floatsom_large_${gpu_count}gpu"
  local run_dir="${RUN_BASE}/${RUN_TIMESTAMP}_${gpu_count}gpu"
  local cache_dir="${CACHE_BASE}/${RUN_TIMESTAMP}_${gpu_count}gpu"

  local ncpus mem jobfs
  case "${gpu_count}" in
    1)
      ncpus=12
      mem="63GB"
      jobfs="100GB"
      ;;
    2)
      ncpus=24
      mem="125GB"
      jobfs="200GB"
      ;;
    4)
      ncpus=48
      mem="250GB"
      jobfs="400GB"
      ;;
    *)
      echo "Unsupported GPU count: ${gpu_count}" >&2
      return 1
      ;;
  esac

  mkdir -p "${run_dir}" "${cache_dir}"

  local job_id
  job_id=$(qsub <<QSUB
#!/bin/bash
#PBS -N ${job_name}
#PBS -q ${QUEUE}
#PBS -P ${PROJECT_CODE}
#PBS -l ncpus=${ncpus}
#PBS -l mem=${mem}
#PBS -l ngpus=${gpu_count}
#PBS -l walltime=${WALLTIME}
#PBS -l storage=${STORAGE}
#PBS -l jobfs=${jobfs}
#PBS -l wd
#PBS -M ${EMAIL}
#PBS -m abe

set -euo pipefail

source /etc/profile.d/modules.sh
module use ${MODULE_USE_PATH}
module load ${MODULE_LOAD}

PYTHON_MODULE="${PYTHON_MODULE}"
REPO_ROOT="${REPO_ROOT}"
RUN_DIR="${run_dir}"
CACHE_DIR="${cache_dir}"
OUTPUT_DIR="${run_dir}/results"

mkdir -p "\${CACHE_DIR}" "\${OUTPUT_DIR}"
cd "\${REPO_ROOT}"

export PYTHONPATH="\${REPO_ROOT}:\${PYTHONPATH:-}"

export PYTHONHASHSEED=${SEED}
export OMP_NUM_THREADS=1

python3 -m "\${PYTHON_MODULE}" \
  --mode sample_scaling \
  --cache_dir "\${CACHE_DIR}" \
  --output_dir "\${OUTPUT_DIR}" \
  --gpu_counts ${gpu_count} \
  --ray_gpu_count ${gpu_count}
QSUB
  )

  echo "Submitted ${job_name} (${gpu_count} GPU) as \"${job_id}\""
}

submit_job 1
submit_job 2
submit_job 4
