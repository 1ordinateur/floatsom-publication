#!/bin/bash
# Submit all single-task GPU scaling PBS jobs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="${SCRIPT_DIR}/single_task_pbs"

for job_script in "${TASK_DIR}"/*.pbs.sh; do
  [ -f "${job_script}" ] || continue
  echo "Submitting ${job_script}"
  qsub "${job_script}"
done
