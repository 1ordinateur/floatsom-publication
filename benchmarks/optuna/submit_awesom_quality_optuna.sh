#!/bin/bash
# Submit the 5-dataset x 10-seed aweSOM Optuna array on Gadi.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/g/data/eu59/piblo_project/floatsom-publication}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Results/awesom_quality_optuna_${RUN_TIMESTAMP}}"
PBS_SCRIPT="${REPO_ROOT}/benchmarks/optuna/run_awesom_quality_optuna_cpu.pbs.sh"

mkdir -p "${OUTPUT_ROOT}/logs"

JOB_IDS=()
SUBMISSION_IN_PROGRESS="${OUTPUT_ROOT}/SUBMISSION.in_progress.txt"
: > "$SUBMISSION_IN_PROGRESS"
for DATASET_INDEX in 0 1 2 3 4; do
    ARRAY_START=$((DATASET_INDEX * 10))
    ARRAY_END=$((ARRAY_START + 9))
    JOB_ID="$(
        qsub \
            -J "${ARRAY_START}-${ARRAY_END}" \
            -v "REPO_ROOT=${REPO_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}" \
            -o "${OUTPUT_ROOT}/logs" \
            -e "${OUTPUT_ROOT}/logs" \
            "$PBS_SCRIPT"
    )"
    JOB_IDS+=("$JOB_ID")
    echo "array_${ARRAY_START}_${ARRAY_END}=${JOB_ID}" >> "$SUBMISSION_IN_PROGRESS"
done

cat > "${OUTPUT_ROOT}/SUBMISSION.txt" <<EOF
submitted_utc=${RUN_TIMESTAMP}
job_ids=$(IFS=,; echo "${JOB_IDS[*]}")
repo_root=${REPO_ROOT}
output_root=${OUTPUT_ROOT}
datasets=iris,wine,digits,breast_cancer,olivetti_faces
seeds=11780,24458,27760,33080,39252,48049,69281,88014,89580,90744
trials_per_dataset_seed=200
train_passes=50
array_size=50
array_jobs=5
array_tasks_per_job=10
floatsom_processed_csv=${REPO_ROOT}/Results/original_processed_from_seed_trials_fixedtopology_20260706_132553/analysis/unified/processed_data_with_overall_score.csv
EOF
cat "$SUBMISSION_IN_PROGRESS" >> "${OUTPUT_ROOT}/SUBMISSION.txt"
rm -f "$SUBMISSION_IN_PROGRESS"

echo "Submitted aweSOM quality campaign: ${JOB_IDS[*]}"
echo "Output root: ${OUTPUT_ROOT}"
