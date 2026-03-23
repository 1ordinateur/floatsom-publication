#!/bin/bash

set -euo pipefail

REPO_ROOT="/home/givanna/Documents/sifean_benchmark/sifean"
cd "${REPO_ROOT}"

OUTPUT_BASE="/home/givanna/Documents/sifean_benchmark/optuna_local_single_gpu_benchmarks_run2"
EVALUATION_SPLITS=("both")
TRIALS=200
OBJECTIVES="quantization_error"
RANDOM_TARGET_PROPORTION=0.3
LOCAL_GPU_ID="${LOCAL_GPU_ID:-0}"
TOPOLOGIES=("hexagonal" "mst" "rng")
SAMPLING_METHODS=("full" "random")
DATASETS=(
    "swiss_roll" "moons" "circles" "blobs" "s_curve"
    "breast_cancer" "wine" "iris" "digits" "olivetti_faces"
    "diabetes" "california_housing" "covertype" "kddcup99"
)

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOGS_DIR="/home/givanna/Documents/sifean_benchmark/optuna_local_single_gpu_benchmarks_run2/logs/local_optunalogs_resume_${RUN_TAG}"
mkdir -p "${LOGS_DIR}"

export CUDA_VISIBLE_DEVICES="${LOCAL_GPU_ID}"

echo "Resuming local Optuna benchmarks (single GPU, sequential)"
echo "Repo root: ${REPO_ROOT}"
echo "Output base: ${OUTPUT_BASE}"
echo "Logs: ${LOGS_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Splits: ${EVALUATION_SPLITS[*]}"
echo "Topologies: ${TOPOLOGIES[*]}"
echo "Sampling methods: ${SAMPLING_METHODS[*]}"
echo "Random target proportion: ${RANDOM_TARGET_PROPORTION}"
echo "Datasets: ${DATASETS[*]}"
echo "Trials/job: ${TRIALS}"

FOUND_SEEDS=0

for split in "${EVALUATION_SPLITS[@]}"; do
    split_output_dir="${OUTPUT_BASE}/${split}"
    if [ -d "${split_output_dir}" ]; then
        for seed_dir in "${split_output_dir}"/seed_*; do
            if [ -d "${seed_dir}" ]; then
                seed="$(basename "${seed_dir}" | sed 's/seed_//')"
                FOUND_SEEDS=1
                echo "Running seed: ${seed}"
                for dataset in "${DATASETS[@]}"; do
                    log_prefix="${LOGS_DIR}/optuna_resume_seed_${seed}_${split}_${dataset}"
                    echo "  -> split=${split} dataset=${dataset}"
                    python3 -m floatsom.benchmarks.optuna.run_optuna \
                        --mode full \
                        --config full \
                        --topology "${TOPOLOGIES[@]}" \
                        --sampling-methods "${SAMPLING_METHODS[@]}" \
                        --random-target-proportion "${RANDOM_TARGET_PROPORTION}" \
                        --processing-methods batch \
                        --batch-modes full_batch \
                        --datasets "${dataset}" \
                        --objectives "${OBJECTIVES}" \
                        --output-dir "${split_output_dir}" \
                        --trials "${TRIALS}" \
                        --seed "${seed}" \
                        --evaluation-split "${split}" \
                        --resume \
                        >"${log_prefix}.out" \
                        2>"${log_prefix}.err"
                done
            fi
        done
    fi
done

if [ "${FOUND_SEEDS}" -eq 0 ]; then
    echo "No split seed runs discovered. Falling back to legacy seed layout."
    for seed_dir in "${OUTPUT_BASE}"/seed_*; do
        if [ -d "${seed_dir}" ]; then
            seed="$(basename "${seed_dir}" | sed 's/seed_//')"
            FOUND_SEEDS=1
            echo "Running seed: ${seed}"
            for dataset in "${DATASETS[@]}"; do
                log_prefix="${LOGS_DIR}/optuna_resume_seed_${seed}_both_${dataset}"
                echo "  -> split=both dataset=${dataset}"
                python3 -m floatsom.benchmarks.optuna.run_optuna \
                    --mode full \
                    --config full \
                    --topology "${TOPOLOGIES[@]}" \
                    --sampling-methods "${SAMPLING_METHODS[@]}" \
                    --random-target-proportion "${RANDOM_TARGET_PROPORTION}" \
                    --processing-methods batch \
                    --batch-modes full_batch \
                    --datasets "${dataset}" \
                    --objectives "${OBJECTIVES}" \
                    --output-dir "${OUTPUT_BASE}" \
                    --trials "${TRIALS}" \
                    --seed "${seed}" \
                    --evaluation-split both \
                    --resume \
                    >"${log_prefix}.out" \
                    2>"${log_prefix}.err"
            done
        fi
    done
fi

if [ "${FOUND_SEEDS}" -eq 0 ]; then
    echo "No matching seed directories found under: ${OUTPUT_BASE}"
    exit 1
fi

echo "Local resume completed successfully"
