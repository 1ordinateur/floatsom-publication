#!/bin/bash

set -euo pipefail

REPO_ROOT="/home/givanna/Documents/sifean_benchmark/sifean"
cd "${REPO_ROOT}"

OUTPUT_BASE="/home/givanna/Documents/sifean_benchmark/optuna_local_single_gpu_benchmarks_run2"
EVALUATION_SPLITS=("both")
NUM_RANDOM_SEEDS=10
TRIALS=200
OBJECTIVES="quantization_error"
RANDOM_TARGET_PROPORTION=0.3
LOCAL_GPU_ID="${LOCAL_GPU_ID:-0}"
TOPOLOGIES=("hexagonal" "mst" "rng")
DATASETS=(
    "swiss_roll" "moons" "circles" "blobs" "s_curve"
    "breast_cancer" "wine" "iris" "digits" "olivetti_faces"
    "diabetes" "california_housing" "covertype" "kddcup99"
)

if ! command -v shuf >/dev/null 2>&1; then
    echo "Error: 'shuf' is required to generate random seeds."
    exit 1
fi

mapfile -t SEEDS < <(shuf -i 1000-99999 -n "${NUM_RANDOM_SEEDS}")

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOGS_DIR="/home/givanna/Documents/sifean_benchmark/optuna_local_single_gpu_benchmarks_run2/logs/local_optunalogs_${RUN_TAG}"
mkdir -p "${LOGS_DIR}"

export CUDA_VISIBLE_DEVICES="${LOCAL_GPU_ID}"

echo "Running local Optuna benchmarks (single GPU, sequential)"
echo "Repo root: ${REPO_ROOT}"
echo "Output base: ${OUTPUT_BASE}"
echo "Logs: ${LOGS_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Seeds: ${SEEDS[*]}"
echo "Splits: ${EVALUATION_SPLITS[*]}"
echo "Topologies: ${TOPOLOGIES[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Trials/job: ${TRIALS}"
echo "Sampling methods: random"
echo "Random target proportion: ${RANDOM_TARGET_PROPORTION}"

for seed in "${SEEDS[@]}"; do
    echo "Running seed: ${seed}"
    for split in "${EVALUATION_SPLITS[@]}"; do
        split_output_dir="${OUTPUT_BASE}/${split}"
        mkdir -p "${split_output_dir}"
        for dataset in "${DATASETS[@]}"; do
            log_prefix="${LOGS_DIR}/optuna_seed_${seed}_${split}_${dataset}"
            echo "  -> split=${split} dataset=${dataset}"
            python3 -m floatsom_benchmarks.optuna.run_optuna \
                --mode full \
                --config full \
                --topology "${TOPOLOGIES[@]}" \
                --sampling-methods random \
                --random-target-proportion "${RANDOM_TARGET_PROPORTION}" \
                --processing-methods batch \
                --batch-modes full_batch \
                --datasets "${dataset}" \
                --objectives "${OBJECTIVES}" \
                --output-dir "${split_output_dir}" \
                --trials "${TRIALS}" \
                --seed "${seed}" \
                --evaluation-split "${split}" \
                >"${log_prefix}.out" \
                2>"${log_prefix}.err"
        done
    done
done

echo "Local run completed successfully"
    