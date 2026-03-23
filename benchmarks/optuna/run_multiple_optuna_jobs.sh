#!/bin/bash

set -euo pipefail

# Logs directory
LOGS_DIR="gadi_optunalogs"
OUTPUT_BASE="/g/data/eu59/SIFEAN/sfa/noninf_optuna_benchmarks"
EVALUATION_SPLITS=("both")
NUM_RANDOM_SEEDS=10
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

# Create logs directory if it doesn't exist
mkdir -p $LOGS_DIR

echo "Running Optuna benchmarks with random seeds: ${SEEDS[@]}"
echo "Logs will be saved to: $LOGS_DIR"
echo "Evaluation splits: ${EVALUATION_SPLITS[@]}"
echo "Topologies: ${TOPOLOGIES[@]}"
echo "Datasets: ${DATASETS[@]}"

# Submit PBS jobs with different seeds and evaluation splits
for seed in "${SEEDS[@]}"; do
    echo "Submitting jobs for seed: $seed"

    for split in "${EVALUATION_SPLITS[@]}"; do
        echo "  -> split: $split"

        objectives="quantization_error"

        SPLIT_OUTPUT_DIR="${OUTPUT_BASE}/${split}"
        mkdir -p "$SPLIT_OUTPUT_DIR"

        for dataset in "${DATASETS[@]}"; do
            dataset_short="${dataset//_/}"
            dataset_short="${dataset_short:0:8}"
            split_short="b"
            if [ "$split" = "holdout" ]; then
                split_short="h"
            elif [ "$split" = "train" ]; then
                split_short="t"
            fi

            PBS_SCRIPT="optuna_job_seed_${seed}_${split}_${dataset}.pbs.sh"

            cat > $PBS_SCRIPT << EOF
#PBS -N optuna_${seed}_${split_short}${dataset_short}
#PBS -q gpuvolta
#PBS -P eu59
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l ngpus=1
#PBS -l wd
#PBS -l walltime=06:00:00
#PBS -l jobfs=400GB
#PBS -l storage=gdata/eu59+gdata/dk92
#PBS -M tony.xu@anu.edu.au
#PBS -m abe
#PBS -o ${LOGS_DIR}/optuna_seed_${seed}_${split}_${dataset}.out
#PBS -e ${LOGS_DIR}/optuna_seed_${seed}_${split}_${dataset}.err

module use /g/data/dk92/apps/Modules/modulefiles/; module load rapids/25.06; cd /g/data/eu59/SIFEAN/sfa/

python3 -m floatsom.benchmarks.optuna.run_optuna \\
--mode full \\
--config full \\
--topology ${TOPOLOGIES[*]} \\
--sampling-methods random \\
--random-target-proportion 0.3 \\
--processing-methods batch \\
--batch-modes full_batch \\
--datasets ${dataset} \\
--objectives ${objectives} \\
--output-dir $SPLIT_OUTPUT_DIR \\
--trials 200 \\
--seed $seed \\
--evaluation-split $split
EOF

            # Submit the job
            qsub $PBS_SCRIPT

            # Clean up temporary script
            rm $PBS_SCRIPT

            # Small delay between submissions
            sleep 0.5
        done
    done

done

echo "All jobs submitted successfully"
