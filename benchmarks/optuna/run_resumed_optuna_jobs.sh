#!/bin/bash

# Script to resume Optuna benchmarks for existing seeds.
# Supports split-aware and legacy output layouts.
# Optional filters:
#   --seeds 42,99
#   --topologies rng (alias: --modes)
#   --sampling-methods full,random
#   --splits both

set -euo pipefail

# Defaults
TODAY_STAMP="$(date +%d%m%Y)"
OUTPUT_DIR="/g/data/eu59/SIFEAN/sfa/noninf_optuna_benchmarks_${TODAY_STAMP}"
CONFIG="full"
TRIALS=200
LOGS_DIR="gadi_optunalogs"
SKLEARN_DATA_HOME="/g/data/eu59/SIFEAN/sfa/sklearn_data"
EVALUATION_SPLITS=("both")
TOPOLOGIES=("hexagonal" "mst" "rng")
SAMPLING_METHODS=("full" "random")
RANDOM_TARGET_PROPORTION=0.3
DATASETS=(
    "swiss_roll" "moons" "circles" "blobs" "s_curve"
    "breast_cancer" "wine" "iris" "digits" "olivetti_faces"
    "diabetes" "california_housing" "covertype" "kddcup99"
)
SEED_FILTER=()
SUBMITTED_JOBS=0

usage() {
    cat << EOF
Usage: $(basename "$0") [options]

Options:
  --output-dir DIR       Output directory containing previous runs
  --config NAME          Config passed to run_optuna (default: full)
  --trials N             Number of trials per job (default: 200)
  --seeds LIST           Comma-separated seeds to resume (default: all discovered seed_* dirs)
  --topologies LIST      Comma-separated topologies to run (e.g. rng or hexagonal,mst,rng)
  --modes LIST           Alias of --topologies
  --sampling-methods LIST
                         Comma-separated sampling methods (default: full,random)
  --random-target-proportion FLOAT
                         Random sampling target proportion (default: 0.3)
  --splits LIST          Comma-separated split dirs to scan (default: both)
  --datasets LIST        Comma-separated datasets (default: 14 standard datasets)
  --scikit-learn-data-home DIR  Sets SCIKIT_LEARN_DATA inside PBS jobs (default: /g/data/eu59/SIFEAN/sfa/sklearn_data)
  -h, --help             Show this help
EOF
}

contains_value() {
    local needle="$1"
    shift
    local value
    for value in "$@"; do
        if [ "$value" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

parse_csv_arg() {
    local raw="$1"
    local -n output_ref="$2"
    local item

    output_ref=()
    raw="${raw//,/ }"
    for item in $raw; do
        output_ref+=("$item")
    done
}

seed_selected() {
    local seed="$1"
    local selected_seed

    if [ "${#SEED_FILTER[@]}" -eq 0 ]; then
        return 0
    fi

    for selected_seed in "${SEED_FILTER[@]}"; do
        if [ "$selected_seed" = "$seed" ]; then
            return 0
        fi
    done

    return 1
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --output-dir)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --output-dir"
                    exit 1
                fi
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --config)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --config"
                    exit 1
                fi
                CONFIG="$2"
                shift 2
                ;;
            --trials)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --trials"
                    exit 1
                fi
                TRIALS="$2"
                shift 2
                ;;
            --seeds)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --seeds"
                    exit 1
                fi
                parse_csv_arg "$2" SEED_FILTER
                shift 2
                ;;
            --topologies|--modes)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --topologies/--modes"
                    exit 1
                fi
                parse_csv_arg "$2" TOPOLOGIES
                shift 2
                ;;
            --sampling-methods)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --sampling-methods"
                    exit 1
                fi
                parse_csv_arg "$2" SAMPLING_METHODS
                shift 2
                ;;
            --random-target-proportion)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --random-target-proportion"
                    exit 1
                fi
                RANDOM_TARGET_PROPORTION="$2"
                shift 2
                ;;
            --splits)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --splits"
                    exit 1
                fi
                parse_csv_arg "$2" EVALUATION_SPLITS
                shift 2
                ;;
            --datasets)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --datasets"
                    exit 1
                fi
                parse_csv_arg "$2" DATASETS
                shift 2
                ;;
            --scikit-learn-data-home)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --scikit-learn-data-home"
                    exit 1
                fi
                SKLEARN_DATA_HOME="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

validate_args() {
    local topology
    local split
    local sampling_method
    local i
    local normalized_seeds=()
    local seed
    local valid_topologies=("hexagonal" "grid" "mst" "rng")
    local valid_splits=("holdout" "train" "both")
    local valid_sampling_methods=("full" "random" "hdsssom")

    if ! [[ "$TRIALS" =~ ^[0-9]+$ ]]; then
        echo "Invalid --trials value: $TRIALS"
        exit 1
    fi

    if ! [[ "$RANDOM_TARGET_PROPORTION" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
        echo "Invalid --random-target-proportion value: $RANDOM_TARGET_PROPORTION"
        exit 1
    fi

    if [ "${#TOPOLOGIES[@]}" -eq 0 ]; then
        echo "--topologies/--modes cannot be empty"
        exit 1
    fi
    for topology in "${TOPOLOGIES[@]}"; do
        if ! contains_value "$topology" "${valid_topologies[@]}"; then
            echo "Invalid topology '$topology'. Valid options: ${valid_topologies[*]}"
            exit 1
        fi
    done

    if [ "${#SAMPLING_METHODS[@]}" -eq 0 ]; then
        echo "--sampling-methods cannot be empty"
        exit 1
    fi
    for sampling_method in "${SAMPLING_METHODS[@]}"; do
        if ! contains_value "$sampling_method" "${valid_sampling_methods[@]}"; then
            echo "Invalid sampling method '$sampling_method'. Valid options: ${valid_sampling_methods[*]}"
            exit 1
        fi
    done

    if [ "${#EVALUATION_SPLITS[@]}" -eq 0 ]; then
        echo "--splits cannot be empty"
        exit 1
    fi
    for split in "${EVALUATION_SPLITS[@]}"; do
        if ! contains_value "$split" "${valid_splits[@]}"; then
            echo "Invalid split '$split'. Valid options: ${valid_splits[*]}"
            exit 1
        fi
    done

    for i in "${!SEED_FILTER[@]}"; do
        seed="${SEED_FILTER[$i]}"
        seed="${seed#seed_}"
        if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
            echo "Invalid seed '$seed'. Use integers (e.g. 42) or seed_<int>."
            exit 1
        fi
        normalized_seeds+=("$seed")
    done
    SEED_FILTER=("${normalized_seeds[@]}")
}

submit_resume_job() {
    local seed="$1"
    local split="$2"
    local dataset="$3"
    local split_output_dir="$4"
    local objectives
    local pbs_script

    echo "Found seed $seed for split '$split' - submitting job with --resume flag"

    objectives="quantization_error"

    local dataset_short="${dataset//_/}"
    dataset_short="${dataset_short:0:8}"
    local split_short="b"
    if [ "$split" = "holdout" ]; then
        split_short="h"
    elif [ "$split" = "train" ]; then
        split_short="t"
    fi

    pbs_script="optuna_resume_seed_${seed}_${split}_${dataset}.pbs.sh"

    cat > "$pbs_script" << EOF
#PBS -N optuna_r_${seed}_${split_short}${dataset_short}
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
#PBS -o ${LOGS_DIR}/optuna_resume_${seed}_${split}_${dataset}.out
#PBS -e ${LOGS_DIR}/optuna_resume_${seed}_${split}_${dataset}.err

module use /g/data/dk92/apps/Modules/modulefiles/
module load rapids/25.06
cd /g/data/eu59/SIFEAN/sfa/
export SCIKIT_LEARN_DATA="${SKLEARN_DATA_HOME}"

python3 -m floatsom.benchmarks.optuna.run_optuna \\
--mode full \\
--config $CONFIG \\
--topology ${TOPOLOGIES[*]} \\
--sampling-methods ${SAMPLING_METHODS[*]} \\
--random-target-proportion ${RANDOM_TARGET_PROPORTION} \\
--processing-methods batch \\
--batch-modes full_batch \\
--datasets ${dataset} \\
--objectives ${objectives} \\
--output-dir $split_output_dir \\
--trials $TRIALS \\
--seed $seed \\
--evaluation-split $split \\
--resume
EOF

    qsub "$pbs_script"
    rm "$pbs_script"
    sleep 0.5

    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
}

parse_args "$@"
validate_args

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

echo "Checking for existing seeds in: $OUTPUT_DIR"
echo "Topologies: ${TOPOLOGIES[*]}"
echo "Sampling methods: ${SAMPLING_METHODS[*]}"
echo "Random target proportion: ${RANDOM_TARGET_PROPORTION}"
echo "Splits: ${EVALUATION_SPLITS[*]}"
echo "Datasets: ${DATASETS[*]}"
if [ "${#SEED_FILTER[@]}" -gt 0 ]; then
    echo "Seed filter: ${SEED_FILTER[*]}"
else
    echo "Seed filter: all discovered seeds"
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Output directory not found: $OUTPUT_DIR"
    echo "Please run initial jobs first"
    exit 1
fi

FOUND_SPLIT_SEEDS=0
for split in "${EVALUATION_SPLITS[@]}"; do
    split_output_dir="${OUTPUT_DIR}/${split}"
    if [ -d "$split_output_dir" ]; then
        for seed_dir in "$split_output_dir"/seed_*; do
            if [ -d "$seed_dir" ]; then
                seed=$(basename "$seed_dir" | sed 's/seed_//')
                if seed_selected "$seed"; then
                    for dataset in "${DATASETS[@]}"; do
                        submit_resume_job "$seed" "$split" "$dataset" "$split_output_dir"
                    done
                    FOUND_SPLIT_SEEDS=1
                fi
            fi
        done
    fi
done

# Backward compatibility with legacy unsplit layout.
if [ "$FOUND_SPLIT_SEEDS" -eq 0 ]; then
    echo "No split seed runs discovered for selected filters. Falling back to legacy seed layout."
    for seed_dir in "$OUTPUT_DIR"/seed_*; do
        if [ -d "$seed_dir" ]; then
            seed=$(basename "$seed_dir" | sed 's/seed_//')
            if seed_selected "$seed"; then
                for dataset in "${DATASETS[@]}"; do
                    submit_resume_job "$seed" "both" "$dataset" "$OUTPUT_DIR"
                done
            fi
        fi
    done
fi

if [ "$SUBMITTED_JOBS" -gt 0 ]; then
    echo "All resume jobs submitted ($SUBMITTED_JOBS total)"
else
    echo "No matching seed directories found to resume under: $OUTPUT_DIR"
fi
