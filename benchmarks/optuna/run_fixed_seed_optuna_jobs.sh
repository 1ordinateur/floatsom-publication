#!/bin/bash

# Submit fresh Optuna jobs for an explicit list of seeds.
# This does not require existing seed_* directories (unlike the resume script).

set -euo pipefail

# Defaults
OUTPUT_DIR="/g/data/eu59/SIFEAN/sfa/noninf_optuna_benchmarks"
CONFIG="full"
TRIALS=200
LOGS_DIR="gadi_optunalogs"
EVALUATION_SPLITS=("both")
TOPOLOGIES=("hexagonal" "mst" "rng")
SEEDS=(1160 7259 24579 27759 27937)
DATASETS=(
    "swiss_roll" "moons" "circles" "blobs" "s_curve"
    "breast_cancer" "wine" "iris" "digits" "olivetti_faces"
    "diabetes" "california_housing" "covertype" "kddcup99"
)
SUBMITTED_JOBS=0

usage() {
    cat << EOF
Usage: $(basename "$0") [options]

Options:
  --seeds LIST           Comma-separated seeds (default: 1160,7259,24579,27759,27937)
  --topologies LIST      Comma-separated topologies (default: hexagonal,mst,rng)
  --modes LIST           Alias of --topologies
  --topology NAME        Convenience alias for a single topology
  --output-dir DIR       Base output directory (default: $OUTPUT_DIR)
  --config NAME          Config passed to run_optuna (default: $CONFIG)
  --trials N             Number of trials per job (default: $TRIALS)
  --splits LIST          Comma-separated splits (default: both)
  --datasets LIST        Comma-separated datasets (default: 14 standard datasets)
  --logs-dir DIR         Log directory for PBS stdout/stderr (default: $LOGS_DIR)
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

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --seeds)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --seeds"
                    exit 1
                fi
                parse_csv_arg "$2" SEEDS
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
            --topology)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --topology"
                    exit 1
                fi
                TOPOLOGIES=("$2")
                shift 2
                ;;
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
            --logs-dir)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --logs-dir"
                    exit 1
                fi
                LOGS_DIR="$2"
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
    local seed
    local split
    local topology
    local valid_topologies=("hexagonal" "grid" "mst" "rng")
    local valid_splits=("holdout" "train" "both")

    for seed in "${SEEDS[@]}"; do
        if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
            echo "Invalid seed '$seed'. Use integers only."
            exit 1
        fi
    done

    if ! [[ "$TRIALS" =~ ^[0-9]+$ ]]; then
        echo "Invalid --trials value: $TRIALS"
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
}

submit_job() {
    local seed="$1"
    local split="$2"
    local topology="$3"
    local dataset="$4"
    local split_output_dir="$5"
    local objectives
    local pbs_script

    objectives="quantization_error"

    dataset_short="${dataset//_/}"
    dataset_short="${dataset_short:0:8}"
    split_short="b"
    if [ "$split" = "holdout" ]; then
        split_short="h"
    elif [ "$split" = "train" ]; then
        split_short="t"
    fi

    pbs_script="optuna_seed_${seed}_${split}_${topology}_${dataset}.pbs.sh"
    cat > "$pbs_script" << EOF
#PBS -N optuna_${seed}_${split_short}${dataset_short}_${topology}
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
#PBS -o ${LOGS_DIR}/optuna_seed_${seed}_${split}_${topology}_${dataset}.out
#PBS -e ${LOGS_DIR}/optuna_seed_${seed}_${split}_${topology}_${dataset}.err

module use /g/data/dk92/apps/Modules/modulefiles/
module load rapids/25.06
cd /g/data/eu59/SIFEAN/sfa/

python3 -m floatsom_benchmarks.optuna.run_optuna \\
--mode full \\
--config ${CONFIG} \\
--topology ${topology} \\
--sampling-methods random \\
--processing-methods batch \\
--datasets ${dataset} \\
--objectives ${objectives} \\
--output-dir ${split_output_dir} \\
--trials ${TRIALS} \\
--seed ${seed} \\
--evaluation-split ${split}
EOF

    qsub "$pbs_script"
    rm "$pbs_script"
    sleep 0.5

    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
}

parse_args "$@"
validate_args

mkdir -p "$LOGS_DIR"

echo "Submitting fresh Optuna jobs"
echo "Seeds: ${SEEDS[*]}"
echo "Topologies: ${TOPOLOGIES[*]}"
echo "Splits: ${EVALUATION_SPLITS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Trials/job: $TRIALS"
echo "Config: $CONFIG"
echo "Output base: $OUTPUT_DIR"
echo "Logs: $LOGS_DIR"

for seed in "${SEEDS[@]}"; do
    echo "Submitting jobs for seed: $seed"
    for split in "${EVALUATION_SPLITS[@]}"; do
        split_output_dir="${OUTPUT_DIR}/${split}"
        mkdir -p "$split_output_dir"
        for topology in "${TOPOLOGIES[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                submit_job "$seed" "$split" "$topology" "$dataset" "$split_output_dir"
            done
        done
    done
done

echo "All jobs submitted successfully ($SUBMITTED_JOBS total)"
