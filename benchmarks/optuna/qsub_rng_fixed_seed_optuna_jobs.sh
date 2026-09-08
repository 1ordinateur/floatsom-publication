#!/bin/bash

# Dedicated RNG-only Optuna submitter for fixed seeds.
# Creates date-stamped PBS files and logs for traceability.

set -euo pipefail

OUTPUT_ROOT="/g/data/eu59/SIFEAN/sfa/optuna_benchmarks"
CONFIG="full"
TRIALS=200
EVALUATION_SPLITS=("both")
SEEDS=(1160 7259 24579 27759 27937)
DATASETS=()
LOGS_ROOT="gadi_optunalogs"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
SUBMITTED_JOBS=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMBINE_SCRIPT="${SCRIPT_DIR}/combine_rng_optuna_dataset_shards.sh"

usage() {
    cat << EOF
Usage: $(basename "$0") [options]

Options:
  --seeds LIST           Comma-separated seeds (default: 1160,7259,24579,27759,27937)
  --datasets LIST        Comma-separated datasets (default depends on --config)
  --output-dir DIR       Output root directory; run tag is appended (default: $OUTPUT_ROOT)
  --config NAME          Config passed to run_optuna (default: $CONFIG)
  --trials N             Number of trials per job (default: $TRIALS)
  --splits LIST          Comma-separated splits (default: both)
  --logs-root DIR        Parent logs directory (default: $LOGS_ROOT)
  --tag VALUE            Explicit run tag (default: current timestamp YYYYmmdd_HHMMSS)
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

set_default_datasets() {
    case "$CONFIG" in
        full)
            DATASETS=(
                swiss_roll moons circles blobs s_curve
                breast_cancer wine iris digits olivetti_faces
                diabetes california_housing covertype kddcup99
            )
            ;;
        development)
            DATASETS=(swiss_roll moons blobs)
            ;;
        *)
            echo "Unsupported --config value: $CONFIG (expected: development|full)"
            exit 1
            ;;
    esac
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
            --output-dir)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --output-dir"
                    exit 1
                fi
                OUTPUT_ROOT="$2"
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
            --logs-root)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --logs-root"
                    exit 1
                fi
                LOGS_ROOT="$2"
                shift 2
                ;;
            --tag)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --tag"
                    exit 1
                fi
                RUN_TAG="$2"
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
    local dataset
    local valid_datasets=()
    local valid_splits=("holdout" "train" "both")

    if [ "${#SEEDS[@]}" -eq 0 ]; then
        echo "--seeds cannot be empty"
        exit 1
    fi

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

    if ! contains_value "$CONFIG" "development" "full"; then
        echo "Invalid --config value '$CONFIG'. Valid options: development full"
        exit 1
    fi

    if [ "$CONFIG" = "full" ]; then
        valid_datasets=(
            swiss_roll moons circles blobs s_curve
            breast_cancer wine iris digits olivetti_faces
            diabetes california_housing covertype kddcup99
        )
    else
        valid_datasets=(swiss_roll moons blobs)
    fi

    if [ "${#DATASETS[@]}" -eq 0 ]; then
        echo "--datasets cannot be empty"
        exit 1
    fi
    for dataset in "${DATASETS[@]}"; do
        if [[ "$dataset" =~ [[:space:]] ]]; then
            echo "Invalid dataset '$dataset'. Use comma-separated names without spaces."
            exit 1
        fi
        if ! contains_value "$dataset" "${valid_datasets[@]}"; then
            echo "Invalid dataset '$dataset' for config '$CONFIG'."
            echo "Valid datasets: ${valid_datasets[*]}"
            exit 1
        fi
    done
}

dataset_short_name() {
    local dataset="$1"
    local short
    short="$(printf '%s' "$dataset" | tr -cd '[:alnum:]' | cut -c1-2)"
    if [ -z "$short" ]; then
        short="ds"
    fi
    echo "$short"
}

submit_job() {
    local seed="$1"
    local split="$2"
    local split_short="$3"
    local dataset="$4"
    local shard_output_dir="$5"
    local logs_dir="$6"
    local pbs_dir="$7"
    local objectives="quantization_error"
    local pbs_script
    local job_tag
    local dataset_short
    local job_name

    # Keep job name compact for PBS implementations with strict limits.
    job_tag="$(date +%d%H)"
    dataset_short="$(dataset_short_name "$dataset")"
    job_name="or${seed}${split_short}${dataset_short}${job_tag}"

    pbs_script="${pbs_dir}/optuna_rng_seed_${seed}_${split}_${dataset}_${RUN_TAG}.pbs.sh"
    cat > "$pbs_script" << EOF
#PBS -N ${job_name}
#PBS -q gpuvolta
#PBS -P eu59
#PBS -l ncpus=12
#PBS -l mem=96GB
#PBS -l ngpus=1
#PBS -l wd
#PBS -l walltime=47:00:00
#PBS -l storage=gdata/eu59+gdata/dk92+scratch/eu59
#PBS -l jobfs=400GB
#PBS -M tony.xu@anu.edu.au
#PBS -m abe
#PBS -o ${logs_dir}/optuna_rng_seed_${seed}_${split}_${dataset}_${RUN_TAG}.out
#PBS -e ${logs_dir}/optuna_rng_seed_${seed}_${split}_${dataset}_${RUN_TAG}.err

module use /g/data/dk92/apps/Modules/modulefiles/
module load rapids/25.06
cd /g/data/eu59/SIFEAN/sfa/

python3 -m floatsom_benchmarks.optuna.run_optuna \\
--mode full \\
--config ${CONFIG} \\
--topology rng \\
--sampling-methods random \\
--processing-methods batch \\
--datasets ${dataset} \\
--objectives ${objectives} \\
--output-dir ${shard_output_dir} \\
--trials ${TRIALS} \\
--seed ${seed} \\
--evaluation-split ${split}
EOF

    qsub "$pbs_script"
    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
    sleep 0.5
}

parse_args "$@"
if [ "${#DATASETS[@]}" -eq 0 ]; then
    set_default_datasets
fi
validate_args

LOGS_DIR="${LOGS_ROOT}/rng_${RUN_TAG}"
PBS_DIR="${LOGS_DIR}/pbs"
OUTPUT_DIR="${OUTPUT_ROOT}/rng_${RUN_TAG}"
mkdir -p "$LOGS_DIR" "$PBS_DIR"

echo "Submitting dedicated RNG-only Optuna jobs"
echo "Run tag: $RUN_TAG"
echo "Seeds: ${SEEDS[*]}"
echo "Splits: ${EVALUATION_SPLITS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Trials/job: $TRIALS"
echo "Config: $CONFIG"
echo "Output root: $OUTPUT_ROOT"
echo "Output run dir: $OUTPUT_DIR"
echo "Shard layout: <run>/<split>/_dataset_shards/<dataset>/seed_<seed>/..."
echo "Logs dir: $LOGS_DIR"
echo "PBS dir: $PBS_DIR"

for seed in "${SEEDS[@]}"; do
    echo "Submitting jobs for seed: $seed"
    for split in "${EVALUATION_SPLITS[@]}"; do
        split_output_dir="${OUTPUT_DIR}/${split}"
        shard_root="${split_output_dir}/_dataset_shards"
        mkdir -p "$shard_root"
        split_short="b"
        if [ "$split" = "holdout" ]; then
            split_short="h"
        elif [ "$split" = "train" ]; then
            split_short="t"
        fi
        for dataset in "${DATASETS[@]}"; do
            dataset_shard_dir="${shard_root}/${dataset}"
            mkdir -p "$dataset_shard_dir"
            submit_job "$seed" "$split" "$split_short" "$dataset" "$dataset_shard_dir" "$LOGS_DIR" "$PBS_DIR"
        done
    done
done

echo "All jobs submitted successfully ($SUBMITTED_JOBS total)"
echo "When shard jobs complete, combine them locally with:"
echo "  ${COMBINE_SCRIPT} --output-dir ${OUTPUT_DIR} --seeds $(IFS=,; echo "${SEEDS[*]}") --splits $(IFS=,; echo "${EVALUATION_SPLITS[*]}") --datasets $(IFS=,; echo "${DATASETS[*]}")"
