#!/bin/bash

# Quick smoke runner for Optuna FloatSOM benchmarks.
# Runs each benchmark dataset with a tiny trial budget to validate end-to-end wiring.

set -u

TRIALS=2
SEED=42
CONFIG="full"
EVALUATION_SPLIT="both"
SKLEARN_DATA_HOME="/g/data/eu59/SIFEAN/sfa/sklearn_data"
OUTPUT_DIR="/g/data/eu59/SIFEAN/sfa/noninf_optuna_smoke_$(date +%d%m%Y_%H%M%S)"
TOPOLOGIES=("hexagonal")
SAMPLING_METHODS=("full")
PROCESSING_METHODS=("batch")
DATASETS=(
    "swiss_roll" "moons" "circles" "blobs" "s_curve"
    "breast_cancer" "wine" "iris" "digits" "olivetti_faces"
    "diabetes" "california_housing" "covertype" "kddcup99"
)

usage() {
    cat << EOF
Usage: $(basename "$0") [options]

Options:
  --output-dir DIR       Output directory root (default: timestamped /g/data/eu59/SIFEAN/sfa/noninf_optuna_smoke_*)
  --trials N             Trials per dataset (default: 2)
  --seed N               Random seed (default: 42)
  --config NAME          run_optuna config preset: development|full (default: full)
  --split NAME           Evaluation split: both|holdout|train (default: both)
  --scikit-learn-data-home DIR  Sets SCIKIT_LEARN_DATA (default: /g/data/eu59/SIFEAN/sfa/sklearn_data)
  --datasets LIST        Comma-separated datasets to run (default: all 14 benchmark datasets)
  --topologies LIST      Comma-separated topologies (default: hexagonal)
  --sampling LIST        Comma-separated sampling methods (default: full)
  --processing LIST      Comma-separated processing methods (default: batch)
  -h, --help             Show this help
EOF
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
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --trials)
                TRIALS="$2"
                shift 2
                ;;
            --seed)
                SEED="$2"
                shift 2
                ;;
            --config)
                CONFIG="$2"
                shift 2
                ;;
            --split)
                EVALUATION_SPLIT="$2"
                shift 2
                ;;
            --scikit-learn-data-home)
                SKLEARN_DATA_HOME="$2"
                shift 2
                ;;
            --datasets)
                parse_csv_arg "$2" DATASETS
                shift 2
                ;;
            --topologies)
                parse_csv_arg "$2" TOPOLOGIES
                shift 2
                ;;
            --sampling)
                parse_csv_arg "$2" SAMPLING_METHODS
                shift 2
                ;;
            --processing)
                parse_csv_arg "$2" PROCESSING_METHODS
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
    if ! [[ "$TRIALS" =~ ^[0-9]+$ ]] || [ "$TRIALS" -le 0 ]; then
        echo "Invalid --trials value: $TRIALS"
        exit 1
    fi
    if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
        echo "Invalid --seed value: $SEED"
        exit 1
    fi
}

parse_args "$@"
validate_args

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

mkdir -p "$OUTPUT_DIR"
export SCIKIT_LEARN_DATA="$SKLEARN_DATA_HOME"

echo "Running mock Optuna smoke sweep"
echo "Repo root: $REPO_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Trials per dataset: $TRIALS"
echo "Seed: $SEED"
echo "Config: $CONFIG"
echo "Evaluation split: $EVALUATION_SPLIT"
echo "Topologies: ${TOPOLOGIES[*]}"
echo "Sampling methods: ${SAMPLING_METHODS[*]}"
echo "Processing methods: ${PROCESSING_METHODS[*]}"
echo "Datasets (${#DATASETS[@]}): ${DATASETS[*]}"
echo

FAILURES=()
for dataset in "${DATASETS[@]}"; do
    dataset_out="${OUTPUT_DIR}/${dataset}"
    mkdir -p "$dataset_out"

    echo "=== Dataset: $dataset ==="
    python3 -m floatsom.benchmarks.optuna.run_optuna \
        --mode full \
        --config "$CONFIG" \
        --topology "${TOPOLOGIES[@]}" \
        --sampling-methods "${SAMPLING_METHODS[@]}" \
        --processing-methods "${PROCESSING_METHODS[@]}" \
        --datasets "$dataset" \
        --objectives quantization_error \
        --output-dir "$dataset_out" \
        --trials "$TRIALS" \
        --seed "$SEED" \
        --evaluation-split "$EVALUATION_SPLIT"
    rc=$?

    if [ "$rc" -ne 0 ]; then
        FAILURES+=("$dataset")
        echo "FAILED: $dataset (exit=$rc)"
    else
        echo "OK: $dataset"
    fi
    echo
done

if [ "${#FAILURES[@]}" -gt 0 ]; then
    echo "Smoke sweep completed with failures (${#FAILURES[@]}): ${FAILURES[*]}"
    exit 1
fi

echo "Smoke sweep completed successfully for all datasets."
exit 0
