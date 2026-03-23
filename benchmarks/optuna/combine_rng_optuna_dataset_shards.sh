#!/bin/bash

# Combine dataset-sharded RNG Optuna outputs into canonical split/seed layout.
# This script is intended to run on a login node after PBS jobs finish.

set -euo pipefail

OUTPUT_DIR=""
EVALUATION_SPLITS=("both")
SEEDS=(1160 7259 24579 27759 27937)
DATASETS=()
MOVED_ITEMS=0
SKIPPED_IDENTICAL_FILES=0
PROCESSED_SOURCE_SEED_DIRS=0
COMBINED_SEED_DIRS=0

usage() {
    cat << EOF
Usage: $(basename "$0") --output-dir DIR [options]

Options:
  --output-dir DIR       RNG run directory produced by qsub submitter (required)
  --seeds LIST           Comma-separated seeds to combine (default: 1160,7259,24579,27759,27937)
  --splits LIST          Comma-separated splits (default: both)
  --datasets LIST        Optional comma-separated dataset filter; default auto-discovers shard datasets
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
            --output-dir)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --output-dir"
                    exit 1
                fi
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --seeds)
                if [ "$#" -lt 2 ]; then
                    echo "Missing value for --seeds"
                    exit 1
                fi
                parse_csv_arg "$2" SEEDS
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
    local valid_splits=("holdout" "train" "both")

    if [ -z "$OUTPUT_DIR" ]; then
        echo "--output-dir is required"
        exit 1
    fi
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Output directory not found: $OUTPUT_DIR"
        exit 1
    fi

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

    for dataset in "${DATASETS[@]}"; do
        if [[ "$dataset" =~ [[:space:]] ]]; then
            echo "Invalid dataset '$dataset'. Use comma-separated names without spaces."
            exit 1
        fi
    done
}

discover_datasets_for_split() {
    local split="$1"
    local -n output_ref="$2"
    local shard_root="${OUTPUT_DIR}/${split}/_dataset_shards"
    local dataset_dir

    output_ref=()
    if [ ! -d "$shard_root" ]; then
        return
    fi

    for dataset_dir in "$shard_root"/*; do
        if [ -d "$dataset_dir" ]; then
            output_ref+=("$(basename "$dataset_dir")")
        fi
    done
}

merge_source_seed_dir() {
    local source_seed_dir="$1"
    local target_seed_dir="$2"
    local split="$3"
    local seed="$4"
    local dataset="$5"
    local entry
    local base_name
    local target_path
    local entries=()

    shopt -s nullglob dotglob
    entries=("$source_seed_dir"/*)
    shopt -u nullglob dotglob

    if [ "${#entries[@]}" -eq 0 ]; then
        return
    fi

    for entry in "${entries[@]}"; do
        base_name="$(basename "$entry")"
        target_path="${target_seed_dir}/${base_name}"

        if [ -e "$target_path" ]; then
            if [ -f "$entry" ] && [ -f "$target_path" ] && cmp -s "$entry" "$target_path"; then
                rm -f "$entry"
                SKIPPED_IDENTICAL_FILES=$((SKIPPED_IDENTICAL_FILES + 1))
                continue
            fi
            echo "Collision while combining split=${split} seed=${seed} dataset=${dataset}: ${target_path}" >&2
            exit 1
        fi

        mv "$entry" "$target_seed_dir/"
        MOVED_ITEMS=$((MOVED_ITEMS + 1))
    done
}

parse_args "$@"
validate_args

echo "Combining RNG Optuna dataset shards"
echo "Output dir: $OUTPUT_DIR"
echo "Seeds: ${SEEDS[*]}"
echo "Splits: ${EVALUATION_SPLITS[*]}"
if [ "${#DATASETS[@]}" -gt 0 ]; then
    echo "Datasets (explicit): ${DATASETS[*]}"
else
    echo "Datasets: auto-discover from _dataset_shards"
fi

for split in "${EVALUATION_SPLITS[@]}"; do
    split_dir="${OUTPUT_DIR}/${split}"
    shard_root="${split_dir}/_dataset_shards"
    split_datasets=()

    if [ ! -d "$shard_root" ]; then
        echo "Skipping split '$split' (no shard directory at ${shard_root})"
        continue
    fi

    if [ "${#DATASETS[@]}" -gt 0 ]; then
        split_datasets=("${DATASETS[@]}")
    else
        discover_datasets_for_split "$split" split_datasets
    fi

    if [ "${#split_datasets[@]}" -eq 0 ]; then
        echo "Skipping split '$split' (no dataset shards found)"
        continue
    fi

    echo "Processing split '$split' with datasets: ${split_datasets[*]}"

    for seed in "${SEEDS[@]}"; do
        target_seed_dir="${split_dir}/seed_${seed}"
        seed_had_shards=0
        mkdir -p "$target_seed_dir"

        for dataset in "${split_datasets[@]}"; do
            source_seed_dir="${shard_root}/${dataset}/seed_${seed}"
            if [ ! -d "$source_seed_dir" ]; then
                continue
            fi

            seed_had_shards=1
            PROCESSED_SOURCE_SEED_DIRS=$((PROCESSED_SOURCE_SEED_DIRS + 1))
            merge_source_seed_dir "$source_seed_dir" "$target_seed_dir" "$split" "$seed" "$dataset"
            rmdir "$source_seed_dir" 2>/dev/null || true
        done

        if [ "$seed_had_shards" -eq 1 ]; then
            study_count="$(find "$target_seed_dir" -maxdepth 1 -type f -name 'study_*.json' | wc -l | tr -d ' ')"
            marker_file="${target_seed_dir}/_COMBINED_OK"
            {
                printf "combined_at=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
                printf "split=%s\n" "$split"
                printf "seed=%s\n" "$seed"
                printf "study_json_count=%s\n" "$study_count"
            } > "$marker_file"
            COMBINED_SEED_DIRS=$((COMBINED_SEED_DIRS + 1))
            echo "  Combined seed_${seed} (${study_count} study files)"
        fi
    done

    find "$shard_root" -type d -empty -delete 2>/dev/null || true
done

echo "Combine complete"
echo "  Source shard seed dirs processed: $PROCESSED_SOURCE_SEED_DIRS"
echo "  Items moved: $MOVED_ITEMS"
echo "  Identical files skipped: $SKIPPED_IDENTICAL_FILES"
echo "  Seed dirs marked combined: $COMBINED_SEED_DIRS"
