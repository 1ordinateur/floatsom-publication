#!/bin/bash
#PBS -N reviewer1_qfigs
#PBS -q expresssr
#PBS -P eu59
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l walltime=12:00:00
#PBS -l jobfs=100GB
#PBS -l storage=gdata/eu59+gdata/dk92
#PBS -l wd
#PBS -j oe

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/g/data/eu59/piblo_project/floatsom-publication}"
RUN_TAG="${RUN_TAG:-reviewer1_qvalues_${PBS_JOBID:-manual}_$(date +%Y%m%d_%H%M%S)}"
WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/Results/${RUN_TAG}}"
PUBLICATION_ROOT="${PUBLICATION_ROOT:-${REPO_ROOT}/Results/publication_figures}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
IMPORT_SHIM_ROOT="${IMPORT_SHIM_ROOT:-${WORK_ROOT}/pythonpath}"
IMPORT_SHIM_PACKAGE="${IMPORT_SHIM_ROOT}/floatsom"

RAW_OPTUNA_DIR="${RAW_OPTUNA_DIR:-${REPO_ROOT}/outputs/noninf_optuna_benchmarks_28022026/both}"
TRUE_DEFAULT_CSV="${TRUE_DEFAULT_CSV:-${REPO_ROOT}/outputs/tuned_default_comparison_10032026/true_default/matched_true_default_runs.csv}"
TUNED_FIXED_CSV="${TUNED_FIXED_CSV:-${REPO_ROOT}/outputs/tuned_default_comparison_10032026/tuned_fixed/matched_tuned_fixed_runs.csv}"
XPYSOM_ROOT="${XPYSOM_ROOT:-${REPO_ROOT}/data/xpysom_hex_batch_full_benchmark_tuned_20032026}"
XPYSOM_SCALING_CSV="${XPYSOM_SCALING_CSV:-${REPO_ROOT}/outputs/18032026_scaling/4gpu_xpysom_rng_scaling_15032026/xpysom_rng_scaling_runs.csv}"

export OMP_NUM_THREADS="${PBS_NCPUS:-12}"
export OPENBLAS_NUM_THREADS="${PBS_NCPUS:-12}"
export MKL_NUM_THREADS="${PBS_NCPUS:-12}"
export NUMEXPR_NUM_THREADS="${PBS_NCPUS:-12}"
export MPLCONFIGDIR="${PBS_JOBFS:-/tmp}/${USER:-floatsom}-matplotlib"
export TMPDIR="${PBS_JOBFS:-/tmp}"
export FLOATSOM_REPO_ROOT="$REPO_ROOT"
export FLOATSOM_IMPORT_SHIM_PACKAGE="$IMPORT_SHIM_PACKAGE"

mkdir -p "$MPLCONFIGDIR" "$WORK_ROOT" "$PUBLICATION_ROOT" "$IMPORT_SHIM_PACKAGE"

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Work root: $WORK_ROOT"
echo "Publication root: $PUBLICATION_ROOT"

module use /g/data/dk92/apps/Modules/modulefiles/
module load rapids/25.06

cd "$REPO_ROOT"

echo "Configuring FloatSOM import path..."
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os

repo_root = Path(os.environ["FLOATSOM_REPO_ROOT"]).resolve()
shim_package = Path(os.environ["FLOATSOM_IMPORT_SHIM_PACKAGE"]).resolve()
shim_package.mkdir(parents=True, exist_ok=True)
(shim_package / "__init__.py").write_text(
    "\n".join(
        [
            '"""Job-local import shim for the FloatSOM publication checkout."""',
            "from pathlib import Path as _Path",
            f"_REPO_ROOT = _Path({str(repo_root)!r})",
            "__path__ = [str(_REPO_ROOT)]",
            "if __spec__ is not None:",
            "    __spec__.submodule_search_locations = __path__",
            "",
        ]
    ),
    encoding="utf-8",
)
PY
export PYTHONPATH="${IMPORT_SHIM_ROOT}:${PYTHONPATH:-}"
echo "PYTHONPATH import shim: $IMPORT_SHIM_ROOT"

echo "Checking Python dependencies..."
"$PYTHON_BIN" - <<'PY'
import floatsom
import floatsom.benchmarks
import matplotlib
import numpy
import pandas
import scipy
import seaborn
import sklearn
print("Python dependency check passed.")
PY

echo "Patching visible forest-legend labels from p to q if needed..."
PLOTS_FILE="benchmarks/optuna/optuna_results_analysis/scripts/analysis/publication_figures/plots.py"
if grep -q 'Significant (p <\|Non-Significant (p >' "$PLOTS_FILE"; then
  cp "$PLOTS_FILE" "${PLOTS_FILE}.pre_qvalue_label_patch"
  perl -0pi -e 's/Significant \(p </Significant (q </g; s/Non-Significant \(p >/Non-Significant (q >=/g' "$PLOTS_FILE"
fi

echo "Validating input paths..."
for path in \
  "$RAW_OPTUNA_DIR" \
  "$TRUE_DEFAULT_CSV" \
  "$TUNED_FIXED_CSV" \
  "$XPYSOM_ROOT/mst/xpysom_mst_batch_full_runs.csv" \
  "$XPYSOM_ROOT/rng/xpysom_rng_batch_full_runs.csv" \
  "$XPYSOM_ROOT/hexagonal/xpysom_hexagonal_batch_full_runs.csv" \
  "$XPYSOM_ROOT/xpysom_batch_topology_sweep_runs.csv" \
  "$XPYSOM_SCALING_CSV"; do
  if [ ! -e "$path" ]; then
    echo "ERROR: missing input path: $path" >&2
    exit 1
  fi
  echo "  OK: $path"
done

echo "Stage 1/4: harmonizing raw Optuna JSON studies..."
"$PYTHON_BIN" benchmarks/optuna/harmonization/harmonize_optuna_results.py \
  --results-dir "$RAW_OPTUNA_DIR" \
  --output-dir "$WORK_ROOT/harmonized" \
  --objectives quantization_error_holdout quantization_error_train

echo "Stage 2/4: exporting harmonized Pareto CSV..."
"$PYTHON_BIN" benchmarks/optuna/harmonization/export_pareto_to_csv.py \
  --input-dir "$WORK_ROOT/harmonized" \
  --output "$WORK_ROOT/pareto_front_results.csv"

echo "Stage 3/4: creating processed analysis CSV..."
"$PYTHON_BIN" benchmarks/optuna/optuna_results_analysis/execute_analysis.py \
  --data-file "$WORK_ROOT/pareto_front_results.csv" \
  --output-dir "$WORK_ROOT/analysis" \
  --no-filter \
  --best-performers \
  --by-architecture

PROCESSED_CSV="$WORK_ROOT/analysis/unified/processed_data_with_overall_score.csv"
if [ ! -f "$PROCESSED_CSV" ]; then
  echo "ERROR: expected processed CSV was not created: $PROCESSED_CSV" >&2
  exit 1
fi

echo "Stage 4/4: regenerating publication figures and paper assets..."
"$PYTHON_BIN" benchmarks/optuna/optuna_results_analysis/scripts/analysis/generate_publication_figures.py \
  --data-file "$PROCESSED_CSV" \
  --output-dir "$PUBLICATION_ROOT" \
  --run-name "$RUN_TAG" \
  --full-suite \
  --default-runs-file "$TRUE_DEFAULT_CSV" \
  --default-aware-tuned-runs-file "$TUNED_FIXED_CSV" \
  --mst-xpysom-defaults-csv "$XPYSOM_ROOT/mst/xpysom_mst_batch_full_runs.csv" \
  --rng-xpysom-defaults-csv "$XPYSOM_ROOT/rng/xpysom_rng_batch_full_runs.csv" \
  --hexagonal-xpysom-defaults-csv "$XPYSOM_ROOT/hexagonal/xpysom_hexagonal_batch_full_runs.csv" \
  --xpysom-default-runs-file "$XPYSOM_ROOT/xpysom_batch_topology_sweep_runs.csv" \
  --xpysom-rng-runs-file "$TUNED_FIXED_CSV" \
  --xpysom-rng-scaling-csv "$XPYSOM_SCALING_CSV" \
  --xpysom-hexagonal-runs-file "$XPYSOM_ROOT/hexagonal/xpysom_hexagonal_batch_full_runs.csv" \
  --xpysom-mst-runs-file "$XPYSOM_ROOT/mst/xpysom_mst_batch_full_runs.csv"

echo "Checking for stale visible p-value legend labels in regenerated SVG assets..."
if grep -R "Significant (p <\|Non-Significant (p >\|p &lt; 0.05\|p &gt; 0.05" \
  paper/assets/figures paper/assets_manual/figures --include='*.svg'; then
  echo "ERROR: stale p-value legend text remains in one or more SVG assets." >&2
  exit 1
fi

echo "Generated run manifest:"
if [ -f "$PUBLICATION_ROOT/$RUN_TAG/manifest.json" ]; then
  echo "$PUBLICATION_ROOT/$RUN_TAG/manifest.json"
fi

echo "Job finished: $(date)"
