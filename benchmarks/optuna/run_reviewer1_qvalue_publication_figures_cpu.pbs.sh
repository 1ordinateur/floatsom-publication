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

PROCESSED_CSV="${PROCESSED_CSV:-}"
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
import floatsom_benchmarks
import matplotlib
import numpy
import pandas
import scipy
import seaborn
import sklearn
print("Python dependency check passed.")
PY

if [ -z "$PROCESSED_CSV" ]; then
  echo "ERROR: PROCESSED_CSV must point to the original processed_data_with_overall_score.csv." >&2
  echo "Do not pass raw Optuna JSON results here; this job only regenerates figures from the original processed analysis table." >&2
  echo "Example:" >&2
  echo "  qsub -v PROCESSED_CSV=/path/to/unified/processed_data_with_overall_score.csv benchmarks/optuna/run_reviewer1_qvalue_publication_figures_cpu.pbs.sh" >&2
  exit 1
fi

echo "Validating input paths..."
for path in \
  "$PROCESSED_CSV" \
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

echo "Stage 1/2: using original processed analysis CSV..."
"$PYTHON_BIN" - "$PROCESSED_CSV" <<'PY'
import sys
from pathlib import Path
import pandas as pd

path = Path(sys.argv[1])
df = pd.read_csv(path, nrows=5)
if "dataset" not in df.columns:
    raise SystemExit("ERROR: processed CSV is missing expected column: dataset")
print(f"Processed CSV check passed: {path}")
print(f"First columns: {', '.join(list(df.columns[:12]))}")
PY

echo "Stage 2/2: regenerating publication figures and paper assets..."
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
STALE_LABEL_REPORT="$WORK_ROOT/stale_p_value_svg_labels.txt"
if grep -R "Significant (p <\|Non-Significant (p >\|p &lt; 0.05\|p &gt; 0.05" \
  paper/assets/figures paper/assets_manual/figures --include='*.svg' > "$STALE_LABEL_REPORT"; then
  echo "WARNING: stale p-value legend text remains in one or more SVG asset(s)." >&2
  echo "WARNING: stale p-value legend text remains in one or more SVG asset(s)."
  echo "Stale label report: $STALE_LABEL_REPORT" >&2
  echo "Stale label report: $STALE_LABEL_REPORT"
  while IFS= read -r stale_line; do
    echo "  $stale_line" >&2
    echo "  $stale_line"
  done < "$STALE_LABEL_REPORT"
else
  rm -f "$STALE_LABEL_REPORT"
  echo "No stale visible p-value legend labels were detected."
fi

echo "Generated run manifest:"
MANIFEST_PATH="$PUBLICATION_ROOT/$RUN_TAG/manifest.json"
if [ -f "$MANIFEST_PATH" ]; then
  echo "$MANIFEST_PATH"
fi

echo "Publication figure sync summary:"
"$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])

def emit_warning(line: str) -> None:
    print(line)
    print(line, file=sys.stderr)

if not manifest_path.exists():
    emit_warning(f"WARNING: publication figure manifest was not created: {manifest_path}")
    raise SystemExit(0)

payload = json.loads(manifest_path.read_text(encoding="utf-8"))
diagnostics = payload.get("diagnostics", {})
missing = []
if isinstance(diagnostics, dict):
    for diagnostic_name, diagnostic_payload in diagnostics.items():
        if not isinstance(diagnostic_payload, dict):
            continue
        for item in diagnostic_payload.get("missing_publication_figure_sources", []) or []:
            if isinstance(item, dict):
                missing.append((diagnostic_name, item))

if not missing:
    print("No missing publication figure sync sources were reported.")
    raise SystemExit(0)

emit_warning("WARNING: publication figure sync completed with missing source figure(s):")
for diagnostic_name, item in missing:
    source = item.get("source", "<unknown source>")
    destination = item.get("destination", "<unknown destination>")
    reason = item.get("reason", "No reason recorded.")
    required = item.get("required", False)
    emit_warning(
        f"  - {diagnostic_name}: {source} -> {destination} "
        f"(required={required}; {reason})"
    )
PY

echo "Job finished: $(date)"
