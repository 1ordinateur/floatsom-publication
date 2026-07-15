#!/bin/bash
#PBS -N figure5_topology_1gpu
#PBS -q gpuvolta
#PBS -P eu59
#PBS -l ncpus=12
#PBS -l mem=90GB
#PBS -l ngpus=1
#PBS -l walltime=06:00:00
#PBS -l storage=gdata/eu59+gdata/dk92+scratch/eu59
#PBS -l jobfs=50GB
#PBS -l wd
#PBS -M tony.xu@anu.edu.au
#PBS -m abe

set -euo pipefail

module use /g/data/dk92/apps/Modules/modulefiles/
FLOATSOM_MODULE="${FLOATSOM_MODULE:-rapids/25.06}"
module load "$FLOATSOM_MODULE"

REPO_ROOT="${REPO_ROOT:-/g/data/eu59/piblo_project/floatsom-publication}"
SECOND_DATASET="${SECOND_DATASET:-sklearn_covertype}"
SECOND_DATASET_TAG="${SECOND_DATASET_TAG:-covertype}"
SECOND_DATASET_DIM="${SECOND_DATASET_DIM:-54}"
FIGURE_STEM="${FIGURE_STEM:-fig_5}"
PUBLISH_ASSETS="${PUBLISH_ASSETS:-1}"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_TAG="${RUN_TAG:-figure5_${SECOND_DATASET_TAG}_gpu_${RUN_TIMESTAMP}_${PBS_JOBID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/Results/${RUN_TAG}}"
WORK_DIR="${WORK_DIR:-${OUTPUT_DIR}/work}"
SKLEARN_DATA_HOME="${SKLEARN_DATA_HOME:-${REPO_ROOT}/sklearn_data}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
IMPORT_SHIM_ROOT="${IMPORT_SHIM_ROOT:-${WORK_DIR}/pythonpath}"
IMPORT_SHIM_PACKAGE="${IMPORT_SHIM_ROOT}/floatsom"

STAGED_SVG="${WORK_DIR}/${FIGURE_STEM}.svg"
STAGED_TABLE="${WORK_DIR}/${FIGURE_STEM}_summary.tsv"
STAGED_CIRCLES_JPEG="${WORK_DIR}/${FIGURE_STEM}_circles_background.jpg"
STAGED_SECOND_JPEG="${WORK_DIR}/${FIGURE_STEM}_${SECOND_DATASET_TAG}_background.jpg"

mkdir -p "$OUTPUT_DIR" "$WORK_DIR" "$SKLEARN_DATA_HOME" "$IMPORT_SHIM_PACKAGE"
export SCIKIT_LEARN_DATA="$SKLEARN_DATA_HOME"
export MPLCONFIGDIR="${PBS_JOBFS:-/tmp}/matplotlib"
export TMPDIR="${PBS_JOBFS:-/tmp}"
export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${PBS_NCPUS:-12}"
export OPENBLAS_NUM_THREADS="${PBS_NCPUS:-12}"
export MKL_NUM_THREADS="${PBS_NCPUS:-12}"
export FLOATSOM_REPO_ROOT="$REPO_ROOT"
export FLOATSOM_IMPORT_SHIM_PACKAGE="$IMPORT_SHIM_PACKAGE"
export SECOND_DATASET SECOND_DATASET_TAG SECOND_DATASET_DIM FIGURE_STEM PUBLISH_ASSETS
mkdir -p "$MPLCONFIGDIR"

cd "$REPO_ROOT"

echo "Started Figure 5 GPU job at ${RUN_TIMESTAMP}"
echo "Repository: ${REPO_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Working directory: ${WORK_DIR}"
echo "SCIKIT_LEARN_DATA: ${SCIKIT_LEARN_DATA}"

# The checkout directory is named floatsom-publication rather than floatsom.
# Provide a job-local package shim so imports resolve to this exact checkout.
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

echo "Checking GPU and Python dependencies..."
"$PYTHON_BIN" - <<'PY'
import cupy as cp
import numpy
import PIL
import scipy
import sklearn
import floatsom

device_count = cp.cuda.runtime.getDeviceCount()
if device_count < 1:
    raise SystemExit("No CUDA device is visible to the Figure 5 job")
print(f"CUDA devices visible: {device_count}")
print(f"CuPy version: {cp.__version__}")
print("Dependency check passed")
PY

# Keep generated dataset caches and all intermediate products out of the
# publication tree until the staged outputs pass the checks below.
cd "$WORK_DIR"

echo "Training matched circles and ${SECOND_DATASET_TAG} SOMs through FloatSOM/CuPy..."
"$PYTHON_BIN" "$REPO_ROOT/benchmarks/run_topology_representative_figure.py" \
  --data-types sklearn_circles "$SECOND_DATASET" \
  --difficulty hard \
  --seed 42 \
  --grid_size 10 \
  --mst_nodes 100 \
  --iterations 50 \
  --display-limit 30000 \
  --jpeg-quality 90 \
  --raster-scale 2 \
  --output_svg "$STAGED_SVG" \
  --summary_tsv "$STAGED_TABLE" \
  --background-dir "$WORK_DIR" \
  --verbose

echo "Validating staged GPU outputs..."
export STAGED_SVG STAGED_TABLE STAGED_CIRCLES_JPEG STAGED_SECOND_JPEG
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import csv
import os
import re

from PIL import Image

svg_path = Path(os.environ["STAGED_SVG"])
table_path = Path(os.environ["STAGED_TABLE"])
circle_jpeg = Path(os.environ["STAGED_CIRCLES_JPEG"])
second_jpeg = Path(os.environ["STAGED_SECOND_JPEG"])
second_dataset_tag = os.environ["SECOND_DATASET_TAG"]
second_dataset_dim = os.environ["SECOND_DATASET_DIM"]

for path in (svg_path, table_path, circle_jpeg, second_jpeg):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Missing or empty staged output: {path}")

svg = svg_path.read_text(encoding="utf-8")
checks = {
    "embedded JPEG definitions": svg.count("data:image/jpeg;base64,") == 2,
    "row background definitions": len(re.findall(r'<image id="cloud-row-', svg)) == 2,
    "Illustrator-compatible image namespace": 'xmlns:xlink="http://www.w3.org/1999/xlink"' in svg,
    "six row-background uses": len(re.findall(r'<use xlink:href="#cloud-row-', svg)) == 6,
    "figure title": "Figure 5: Representative Topologies" in svg,
    "six panel labels": all(f">{label}</text>" in svg for label in "ABCDEF"),
    "no row subtitles": "native 2D display" not in svg and "shared 2D PCA projection" not in svg,
    "no PCA caveat": "may distort graph geometry" not in svg,
    "no vector observation cloud": len(re.findall(r'<circle[^>]+fill="#808080"', svg)) == 1,
    "expected vector nodes and legend": svg.count("<circle") == 602,
    "source SVG below 7.2 MiB": svg_path.stat().st_size < int(7.2 * 1024 * 1024),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Figure 5 SVG validation failed: " + ", ".join(failed))

for path in (circle_jpeg, second_jpeg):
    with Image.open(path) as image:
        if image.format != "JPEG" or image.mode != "RGB":
            raise SystemExit(f"Unexpected raster format for {path}: {image.format}/{image.mode}")
        if image.size != (851, 660):
            raise SystemExit(f"Unexpected two-times panel raster size for {path}: {image.size}")

with table_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != 6:
    raise SystemExit(f"Expected six summary rows, found {len(rows)}")
if {row["dataset"] for row in rows} != {"circles", second_dataset_tag}:
    raise SystemExit("Summary table does not contain the expected datasets")
if {row["training_dimensions"] for row in rows} != {"2", second_dataset_dim}:
    raise SystemExit("Summary table does not contain the expected training dimensions")
if {row["topology"] for row in rows} != {"hexagonal", "mst", "rng"}:
    raise SystemExit("Summary table does not contain all three topologies")
if {row["generation_backend"] for row in rows} != {"FloatSOM_CuPy"}:
    raise SystemExit("Summary table was not produced by the FloatSOM/CuPy pathway")
if {row["seed"] for row in rows} != {"42"}:
    raise SystemExit("Unexpected seed in summary table")
if {row["iterations_requested"] for row in rows} != {"50"}:
    raise SystemExit("Unexpected iteration count in summary table")
if {row["n_nodes"] for row in rows} != {"100"}:
    raise SystemExit("Unexpected node count in summary table")
if {row["initial_learning_rate"] for row in rows} != {"0.5"}:
    raise SystemExit("Figure 5 did not use the default XPySOM learning rate")
if {row["initial_radius"] for row in rows} != {"5.0"}:
    raise SystemExit("Figure 5 did not use the untuned XPySOM-like radius")
if {row["radius_decay_type"] for row in rows} != {"exponential"}:
    raise SystemExit("Figure 5 did not use the untuned exponential radius decay")
if {row["momentum_enabled"] for row in rows} != {"False"}:
    raise SystemExit("Figure 5 unexpectedly enabled momentum")
if {row["initial_momentum"] for row in rows} != {"0.5"}:
    raise SystemExit("Unexpected default momentum coefficient")
if {row["normalization"] for row in rows} != {"xpysom"}:
    raise SystemExit("Figure 5 did not use XPySOM-compatible normalization")

print("Staged Figure 5 GPU outputs passed validation")
PY

if [[ "$PUBLISH_ASSETS" == "1" ]]; then
  echo "Publishing validated Figure 5 assets..."
  install -m 0644 "$STAGED_SVG" "$REPO_ROOT/paper/assets/figures/fig_5.svg"
  install -m 0644 "$STAGED_CIRCLES_JPEG" "$REPO_ROOT/paper/assets/figures/fig_5_circles_background.jpg"
  install -m 0644 "$STAGED_SECOND_JPEG" "$REPO_ROOT/paper/assets/figures/fig_5_${SECOND_DATASET_TAG}_background.jpg"
  install -m 0644 "$STAGED_TABLE" "$REPO_ROOT/paper/assets/tables/table_topology_circles_representative.tsv"
  install -m 0644 "$STAGED_SVG" "$REPO_ROOT/paper/assets_manual/figures/fig_5.svg"
  install -m 0644 "$STAGED_TABLE" "$REPO_ROOT/paper/assets_manual/tables/table_topology_circles_representative.tsv"

  cmp "$REPO_ROOT/paper/assets/figures/fig_5.svg" "$REPO_ROOT/paper/assets_manual/figures/fig_5.svg"
  cmp "$REPO_ROOT/paper/assets/tables/table_topology_circles_representative.tsv" "$REPO_ROOT/paper/assets_manual/tables/table_topology_circles_representative.tsv"
else
  echo "Candidate mode: validated assets retained in ${WORK_DIR}; publication assets were not changed."
fi

echo "Finished Figure 5 GPU job at $(date -u +%Y%m%dT%H%M%SZ)"
echo "Validated run products retained in: ${OUTPUT_DIR}"
