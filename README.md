# FloatSOM publication and reproducibility

This repository accompanies **FloatSOM: GPU Accelerated, Distributed,
Topology-Flexible Self-Organizing Maps**. The actively developed library is in
[1ordinateur/floatsom](https://github.com/1ordinateur/floatsom).

## Paper and results

- [Current TMLR manuscript](tmlr_paper/main.pdf) and [LaTeX source](tmlr_paper/main.tex).
- [Editable figures and exported PDFs](tmlr_paper/assets/figures), including
  [figure editing instructions](tmlr_paper/assets/figures/STYLE_NOTES.md).
- [Current supplementary tables](tmlr_paper/assets/tables).
- [Extended supplementary archive](zenodo/floatsom_extended_supplementary/README.md).
- [Analysis tables](paper/README.md) retained as numerical supporting material.
- [Benchmark scripts](benchmarks) and [default-profile guide](benchmarks/optuna/config/DEFAULTS.md).

The accepted-format manuscript still needs its publication month/year, OpenReview
forum URL and the remaining author email addresses before final submission.

## Build the manuscript

Use the included TMLR style files and a LaTeX installation with `latexmk`:

```bash
cd tmlr_paper
latexmk -pdf main.tex
```

Alternatively, run `tectonic main.tex` from `tmlr_paper/`. No benchmark run is
needed to build the paper. The PDF figure exports are already included.

Figures contain substantial manual edits. Edit the SVGs directly and export
matching PDFs; do not overwrite them with the benchmark figure generators.
Use Figure 7 for forest styling and Figure 5 for the title-to-content gap.

## Install and run the benchmarks

Clone the standalone FloatSOM repository alongside this one. Install both in the
same environment: the library provides `floatsom`, and this repository provides
`floatsom_benchmarks`. Runtime code and tuned profiles are maintained in the library.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '../floatsom[cuda12]'
python -m pip install -e '.[dev]'
```

Training requires a working NVIDIA driver and compatible CUDA toolkit. Use
`'../floatsom[cuda13]'` for a CUDA 13 environment; use CUDA 12 for V100 hardware.
Install only one CUDA extra. See the [library installation guide](https://github.com/1ordinateur/floatsom#installation)
for details. Optional external baselines must be installed separately.

Entry points include:

- `python -m floatsom_benchmarks.run_sklearn_benchmarks --help`
- `python -m floatsom_benchmarks.speed_benchmarks.run_gpu_scaling_benchmark --help`
- Scripts and job templates in `benchmarks/optuna/` for tuning and paired analysis.

Benchmark constructors explicitly select the `publication` tuned profile. The
library also retains its original `library` profile and the named experimental
variants; see the [profile guide](benchmarks/optuna/config/DEFAULTS.md).

Cluster job scripts preserve the experiment workflows. Set project allocations,
queues, paths and GPU resources for your environment before submitting them.
Generated analysis plots may be written to the ignored `paper/assets/figures/`
and `paper/assets_manual/figures/` directories; these are not manuscript sources.

## Verification

```bash
python -m pytest tests
python -m build
python tools/check_distribution.py dist/*.whl
```

The full suite requires CUDA and, for distributed tests, Ray GPU resources.
CPU-only publication checks can be run after installing the non-CUDA dependencies:

```bash
python -m pytest tests/benchmarks/test_xpysom_figure2_wiring.py tests/benchmarks/test_paired_analysis_stats.py tests/test_default_profiles.py
```

The numerical supplement includes checksums; verify it with:

```bash
cd zenodo/floatsom_extended_supplementary
sha256sum -c SHA256SUMS
```

Historical drafts, reviewer correspondence, duplicate figure exports, submission
bundles and intermediate LaTeX files are excluded from the current publication
tree. Tracked history is unchanged. The license in `tmlr_paper/LICENSE` covers
the TMLR template; it is not a license for FloatSOM code. The extended supplement
has its own CC BY 4.0 license. A repository-level software license is pending.
