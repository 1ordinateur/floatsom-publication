"""Regression checks for XPySOM-vs-FloatSOM Figure 2 manuscript wiring."""

from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANUSCRIPT_PATH = _REPO_ROOT / "floatsom" / "paper" / "manuscript.md"
_BENCHMARK_SCRIPT_PATH = (
    _REPO_ROOT / "floatsom" / "benchmarks" / "optuna" / "benchmark_xpysom_hex_batch_full.py"
)
_FIGURE2_CANONICAL_PATH = (
    _REPO_ROOT / "floatsom" / "paper" / "assets" / "figures" / "fig_xpysom_calibration_qe_hexagonal.svg"
)


def test_manuscript_figure2_points_to_generated_calibration_asset():
    text = _MANUSCRIPT_PATH.read_text(encoding="utf-8")
    assert (
        "![Figure 2](assets/figures/fig_xpysom_calibration_qe_hexagonal.svg)" in text
    ), "Figure 2 must reference the generated calibration SVG in paper/assets/figures."
    assert (
        "fig_xpysom_equivalence_placeholder.svg" not in text
    ), "Figure 2 must not reference the placeholder SVG."


def test_manuscript_supplementary_figure3_points_to_supplementary_hex_asset():
    text = _MANUSCRIPT_PATH.read_text(encoding="utf-8")
    assert (
        "![Supplementary Figure S3](assets/figures/supp_fig_xpysom_calibration_qe_hexagonal.svg)" in text
    ), "Supplementary Figure S3 must reference the supplementary hex calibration SVG."


def test_hex_publish_writes_main_figure_asset_copy():
    text = _BENCHMARK_SCRIPT_PATH.read_text(encoding="utf-8")
    assert (
        'PAPER_MAIN_FIGURE_FILENAME = "fig_xpysom_calibration_qe_hexagonal.svg"' in text
    ), "Benchmark script must define the canonical Figure 2 filename."
    assert (
        'PAPER_SUPP_FIGURE_FILENAME = "supp_fig_xpysom_calibration_qe_hexagonal.svg"' in text
    ), "Benchmark script must define the canonical supplementary Figure S3 filename."
    assert (
        "return PAPER_SUPP_FIGURE_FILENAME" in text
    ), "Hex topology outputs must use the supplementary filename before publishing the Figure 2 alias."
    assert (
        'if str(topology) == "hexagonal":' in text
    ), "Hex topology branch must exist for publishing Figure 2."
    assert (
        "figure_main_target = paper_figures_dir / PAPER_MAIN_FIGURE_FILENAME" in text
    ), "Hex publish branch must target the canonical Figure 2 asset path."
    assert (
        "shutil.copy2(figure_path, figure_main_target)" in text
    ), "Hex publish branch must copy the generated figure into the canonical Figure 2 asset."


def test_canonical_figure2_asset_is_not_placeholder_svg():
    text = _FIGURE2_CANONICAL_PATH.read_text(encoding="utf-8")
    assert (
        "Draft asset: replace with finalized calibration plot." not in text
    ), "Canonical Figure 2 asset must contain generated figure content, not placeholder text."
