"""Regression checks for manuscript figure wiring around XPySOM calibration assets."""

from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANUSCRIPT_PATH = _REPO_ROOT / "floatsom" / "paper" / "manuscript.md"
_BENCHMARK_SCRIPT_PATH = (
    _REPO_ROOT / "floatsom" / "benchmarks" / "optuna" / "benchmark_xpysom_hex_batch_full.py"
)
_SUPP_FIGURE3_CANONICAL_PATH = _REPO_ROOT / "floatsom" / "paper" / "assets" / "figures" / "supp_fig_s3.svg"


def test_manuscript_figure3_points_to_promoted_hdsssom_asset():
    text = _MANUSCRIPT_PATH.read_text(encoding="utf-8")
    assert (
        "![Figure 3](assets_manual/figures/fig_3.svg)" in text
    ), "Figure 3 must reference the promoted manual Figure 3 SVG."
    assert (
        "assets_manual/figures/legacy_supp_fig_s4_wrapper.svg" not in text
    ), "Figure 3 must not reference the legacy HDSSSOM wrapper SVG."


def test_manuscript_supplementary_figure3_points_to_supplementary_hex_asset():
    text = _MANUSCRIPT_PATH.read_text(encoding="utf-8")
    assert (
        "![Supplementary Figure S3](assets_manual/figures/supp_fig_s3.svg)" in text
    ), "Supplementary Figure S3 must reference the supplementary hex calibration SVG."


def test_hex_publish_writes_canonical_supplementary_asset_copy():
    text = _BENCHMARK_SCRIPT_PATH.read_text(encoding="utf-8")
    assert (
        'PAPER_SUPP_FIGURE_FILENAME = "supp_fig_s3.svg"' in text
    ), "Benchmark script must define the canonical supplementary Figure S3 filename."
    assert (
        "return PAPER_SUPP_FIGURE_FILENAME" in text
    ), "Hex topology outputs must use the canonical supplementary filename."


def test_canonical_supplementary_figure3_asset_is_not_placeholder_svg():
    text = _SUPP_FIGURE3_CANONICAL_PATH.read_text(encoding="utf-8")
    assert (
        "Draft asset: replace with finalized calibration plot." not in text
    ), "Canonical Supplementary Figure S3 asset must contain generated figure content, not placeholder text."
