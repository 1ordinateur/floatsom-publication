"""Regression checks for publication-workflow XPySOM calibration wiring."""

from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLI_PATH = (
    _REPO_ROOT
    / "floatsom"
    / "benchmarks"
    / "optuna"
    / "optuna_results_analysis"
    / "scripts"
    / "analysis"
    / "publication_figures"
    / "cli.py"
)
_WORKFLOW_DEFAULT_AWARE_PATH = (
    _REPO_ROOT
    / "floatsom"
    / "benchmarks"
    / "optuna"
    / "optuna_results_analysis"
    / "scripts"
    / "analysis"
    / "publication_figures"
    / "workflow_default_aware.py"
)


def test_publication_workflow_builds_xpysom_calibration_from_explicit_topology_csvs():
    text = _WORKFLOW_DEFAULT_AWARE_PATH.read_text(encoding="utf-8")
    assert (
        "def _render_xpysom_default_calibration_publication_figures(" in text
    ), "Publication workflow must define the XPySOM calibration builder."
    assert (
        "def _load_explicit_xpysom_topology_benchmark_csv(" in text
    ), "Publication workflow must define an explicit topology benchmark CSV loader."
    assert (
        "combined_runs_df = _load_explicit_xpysom_topology_benchmark_csv(" in text
    ), "XPySOM calibration must load each topology comparison from its explicit self-contained CSV."
    assert (
        'required_methods = {"floatsom", "xpysom"}' in text
    ), "Explicit topology benchmark CSVs must contain both floatsom and xpysom rows."
    assert (
        '"comparison_topology"] == topology_key' in text
    ), "Explicit topology benchmark CSVs must be filtered/validated against the expected topology."


def test_cli_wires_xpysom_calibration_publication_from_explicit_topology_args():
    text = _CLI_PATH.read_text(encoding="utf-8")
    assert (
        "--mst-xpysom-defaults-csv" in text
    ), "CLI must expose an explicit MST XPySOM calibration CSV argument."
    assert (
        "--rng-xpysom-defaults-csv" in text
    ), "CLI must expose an explicit RNG XPySOM calibration CSV argument."
    assert (
        "--hexagonal-xpysom-defaults-csv" in text
    ), "CLI must expose an explicit hexagonal XPySOM calibration CSV argument."
    assert (
        "xpysom_calibration_csvs = {" in text
    ), "CLI must assemble the explicit topology calibration CSV mapping."
    assert (
        "if any(bool(value) for value in xpysom_calibration_csvs.values()):" in text
    ), "CLI must trigger the calibration workflow from the explicit topology inputs."


def test_cli_wires_explicit_xpysom_tripanel_topology_args():
    text = _CLI_PATH.read_text(encoding="utf-8")
    assert (
        "--xpysom-hexagonal-runs-file" in text
    ), "CLI must expose an explicit hexagonal XPySOM tripanel CSV argument."
    assert (
        "--xpysom-mst-runs-file" in text
    ), "CLI must expose an explicit MST XPySOM tripanel CSV argument."
    assert (
        "xpysom_topology_tripanel_csvs = {" in text
    ), "CLI must assemble the explicit topology tripanel CSV mapping."
    assert (
        "_render_xpysom_topology_tripanel_publication_figures(" in text
    ), "CLI must trigger the XPySOM topology tripanel workflow from the explicit topology inputs."


def test_publication_workflow_builds_xpysom_tripanels_from_explicit_topology_csvs():
    text = _WORKFLOW_DEFAULT_AWARE_PATH.read_text(encoding="utf-8")
    assert (
        "def _render_xpysom_topology_tripanel_publication_figures(" in text
    ), "Publication workflow must define the XPySOM topology tripanel builder."
    assert (
        "def _normalize_xpysom_topology_tripanel_runs_frame(" in text
    ), "Publication workflow must normalize explicit XPySOM topology CSVs for tripanel pairing."
    assert (
        "_load_explicit_xpysom_topology_benchmark_csv(" in text
    ), "Tripanel workflow must validate the explicit self-contained topology CSVs."
    assert (
        "_render_matched_parameter_tripanel_figure(" in text
    ), "Tripanel workflow must reuse the shared tripanel composition helper."
