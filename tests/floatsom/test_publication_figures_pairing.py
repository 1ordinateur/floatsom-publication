"""
Unit tests for publication-figure pairing utilities.

These tests cover strict pairing/metric validation and paired-delta math in
the standalone publication figure script that lives under the optuna analysis
submodule.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "floatsom/benchmarks/optuna/optuna_results_analysis/scripts/analysis/generate_publication_figures.py"
)


@pytest.fixture(scope="module")
def publication_figures_module():
    if not _SCRIPT_PATH.exists():
        pytest.fail(f"Missing script under test: {_SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location("generate_publication_figures_test", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        pytest.fail("Unable to create import spec for generate_publication_figures.py")

    # The script is inside the optuna_results_analysis sub-repo and imports from `modules.*`.
    # Add the sub-repo root so those imports resolve when loading via `spec_from_file_location`.
    analysis_repo_root = _SCRIPT_PATH.parents[2]
    if str(analysis_repo_root) not in sys.path:
        sys.path.insert(0, str(analysis_repo_root))

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_metric_columns_strict_mode_rejects_fallback(publication_figures_module):
    df = pd.DataFrame({"quantization_error_holdout": [1.0, 2.0]})

    with pytest.raises(ValueError, match="strict mode is enabled"):
        publication_figures_module._resolve_metric_columns(
            df,
            requested=["quantization_error_holdout_normalized"],
            strict_mode=True,
        )


def test_resolve_metric_columns_non_strict_uses_raw_fallback(publication_figures_module):
    df = pd.DataFrame({"quantization_error_holdout": [1.0, 2.0]})

    resolved = publication_figures_module._resolve_metric_columns(
        df,
        requested=["quantization_error_holdout_normalized"],
        strict_mode=False,
    )

    assert resolved == ["quantization_error_holdout"]


def test_validate_required_pairing_columns_rejects_all_nan(publication_figures_module):
    df = pd.DataFrame(
        {
            "dataset": ["d1", "d2"],
            "pair_sampling": ["full", "random"],
            "pair_seed": [np.nan, np.nan],
        }
    )

    with pytest.raises(ValueError, match="pair_seed"):
        publication_figures_module._validate_required_pairing_columns(
            df=df,
            required_cols=["dataset", "pair_sampling", "pair_seed"],
            context="hex colors-vs-batch",
        )


def test_prepare_dataframe_defaults_pair_split_to_both_when_missing_split(publication_figures_module):
    df = pd.DataFrame(
        {
            "dataset": ["d1"],
            "processing_type": ["batch"],
            "sampling_method": ["full"],
            "seed": [7],
            "config_topology_type": ["hexagonal"],
            "config_batch_mode": ["full_batch"],
            "quantization_error_holdout": [0.4],
            "quantization_error_train": [0.3],
        }
    )

    prepared = publication_figures_module._prepare_dataframe(df)
    assert "pair_split" in prepared.columns
    assert prepared["pair_split"].iloc[0] == "both"


def test_prepare_tuned_vs_default_dataset_summary_for_forest_maps_effect_direction(publication_figures_module):
    summary_df = pd.DataFrame(
        {
            "dataset": ["d1"],
            "pairs": [5],
            "mean_pct": [5.25],
            "ci_low_pct": [4.10],
            "ci_high_pct": [6.40],
            "pct_p_value": [0.01],
        }
    )

    out = publication_figures_module._prepare_tuned_vs_default_dataset_summary_for_forest(summary_df)
    assert len(out) == 1
    row = out.iloc[0]
    assert float(row["median_pct"]) == pytest.approx(5.25)
    assert float(row["mean_pct"]) == pytest.approx(5.25)
    assert float(row["ci_low_pct"]) == pytest.approx(4.10)
    assert float(row["ci_high_pct"]) == pytest.approx(6.40)
    assert int(row["n_pairs"]) == 5


def test_resolve_default_aware_pairing_input_defaults_to_data_file(publication_figures_module, tmp_path):
    base_df = pd.DataFrame(
        {
            "dataset": ["d1"],
            "processing_type": ["batch"],
            "sampling_method": ["full"],
            "seed": [7],
            "config_topology_type": ["hexagonal"],
            "quantization_error_holdout": [0.4],
            "quantization_error_train": [0.3],
        }
    )
    data_file = tmp_path / "main.csv"
    data_file.write_text("dataset\nplaceholder\n", encoding="utf-8")

    out_df, source = publication_figures_module._resolve_default_aware_pairing_input(
        base_df=base_df,
        data_file=data_file,
        override_csv_path=None,
    )

    assert str(source) == str(data_file.resolve())
    assert len(out_df) == 1
    assert out_df.iloc[0]["dataset"] == "d1"


def test_resolve_default_aware_pairing_input_uses_override_csv(publication_figures_module, tmp_path):
    base_df = pd.DataFrame({"dataset": ["base_only"]})
    data_file = tmp_path / "main.csv"
    data_file.write_text("dataset\nplaceholder\n", encoding="utf-8")

    override_path = tmp_path / "override.csv"
    pd.DataFrame(
        {
            "dataset": ["override_ds"],
            "processing_type": ["batch"],
            "sampling_method": ["random"],
            "seed": [11],
            "config_topology_type": ["mst"],
            "config_batch_mode": ["full_batch"],
            "quantization_error_holdout": [0.55],
            "quantization_error_train": [0.45],
        }
    ).to_csv(override_path, index=False)

    out_df, source = publication_figures_module._resolve_default_aware_pairing_input(
        base_df=base_df,
        data_file=data_file,
        override_csv_path=str(override_path),
    )

    assert str(source) == str(override_path.resolve())
    assert len(out_df) == 1
    assert out_df.iloc[0]["dataset"] == "override_ds"
    assert out_df.iloc[0]["pair_split"] == "both"


def test_build_default_aware_pairing_variants_single_source(publication_figures_module):
    selected_df = pd.DataFrame({"dataset": ["d1"]})
    base_df = pd.DataFrame({"dataset": ["base"]})

    variants = publication_figures_module._build_default_aware_pairing_variants(
        selected_df=selected_df,
        selected_source="/tmp/source.csv",
        base_df=base_df,
        base_source="/tmp/source.csv",
    )

    assert len(variants) == 1
    assert variants[0]["key"] == "selected_source"
    assert variants[0]["output_subdir"] == "tuned_vs_default"
    assert variants[0]["markdown_filename"] == "TUNED_VS_TRUE_DEFAULT_PAIRED_TTEST.md"


def test_build_default_aware_pairing_variants_adds_data_file_variant_when_override_differs(
    publication_figures_module,
):
    selected_df = pd.DataFrame({"dataset": ["override_ds"]})
    base_df = pd.DataFrame({"dataset": ["base_ds"]})

    variants = publication_figures_module._build_default_aware_pairing_variants(
        selected_df=selected_df,
        selected_source="/tmp/override.csv",
        base_df=base_df,
        base_source="/tmp/base.csv",
    )

    assert len(variants) == 2
    assert variants[0]["key"] == "selected_source"
    assert variants[1]["key"] == "data_file_source"
    assert variants[1]["output_subdir"] == "tuned_vs_default_data_file"
    assert variants[1]["markdown_filename"] == "TUNED_VS_TRUE_DEFAULT_PAIRED_TTEST_DATA_FILE.md"
    assert str(variants[0]["df"].iloc[0]["dataset"]) == "override_ds"
    assert str(variants[1]["df"].iloc[0]["dataset"]) == "base_ds"


def test_build_default_aware_pairing_variants_adds_selected_topology_specific_variants(
    publication_figures_module,
):
    selected_df = pd.DataFrame(
        {
            "dataset": ["d_mst", "d_rng", "d_hex"],
            "pair_topology": ["mst", "rng", "hexagonal"],
        }
    )
    base_df = pd.DataFrame({"dataset": ["base"]})

    variants = publication_figures_module._build_default_aware_pairing_variants(
        selected_df=selected_df,
        selected_source="/tmp/source.csv",
        base_df=base_df,
        base_source="/tmp/source.csv",
    )

    keys = [str(item.get("key")) for item in variants]
    assert keys == ["selected_source", "selected_source_hex", "selected_source_mst", "selected_source_rng"]

    hex_variant = variants[1]
    mst_variant = variants[2]
    rng_variant = variants[3]
    assert hex_variant["output_subdir"] == "tuned_vs_default_hex"
    assert mst_variant["output_subdir"] == "tuned_vs_default_mst"
    assert rng_variant["output_subdir"] == "tuned_vs_default_rng"
    assert set(hex_variant["df"]["pair_topology"].astype(str).str.lower().tolist()) == {"hexagonal"}
    assert set(mst_variant["df"]["pair_topology"].astype(str).str.lower().tolist()) == {"mst"}
    assert set(rng_variant["df"]["pair_topology"].astype(str).str.lower().tolist()) == {"rng"}


def test_run_default_aware_analysis_override_keeps_topology_specific_variants(
    publication_figures_module,
    tmp_path,
    monkeypatch,
):
    tuned_df = pd.DataFrame(
        {
            "dataset": ["base_dataset"],
            "processing_type": ["batch"],
            "sampling_method": ["full"],
            "batch_mode": ["full_batch"],
            "architecture": ["hexagonal"],
            "seed": [1],
            "evaluation_split": ["both"],
            "balanced_qe_raw": [1.0],
            "quantization_error_holdout": [1.0],
            "quantization_error_train": [1.0],
        }
    )
    override_df = pd.DataFrame(
        {
            "dataset": ["hex_dataset", "mst_dataset", "rng_dataset"],
            "processing_type": ["batch", "batch", "batch"],
            "sampling_method": ["full", "full", "full"],
            "batch_mode": ["full_batch", "full_batch", "full_batch"],
            "architecture": ["hexagonal", "mst", "rng"],
            "seed": [1, 2, 3],
            "evaluation_split": ["both", "both", "both"],
            "balanced_qe_raw": [1.0, 1.1, 1.2],
            "quantization_error_holdout": [1.0, 1.1, 1.2],
            "quantization_error_train": [1.0, 1.1, 1.2],
        }
    )

    default_runs_file = tmp_path / "defaults.csv"
    default_runs_file.write_text("dataset\nplaceholder\n", encoding="utf-8")
    generated_files: list[str] = []
    variant_keys: list[str] = []

    monkeypatch.setattr(
        publication_figures_module,
        "load_default_runs_csv",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "balanced_qe_raw": [1.0],
                "quantization_error_holdout": [1.0],
                "quantization_error_train": [1.0],
            }
        ),
    )

    def _fake_run_tuned_vs_default_variant(
        *,
        variant,
        defaults_df,
        metric_specs,
        default_aware_dir,
        generated_files,
        dpi,
        alpha,
    ):
        del defaults_df, metric_specs, generated_files, dpi, alpha
        variant_keys.append(str(variant.get("key", "")))
        variant_dir = default_aware_dir / str(variant["output_subdir"])
        variant_dir.mkdir(parents=True, exist_ok=True)
        report_path = variant_dir / str(variant["markdown_filename"])
        report_path.write_text("ok\n", encoding="utf-8")
        combined_figure = variant_dir / "figures" / "fig_tuned_vs_true_default_metrics.svg"
        combined_figure.parent.mkdir(parents=True, exist_ok=True)
        combined_figure.write_text("<svg/>", encoding="utf-8")
        return {
            "key": str(variant.get("key", "")),
            "label": str(variant.get("label", "")),
            "source": str(variant.get("source", "")),
            "rows_total": int(len(variant["df"])),
            "output_dir": str(variant_dir.resolve()),
            "report": str(report_path.resolve()),
            "publication": {"combined_figure": str(combined_figure.resolve())},
        }

    def _fake_build_hyperparameter_stability_report(
        *,
        df,
        metric,
        output_dir,
        topology_pairs,
        markdown_filename,
    ):
        del df, metric, topology_pairs
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / markdown_filename
        report_path.write_text("ok\n", encoding="utf-8")
        return report_path

    monkeypatch.setattr(
        publication_figures_module,
        "_run_tuned_vs_default_variant",
        _fake_run_tuned_vs_default_variant,
    )
    monkeypatch.setattr(
        publication_figures_module,
        "_filter_full_batch_rows_for_stability",
        lambda df: (df.copy(), 0),
    )
    monkeypatch.setattr(
        publication_figures_module,
        "build_hyperparameter_stability_report",
        _fake_build_hyperparameter_stability_report,
    )
    monkeypatch.setattr(
        publication_figures_module,
        "_render_default_aware_stability_publication_figures",
        lambda **_kwargs: {"generated": True},
    )
    monkeypatch.setattr(
        publication_figures_module,
        "_render_matched_parameter_publication_figures",
        lambda **_kwargs: {"generated": True},
    )

    summary = publication_figures_module._run_default_aware_analysis(
        tuned_df=tuned_df,
        tuned_vs_default_df=override_df,
        tuned_vs_default_source=str(tmp_path / "override.csv"),
        tuned_vs_default_base_source=str(tmp_path / "base.csv"),
        default_runs_file=default_runs_file,
        run_output_dir=tmp_path / "run",
        output_subdir="default_aware_analysis",
        generated_files=generated_files,
        dpi=200,
        alpha=0.05,
    )

    assert variant_keys == ["selected_source", "selected_source_hex", "selected_source_mst", "selected_source_rng"]
    assert [str(item["key"]) for item in summary["tuned_vs_default_variants"]] == variant_keys


def test_build_topology_sensitivity_series_excludes_colors_label(publication_figures_module, monkeypatch):
    df = pd.DataFrame(
        {
            "algorithm": ["batch", "batch", "colors"],
            "pair_batch_mode": ["full_batch", "minibatch", "all"],
            "dataset": ["d1", "d1", "d1"],
            "pair_sampling": ["full", "full", "full"],
            "pair_seed": [1, 2, 3],
            "metric_x": [0.1, 0.2, 0.3],
        }
    )
    pairs = pd.DataFrame(
        {
            "algorithm_batch_mode": ["Batch (Full batch)", "Batch (Mini-batch)", "Colors"],
            "algorithm": ["batch", "batch", "colors"],
            "pair_batch_mode": ["full_batch", "minibatch", "all"],
        }
    )

    def _fake_sensitivity_summary(**_kwargs):
        return pd.DataFrame(
            {
                "top_k": [1, 3],
                "n_pairs": [1, 1],
                "median_pct": [0.05, 0.07],
                "mean_pct": [0.05, 0.07],
                "ci_low_pct": [0.01, 0.03],
                "ci_high_pct": [0.09, 0.11],
                "p_value": [0.2, 0.3],
                "effect_size_signed": [0.0, 0.0],
            }
        )

    monkeypatch.setattr(publication_figures_module, "_build_sensitivity_summary", _fake_sensitivity_summary)

    series, styles, primary = publication_figures_module._build_topology_sensitivity_series(
        df=df,
        pairs=pairs,
        metric="metric_x",
        compare_col="architecture",
        value_a="mst",
        value_b="hexagonal",
        key_cols=["dataset", "pair_sampling", "pair_seed"],
        top_k_values=(1, 3),
        bootstrap_iterations=10,
        seed=42,
        exclude_algorithm_batch_mode_labels={"colors"},
    )

    assert "Colors" not in series
    assert set(series.keys()) == {"Batch (Full batch)", "Batch (Mini-batch)"}
    assert set(styles.keys()) == set(series.keys())
    assert primary in series


def test_render_tuned_vs_default_publication_figure_shows_dataset_labels_only_in_panel_a(
    publication_figures_module, tmp_path
):
    tuned_dir = tmp_path / "tuned_vs_default"
    tables_dir = tuned_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = ["alpha_dataset_label", "beta_dataset_label"]
    for metric_slug in ["qe_holdout", "qe_train", "balanced_qe_raw"]:
        pd.DataFrame(
            {
                "dataset": dataset_names,
                "pairs": [5, 5],
                "mean_default": [1.0, 1.1],
                "mean_tuned": [0.9, 1.0],
                "mean_delta": [-0.1, -0.1],
                "std_delta": [0.01, 0.01],
                "t_stat": [5.0, 5.0],
                "p_value": [0.01, 0.02],
                "ci_low": [-0.12, -0.12],
                "ci_high": [-0.08, -0.08],
                "tuned_better_rate": [1.0, 1.0],
                "mean_pct": [5.0, 6.0],
                "pct_p_value": [0.01, 0.02],
                "ci_low_pct": [4.0, 5.0],
                "ci_high_pct": [6.0, 7.0],
            }
        ).to_csv(tables_dir / f"tuned_vs_default_dataset_summary_{metric_slug}.csv", index=False)

    generated_files: list[str] = []
    output = publication_figures_module._render_tuned_vs_default_publication_figure(
        tuned_vs_default_dir=tuned_dir,
        metric_specs=[
            ("quantization_error_holdout", "QE Holdout", "qe_holdout"),
            ("quantization_error_train", "QE Train", "qe_train"),
            ("balanced_qe_raw", "Balanced QE", "balanced_qe_raw"),
        ],
        generated_files=generated_files,
        dpi=80,
        alpha=0.05,
    )

    assert output["generated"] is True
    holdout_svg = (tuned_dir / "figures" / "fig_tuned_vs_true_default_qe_holdout.svg").read_text(encoding="utf-8")
    train_svg = (tuned_dir / "figures" / "fig_tuned_vs_true_default_qe_train.svg").read_text(encoding="utf-8")
    balanced_svg = (tuned_dir / "figures" / "fig_tuned_vs_true_default_balanced_qe_raw.svg").read_text(encoding="utf-8")

    assert "alpha_dataset_label" in holdout_svg
    assert "beta_dataset_label" in holdout_svg
    assert "alpha_dataset_label" not in train_svg
    assert "beta_dataset_label" not in train_svg
    assert "alpha_dataset_label" not in balanced_svg
    assert "beta_dataset_label" not in balanced_svg


def test_render_tuned_vs_default_publication_figure_includes_global_overall_row(
    publication_figures_module, tmp_path
):
    tuned_dir = tmp_path / "tuned_vs_default"
    tables_dir = tuned_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    metric_slug = "qe_holdout"
    pd.DataFrame(
        {
            "dataset": ["alpha_dataset_label", "beta_dataset_label"],
            "pairs": [5, 5],
            "mean_default": [1.0, 1.1],
            "mean_tuned": [0.9, 1.0],
            "mean_delta": [-0.1, -0.1],
            "std_delta": [0.01, 0.01],
            "t_stat": [5.0, 5.0],
            "p_value": [0.01, 0.02],
            "ci_low": [-0.12, -0.12],
            "ci_high": [-0.08, -0.08],
            "tuned_better_rate": [1.0, 1.0],
            "mean_pct": [5.0, 6.0],
            "pct_p_value": [0.01, 0.02],
            "ci_low_pct": [4.0, 5.0],
            "ci_high_pct": [6.0, 7.0],
        }
    ).to_csv(tables_dir / f"tuned_vs_default_dataset_summary_{metric_slug}.csv", index=False)
    pd.DataFrame(
        {
            "global_label": ["GLOBAL"],
            "pairs": [10],
            "mean_default": [1.05],
            "mean_tuned": [0.95],
            "mean_delta": [-0.10],
            "std_delta": [0.01],
            "t_stat": [6.0],
            "p_value": [0.01],
            "ci_low": [-0.12],
            "ci_high": [-0.08],
            "tuned_better_rate": [1.0],
            "mean_pct": [5.5],
            "pct_p_value": [0.01],
            "ci_low_pct": [4.5],
            "ci_high_pct": [6.5],
        }
    ).to_csv(tables_dir / f"tuned_vs_default_global_summary_{metric_slug}.csv", index=False)

    generated_files: list[str] = []
    output = publication_figures_module._render_tuned_vs_default_publication_figure(
        tuned_vs_default_dir=tuned_dir,
        metric_specs=[("quantization_error_holdout", "QE Holdout", metric_slug)],
        generated_files=generated_files,
        dpi=80,
        alpha=0.05,
    )

    assert output["generated"] is True
    holdout_svg = (tuned_dir / "figures" / f"fig_tuned_vs_true_default_{metric_slug}.svg").read_text(encoding="utf-8")
    assert "GLOBAL_OVERALL" in holdout_svg


def test_render_tuned_vs_default_publication_figure_uses_figure_7_title(
    publication_figures_module, tmp_path
):
    tuned_dir = tmp_path / "tuned_vs_default"
    tables_dir = tuned_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    metric_slugs = ["qe_holdout", "qe_train", "balanced_qe_raw"]
    for metric_slug in metric_slugs:
        pd.DataFrame(
            {
                "dataset": ["alpha_dataset_label"],
                "pairs": [5],
                "mean_default": [1.0],
                "mean_tuned": [0.9],
                "mean_delta": [-0.1],
                "std_delta": [0.01],
                "t_stat": [5.0],
                "p_value": [0.01],
                "ci_low": [-0.12],
                "ci_high": [-0.08],
                "tuned_better_rate": [1.0],
                "mean_pct": [5.0],
                "pct_p_value": [0.01],
                "ci_low_pct": [4.0],
                "ci_high_pct": [6.0],
            }
        ).to_csv(tables_dir / f"tuned_vs_default_dataset_summary_{metric_slug}.csv", index=False)

    generated_files: list[str] = []
    output = publication_figures_module._render_tuned_vs_default_publication_figure(
        tuned_vs_default_dir=tuned_dir,
        metric_specs=[
            ("quantization_error_holdout", "QE Holdout", "qe_holdout"),
            ("quantization_error_train", "QE Train", "qe_train"),
            ("balanced_qe_raw", "Balanced QE", "balanced_qe_raw"),
        ],
        generated_files=generated_files,
        dpi=80,
        alpha=0.05,
    )

    assert output["generated"] is True
    combined_svg = (tuned_dir / "figures" / "fig_tuned_vs_true_default_metrics.svg").read_text(encoding="utf-8")
    assert "Figure 7: Tuned Configuration vs Untuned Reference Across QE Metrics" in combined_svg


def test_compose_three_panel_stats_figure_uses_auto_shared_xlabel_positioning(
    publication_figures_module, tmp_path, monkeypatch
):
    capture: dict[str, object] = {}

    def _fake_compose(**kwargs):
        capture.update(kwargs)
        return True

    monkeypatch.setattr(publication_figures_module, "_compose_panel_matrix_from_images", _fake_compose)

    panel_a = tmp_path / "a.svg"
    panel_b = tmp_path / "b.svg"
    panel_c = tmp_path / "c.svg"
    for path in [panel_a, panel_b, panel_c]:
        path.write_text("<svg/>", encoding="utf-8")

    out_path = tmp_path / "out.svg"
    ok = publication_figures_module._compose_three_panel_stats_figure(
        panel_rows=[[("A", "One", panel_a), ("B", "Two", panel_b), ("C", "Three", panel_c)]],
        title="Demo",
        output_path=out_path,
        dpi=80,
        shared_x_labels_by_row=["QE % Change"],
    )

    assert ok is True
    assert float(capture["panel_width_override"]) == pytest.approx(1220.0)
    assert float(capture["panel_height_override"]) == pytest.approx(860.0)


def test_direction_banner_text_uses_unicode_arrows(publication_figures_module):
    banner = publication_figures_module._direction_banner_text("Baseline", "Tuned")

    assert banner == "← Favours Baseline | Favours Tuned →"


def test_compose_panel_matrix_svg_renders_larger_direction_banner(
    publication_figures_module, tmp_path
):
    panel_a = tmp_path / "a.svg"
    panel_b = tmp_path / "b.svg"
    panel_c = tmp_path / "c.svg"
    panel_markup = "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'></svg>"
    for path in [panel_a, panel_b, panel_c]:
        path.write_text(panel_markup, encoding="utf-8")

    out_path = tmp_path / "out.svg"
    ok = publication_figures_module._compose_panel_matrix_svg(
        panel_rows=[[("A", "One", panel_a), ("B", "Two", panel_b), ("C", "Three", panel_c)]],
        title="Demo",
        output_path=out_path,
        panel_caption_font_size_override=40.0,
        direction_labels_by_row=["← Favours Baseline | Favours Tuned →"],
    )

    assert ok is True
    output_svg = out_path.read_text(encoding="utf-8")
    assert "← Favours Baseline | Favours Tuned →" in output_svg
    assert 'font-size="40.0"' in output_svg


def test_resolve_shared_x_label_baseline_y_reserves_space_for_legend_obstacle(
    publication_figures_module,
):
    baseline_y = publication_figures_module._resolve_shared_x_label_baseline_y(
        row_bottom=860.0,
        obstacle_top=942.0,
        font_size=44.0,
    )

    assert baseline_y == pytest.approx(913.32)
    assert baseline_y > 860.0
    assert baseline_y < 942.0


def test_resolve_shared_x_label_baseline_y_reserves_space_for_next_row_obstacle(
    publication_figures_module,
):
    baseline_y = publication_figures_module._resolve_shared_x_label_baseline_y(
        row_bottom=860.0,
        obstacle_top=920.0,
        font_size=44.0,
    )

    assert baseline_y == pytest.approx(902.32)
    assert baseline_y > 860.0
    assert baseline_y < 920.0


def test_build_pairs_includes_baseline_reference_improvement(publication_figures_module):
    df = pd.DataFrame(
        {
            "dataset": ["d1", "d1"],
            "pair_sampling": ["full", "full"],
            "pair_seed": [7, 7],
            "pair_split": ["holdout", "holdout"],
            "algorithm": ["colors", "batch"],
            "metric_x": [20.0, 10.0],
        }
    )

    pairs = publication_figures_module._build_pairs(
        df=df,
        metric="metric_x",
        compare_col="algorithm",
        value_a="colors",
        value_b="batch",
        key_cols=["dataset", "pair_sampling", "pair_seed", "pair_split"],
        top_k=1,
    )

    assert len(pairs) == 1
    row = pairs.iloc[0]

    # score_a=20, score_b=10 => numerator = -10.
    assert row["pct_improvement_a_over_b"] == pytest.approx((-10.0 / 20.0) * 100.0)
    assert row["pct_improvement_a_over_b_baseline_ref"] == pytest.approx((-10.0 / 10.0) * 100.0)


def test_optimal_parameter_reporting_outputs_sampling_and_overall_tables(tmp_path, publication_figures_module):
    df = pd.DataFrame(
        [
            {
                "dataset": "d1",
                "pair_sampling": "full",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_seed": 7,
                "trial_number": 9,
                "balanced_qe_raw": 0.40,
                "param_learning_rate": 0.3,
                "param_initialization_method": "random",
            },
            {
                "dataset": "d1",
                "pair_sampling": "full",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_seed": 7,
                "trial_number": 3,
                "balanced_qe_raw": 0.40,
                "param_learning_rate": 0.2,
                "param_initialization_method": "pca",
            },
            {
                "dataset": "d1",
                "pair_sampling": "full",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_seed": 8,
                "trial_number": 4,
                "balanced_qe_raw": 0.30,
                "param_learning_rate": 0.4,
                "param_initialization_method": "random",
            },
            {
                "dataset": "d1",
                "pair_sampling": "full",
                "architecture": "mst",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_seed": 7,
                "trial_number": 4,
                "balanced_qe_raw": 0.35,
                "param_learning_rate": 0.1,
                "param_initialization_method": "pca",
            },
            {
                "dataset": "d2",
                "pair_sampling": "random",
                "architecture": "rng",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_seed": 11,
                "trial_number": 5,
                "balanced_qe_raw": 0.27,
                "param_learning_rate": 0.5,
                "param_initialization_method": "random",
            },
            # Should be excluded from dataset-level selection by global-label filter.
            {
                "dataset": "GLOBAL_OVERALL",
                "pair_sampling": "full",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_seed": 99,
                "trial_number": 1,
                "balanced_qe_raw": 0.01,
                "param_learning_rate": 0.9,
            },
        ]
    )

    generated_files: list[str] = []
    summary = publication_figures_module._run_optimal_parameter_reporting(
        tuned_df=df,
        run_output_dir=tmp_path,
        generated_files=generated_files,
        metric_col="balanced_qe_raw",
    )

    dataset_csv = Path(summary["dataset_topology_table_csv"])
    dataset_type_csv = Path(summary["dataset_type_topology_table_csv"])
    overall_csv = Path(summary["overall_table_csv"])
    markdown_path = Path(summary["markdown_report"])

    assert dataset_csv.exists()
    assert dataset_type_csv.exists()
    assert overall_csv.exists()
    assert markdown_path.exists()

    dataset_table = pd.read_csv(dataset_csv)
    dataset_type_table = pd.read_csv(dataset_type_csv)
    overall_table = pd.read_csv(overall_csv)

    # Aggregation is applied on per-seed best rows:
    # seed 7 picks trial_number=3 (lr=0.2), seed 8 picks lr=0.4, so mean=0.3.
    selected = dataset_table[
        (dataset_table["sampling_mode"] == "full")
        & (dataset_table["dataset"] == "d1")
        & (dataset_table["topology"] == "hexagonal")
    ].iloc[0]
    assert int(selected["rows_aggregated"]) == 2
    assert float(selected["best_balanced_qe_raw"]) == pytest.approx(0.35)
    assert float(selected["param_learning_rate"]) == pytest.approx(0.30)
    # Categorical aggregation uses mode with deterministic tie-break.
    assert str(selected["param_initialization_method"]) == "pca"

    # Global synthetic labels must be excluded from dataset-level table.
    assert "GLOBAL_OVERALL" not in set(dataset_table["dataset"].astype(str))

    # Dataset-type table should be present and grouped as real/synthetic labels.
    assert {"sampling_mode", "dataset_type", "topology"}.issubset(set(dataset_type_table.columns))

    # Overall table should include stratified rows by sampling+topology.
    assert {"sampling_mode", "topology", "best_balanced_qe_raw"}.issubset(set(overall_table.columns))

    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert "Sampling: Full" in markdown_text
    assert "Per-Sampling Best by Dataset Type and Topology" in markdown_text
    assert "Overall Best (Stratified by Topology and Sampling Method)" in markdown_text


def test_optimal_parameter_reporting_excludes_minibatch_rows(tmp_path, publication_figures_module):
    df = pd.DataFrame(
        [
            {
                "dataset": "d1",
                "pair_sampling": "full",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_seed": 7,
                "trial_number": 9,
                "balanced_qe_raw": 0.40,
                "param_learning_rate": 0.3,
            },
            {
                "dataset": "d1",
                "pair_sampling": "full",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "minibatch",
                "pair_seed": 7,
                "trial_number": 1,
                "balanced_qe_raw": 0.01,
                "param_learning_rate": 0.99,
            },
        ]
    )

    generated_files: list[str] = []
    summary = publication_figures_module._run_optimal_parameter_reporting(
        tuned_df=df,
        run_output_dir=tmp_path,
        generated_files=generated_files,
        metric_col="balanced_qe_raw",
    )

    dataset_table = pd.read_csv(Path(summary["dataset_topology_table_csv"]))
    assert sorted(dataset_table["batch_mode"].astype(str).str.lower().unique().tolist()) == ["full_batch"]
    assert float(dataset_table["best_balanced_qe_raw"].iloc[0]) == pytest.approx(0.40)
    assert int(summary["rows_excluded_minibatch"]) == 1
    assert int(summary["rows_used"]) == 1


def test_build_selected_parameter_stability_table_derives_overall_from_selected(publication_figures_module):
    source = pd.DataFrame(
        [
            {
                "global_label": "GLOBAL",
                "topology_left": "hexagonal",
                "topology_right": "rng",
                "parameter": "param_initial_radius",
                "parameter_type": "numeric",
                "seed_pairs_left": 300,
                "seed_pairs_right": 300,
                "stability_score_mean_left": 0.280681,
                "stability_score_mean_right": 0.225434,
                "stability_score_median_left": 0.255112,
                "stability_score_median_right": 0.164672,
            },
            {
                "global_label": "GLOBAL",
                "topology_left": "hexagonal",
                "topology_right": "rng",
                "parameter": "param_initialization_method",
                "parameter_type": "categorical",
                "seed_pairs_left": 300,
                "seed_pairs_right": 300,
                "stability_score_mean_left": 0.673333,
                "stability_score_mean_right": 0.403333,
                "stability_score_median_left": 1.0,
                "stability_score_median_right": 0.0,
            },
            {
                "global_label": "GLOBAL",
                "topology_left": "hexagonal",
                "topology_right": "rng",
                "parameter": "param_radius_decay_type",
                "parameter_type": "categorical",
                "seed_pairs_left": 300,
                "seed_pairs_right": 300,
                "stability_score_mean_left": 0.0566667,
                "stability_score_mean_right": 0.0,
                "stability_score_median_left": 0.0,
                "stability_score_median_right": 0.0,
            },
            {
                "global_label": "GLOBAL",
                "topology_left": "hexagonal",
                "topology_right": "rng",
                "parameter": "param_use_momentum",
                "parameter_type": "categorical",
                "seed_pairs_left": 300,
                "seed_pairs_right": 300,
                "stability_score_mean_left": 0.32,
                "stability_score_mean_right": 0.286667,
                "stability_score_median_left": 0.0,
                "stability_score_median_right": 0.0,
            },
            # Noise row that should be ignored by selected-parameter filter.
            {
                "global_label": "GLOBAL",
                "topology_left": "hexagonal",
                "topology_right": "rng",
                "parameter": "param_chunk_size",
                "parameter_type": "numeric",
                "seed_pairs_left": 60,
                "seed_pairs_right": 59,
                "stability_score_mean_left": 0.488013,
                "stability_score_mean_right": 0.457164,
                "stability_score_median_left": 0.476602,
                "stability_score_median_right": 0.50431,
            },
        ]
    )

    selected = [
        "param_initial_radius",
        "param_initialization_method",
        "param_radius_decay_type",
        "param_use_momentum",
    ]
    out = publication_figures_module._build_selected_parameter_stability_table(
        source,
        selected_parameters=selected,
        topology_left="hexagonal",
        topology_right="rng",
    )

    assert list(out["parameter"].astype(str)) == selected + ["OVERALL_SELECTED"]
    overall = out[out["parameter"] == "OVERALL_SELECTED"].iloc[0]

    expected_left = (0.280681 + 0.673333 + 0.0566667 + 0.32) / 4.0
    expected_right = (0.225434 + 0.403333 + 0.0 + 0.286667) / 4.0
    assert float(overall["stability_score_mean_left"]) == pytest.approx(expected_left)
    assert float(overall["stability_score_mean_right"]) == pytest.approx(expected_right)
    assert float(overall["delta_left_minus_right"]) == pytest.approx(expected_left - expected_right)
    assert str(overall["winner"]) == "rng"
    assert int(overall["parameters_compared"]) == 4


def test_build_selected_parameter_stability_three_topology_table_derives_overall(publication_figures_module):
    hex_mst = pd.DataFrame(
        [
            {
                "parameter": "param_initial_radius",
                "stability_score_mean_left": 0.28,
                "stability_score_mean_right": 0.31,
            },
            {
                "parameter": "param_initialization_method",
                "stability_score_mean_left": 0.67,
                "stability_score_mean_right": 0.42,
            },
            {
                "parameter": "param_radius_decay_type",
                "stability_score_mean_left": 0.05,
                "stability_score_mean_right": 0.02,
            },
            {
                "parameter": "param_use_momentum",
                "stability_score_mean_left": 0.32,
                "stability_score_mean_right": 0.29,
            },
        ]
    )
    hex_rng = pd.DataFrame(
        [
            {
                "parameter": "param_initial_radius",
                "stability_score_mean_left": 0.28,
                "stability_score_mean_right": 0.22,
            },
            {
                "parameter": "param_initialization_method",
                "stability_score_mean_left": 0.67,
                "stability_score_mean_right": 0.40,
            },
            {
                "parameter": "param_radius_decay_type",
                "stability_score_mean_left": 0.05,
                "stability_score_mean_right": 0.00,
            },
            {
                "parameter": "param_use_momentum",
                "stability_score_mean_left": 0.32,
                "stability_score_mean_right": 0.28,
            },
        ]
    )

    selected = [
        "param_initial_radius",
        "param_initialization_method",
        "param_radius_decay_type",
        "param_use_momentum",
    ]
    out = publication_figures_module._build_selected_parameter_stability_three_topology_table(
        parameter_global_hex_mst_df=hex_mst,
        parameter_global_hex_rng_df=hex_rng,
        selected_parameters=selected,
    )

    assert list(out["parameter"].astype(str)) == selected + ["OVERALL_SELECTED"]
    overall = out[out["parameter"] == "OVERALL_SELECTED"].iloc[0]
    assert float(overall["stability_score_mean_hexagonal"]) == pytest.approx((0.28 + 0.67 + 0.05 + 0.32) / 4.0)
    assert float(overall["stability_score_mean_mst"]) == pytest.approx((0.31 + 0.42 + 0.02 + 0.29) / 4.0)
    assert float(overall["stability_score_mean_rng"]) == pytest.approx((0.22 + 0.40 + 0.00 + 0.28) / 4.0)


def test_build_dataset_type_stability_three_topology_table(publication_figures_module):
    hex_mst = pd.DataFrame(
        [
            {
                "dataset_type": "synthetic",
                "parameter": "OVERALL",
                "stability_score_mean_left": 0.30,
                "stability_score_mean_right": 0.24,
            },
            {
                "dataset_type": "real",
                "parameter": "OVERALL",
                "stability_score_mean_left": 0.44,
                "stability_score_mean_right": 0.31,
            },
        ]
    )
    hex_rng = pd.DataFrame(
        [
            {
                "dataset_type": "synthetic",
                "parameter": "OVERALL",
                "stability_score_mean_left": 0.32,
                "stability_score_mean_right": 0.21,
            },
            {
                "dataset_type": "real",
                "parameter": "OVERALL",
                "stability_score_mean_left": 0.46,
                "stability_score_mean_right": 0.28,
            },
        ]
    )

    out = publication_figures_module._build_dataset_type_stability_three_topology_table(
        parameter_dataset_type_hex_mst_df=hex_mst,
        parameter_dataset_type_hex_rng_df=hex_rng,
    )

    assert list(out["parameter"].astype(str)) == [
        "DATASET_TYPE_SYNTHETIC",
        "DATASET_TYPE_REAL",
        "OVERALL_DATASET_TYPE",
    ]
    assert list(out["parameter_label"].astype(str)) == [
        "Synthetic",
        "Non-synthetic",
        "Overall (Synthetic + Non-synthetic)",
    ]
    synthetic = out[out["parameter"] == "DATASET_TYPE_SYNTHETIC"].iloc[0]
    real = out[out["parameter"] == "DATASET_TYPE_REAL"].iloc[0]
    overall = out[out["parameter"] == "OVERALL_DATASET_TYPE"].iloc[0]

    assert float(synthetic["stability_score_mean_hexagonal"]) == pytest.approx((0.30 + 0.32) / 2.0)
    assert float(synthetic["stability_score_mean_mst"]) == pytest.approx(0.24)
    assert float(synthetic["stability_score_mean_rng"]) == pytest.approx(0.21)
    assert float(real["stability_score_mean_hexagonal"]) == pytest.approx((0.44 + 0.46) / 2.0)
    assert float(real["stability_score_mean_mst"]) == pytest.approx(0.31)
    assert float(real["stability_score_mean_rng"]) == pytest.approx(0.28)
    assert float(overall["stability_score_mean_hexagonal"]) == pytest.approx(
        (float(synthetic["stability_score_mean_hexagonal"]) + float(real["stability_score_mean_hexagonal"])) / 2.0
    )

def test_build_dataset_selected_parameter_stability_three_topology_table(publication_figures_module):
    hex_mst = pd.DataFrame(
        [
            {
                "dataset": "iris",
                "parameter": "param_initial_radius",
                "stability_score_mean_left": 0.20,
                "stability_score_mean_right": 0.24,
            },
            {
                "dataset": "iris",
                "parameter": "param_initialization_method",
                "stability_score_mean_left": 0.40,
                "stability_score_mean_right": 0.36,
            },
            {
                "dataset": "wine",
                "parameter": "param_initial_radius",
                "stability_score_mean_left": 0.30,
                "stability_score_mean_right": 0.28,
            },
            {
                "dataset": "wine",
                "parameter": "param_initialization_method",
                "stability_score_mean_left": 0.50,
                "stability_score_mean_right": 0.44,
            },
        ]
    )
    hex_rng = pd.DataFrame(
        [
            {
                "dataset": "iris",
                "parameter": "param_initial_radius",
                "stability_score_mean_left": 0.22,
                "stability_score_mean_right": 0.18,
            },
            {
                "dataset": "iris",
                "parameter": "param_initialization_method",
                "stability_score_mean_left": 0.42,
                "stability_score_mean_right": 0.30,
            },
            {
                "dataset": "wine",
                "parameter": "param_initial_radius",
                "stability_score_mean_left": 0.32,
                "stability_score_mean_right": 0.20,
            },
            {
                "dataset": "wine",
                "parameter": "param_initialization_method",
                "stability_score_mean_left": 0.52,
                "stability_score_mean_right": 0.40,
            },
        ]
    )

    out = publication_figures_module._build_dataset_selected_parameter_stability_three_topology_table(
        parameter_dataset_hex_mst_df=hex_mst,
        parameter_dataset_hex_rng_df=hex_rng,
        selected_parameters=["param_initial_radius", "param_initialization_method"],
    )

    assert list(out["dataset"].astype(str)) == ["iris", "wine"]
    iris = out[out["dataset"] == "iris"].iloc[0]
    wine = out[out["dataset"] == "wine"].iloc[0]

    assert int(iris["n_selected_parameters"]) == 2
    assert float(iris["stability_score_mean_hexagonal"]) == pytest.approx(((0.20 + 0.22) / 2.0 + (0.40 + 0.42) / 2.0) / 2.0)
    assert float(iris["stability_score_mean_mst"]) == pytest.approx((0.24 + 0.36) / 2.0)
    assert float(iris["stability_score_mean_rng"]) == pytest.approx((0.18 + 0.30) / 2.0)

    assert int(wine["n_selected_parameters"]) == 2
    assert float(wine["stability_score_mean_hexagonal"]) == pytest.approx(((0.30 + 0.32) / 2.0 + (0.50 + 0.52) / 2.0) / 2.0)
    assert float(wine["stability_score_mean_mst"]) == pytest.approx((0.28 + 0.44) / 2.0)
    assert float(wine["stability_score_mean_rng"]) == pytest.approx((0.20 + 0.40) / 2.0)


def test_sync_default_aware_assets_to_paper_copies_stability_and_figure8(
    tmp_path,
    monkeypatch,
    publication_figures_module,
):
    assets_dir = tmp_path / "paper_assets"
    (assets_dir / "figures").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(publication_figures_module, "_resolve_paper_assets_dir", lambda: assets_dir)

    src_selected = tmp_path / "fig_stability_selected_params_hexagonal_mst_rng_full_vs_random_balanced_qe_raw.svg"
    src_selected.write_text("<svg/>", encoding="utf-8")
    src_figure8 = tmp_path / "fig_hyperparameter_stability_synthetic_vs_nonsynthetic.svg"
    src_figure8.write_text("<svg/>", encoding="utf-8")

    generated_files: list[str] = []
    summary = publication_figures_module._sync_default_aware_assets_to_paper(
        default_aware_analysis={
            "enabled": True,
            "selected_stability_publication": {
                "figures": {"balanced_qe_raw": str(src_selected)},
                "figure_8": str(src_figure8),
            },
            "tuned_vs_default_variants": [],
        },
        generated_files=generated_files,
    )

    copied = summary["figures"]
    assert summary["copied"] is True
    assert "selected_stability:balanced_qe_raw" in copied
    assert "selected_stability:figure_8" in copied
    assert (assets_dir / "figures" / src_selected.name).exists()
    assert (assets_dir / "figures" / src_figure8.name).exists()


def test_sync_default_aware_assets_to_paper_copies_topology_specific_tuned_vs_default_figures(
    tmp_path,
    monkeypatch,
    publication_figures_module,
):
    assets_dir = tmp_path / "paper_assets"
    (assets_dir / "figures").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(publication_figures_module, "_resolve_paper_assets_dir", lambda: assets_dir)

    src_main = tmp_path / "fig_main.svg"
    src_hex = tmp_path / "fig_hex.svg"
    src_mst = tmp_path / "fig_mst.svg"
    src_rng = tmp_path / "fig_rng.svg"
    for path in [src_main, src_hex, src_mst, src_rng]:
        path.write_text("<svg/>", encoding="utf-8")

    generated_files: list[str] = []
    summary = publication_figures_module._sync_default_aware_assets_to_paper(
        default_aware_analysis={
            "enabled": True,
            "selected_stability_publication": {},
            "tuned_vs_default_variants": [
                {"key": "selected_source", "publication": {"combined_figure": str(src_main)}},
                {"key": "selected_source_hex", "publication": {"combined_figure": str(src_hex)}},
                {"key": "selected_source_mst", "publication": {"combined_figure": str(src_mst)}},
                {"key": "selected_source_rng", "publication": {"combined_figure": str(src_rng)}},
            ],
        },
        generated_files=generated_files,
    )

    copied = summary["figures"]
    assert summary["copied"] is True
    assert "selected_source:combined_figure" in copied
    assert "selected_source_hex:combined_figure" in copied
    assert "selected_source_mst:combined_figure" in copied
    assert "selected_source_rng:combined_figure" in copied
    assert (assets_dir / "figures" / "fig_tuned_vs_true_default_metrics.svg").exists()
    assert (assets_dir / "figures" / "supp_fig_tuned_vs_true_default_hex_metrics.svg").exists()
    assert (assets_dir / "figures" / "supp_fig_tuned_vs_true_default_mst_metrics.svg").exists()
    assert (assets_dir / "figures" / "supp_fig_tuned_vs_true_default_rng_metrics.svg").exists()


def test_sync_default_aware_stats_to_manuscript_updates_live_counts(
    tmp_path,
    monkeypatch,
    publication_figures_module,
):
    assets_dir = tmp_path / "paper_assets"
    (assets_dir / "figures").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(publication_figures_module, "_resolve_paper_assets_dir", lambda: assets_dir)

    manuscript_path = tmp_path / "manuscript.md"
    manuscript_path.write_text(
        "\n".join(
            [
                "Figure 6 intro.",
                "<!-- AUTO-DEFAULT-AWARE-FIGURE6-STATS:START -->",
                "stale fig6 block",
                "<!-- AUTO-DEFAULT-AWARE-FIGURE6-STATS:END -->",
                "At the pooled overall level, the paired summaries across all matched tuned/default pairs also favor tuning for all three metrics, consistent with the per-dataset pattern in Fig. 6.",
                "<!-- AUTO-DEFAULT-AWARE-TOPOLOGY-STATS:START -->",
                "stale topology block",
                "<!-- AUTO-DEFAULT-AWARE-TOPOLOGY-STATS:END -->",
                "### 7.4 Tuning benefit under matched defaults",
                "<!-- AUTO-DEFAULT-AWARE-DISCUSSION:START -->",
                "stale discussion block",
                "<!-- AUTO-DEFAULT-AWARE-DISCUSSION:END -->",
                "To our knowledge, comprehensive cross-topology and cross-sampling evaluations of chosen SOM hyperparameters have been limited in prior large-scale deployment-oriented studies. In this context, the present paired results make the practical point clear: hyperparameter choice has a large impact on achieved performance.",
                "## 8. Conclusion",
                "<!-- AUTO-DEFAULT-AWARE-CONCLUSION:START -->",
                "stale conclusion block",
                "<!-- AUTO-DEFAULT-AWARE-CONCLUSION:END -->",
                "Taken together, the best observed operating profile in this study uses the maximum practical GPU count supported by adequate file I/O, RNG topology, and the derived default hyperparameters, with sampling chosen by scale: full for smaller datasets when stability is critical, and random as a practical throughput option in the larger-dataset regime (>10000 samples) where paired QE differences are not meaningfully detected.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    variant_dir = tmp_path / "default_aware" / "tuned_vs_default"
    tables_dir = variant_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "pair_dataset": "d1",
                "pair_seed": "1",
                "pair_sampling": "full",
                "pair_topology": "hexagonal",
                "default_value": 10.0,
                "tuned_value": 8.0,
                "pct_improvement": 20.0,
            },
            {
                "pair_dataset": "d2",
                "pair_seed": "2",
                "pair_sampling": "random",
                "pair_topology": "rng",
                "default_value": 10.0,
                "tuned_value": 10.0,
                "pct_improvement": 0.0,
            },
        ]
    ).to_csv(tables_dir / "tuned_vs_default_pairs_balanced_qe_raw.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair_dataset": "d1",
                "pair_seed": "1",
                "pair_sampling": "full",
                "pair_topology": "hexagonal",
                "default_value": 10.0,
                "tuned_value": 7.0,
                "pct_improvement": 30.0,
            },
            {
                "pair_dataset": "d2",
                "pair_seed": "2",
                "pair_sampling": "random",
                "pair_topology": "rng",
                "default_value": 10.0,
                "tuned_value": 12.0,
                "pct_improvement": -20.0,
            },
        ]
    ).to_csv(tables_dir / "tuned_vs_default_pairs_qe_holdout.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair_dataset": "d1",
                "pair_seed": "1",
                "pair_sampling": "full",
                "pair_topology": "hexagonal",
                "default_value": 10.0,
                "tuned_value": 6.0,
                "pct_improvement": 40.0,
            },
            {
                "pair_dataset": "d2",
                "pair_seed": "2",
                "pair_sampling": "random",
                "pair_topology": "rng",
                "default_value": 10.0,
                "tuned_value": 9.0,
                "pct_improvement": 10.0,
            },
        ]
    ).to_csv(tables_dir / "tuned_vs_default_pairs_qe_train.csv", index=False)

    variant_hex_dir = tmp_path / "default_aware" / "tuned_vs_default_hex"
    variant_hex_tables_dir = variant_hex_dir / "tables"
    variant_hex_tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"pct_improvement": 20.0},
            {"pct_improvement": 10.0},
        ]
    ).to_csv(variant_hex_tables_dir / "tuned_vs_default_pairs_balanced_qe_raw.csv", index=False)

    variant_mst_dir = tmp_path / "default_aware" / "tuned_vs_default_mst"
    variant_mst_tables_dir = variant_mst_dir / "tables"
    variant_mst_tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"pct_improvement": 18.0},
            {"pct_improvement": 12.0},
        ]
    ).to_csv(variant_mst_tables_dir / "tuned_vs_default_pairs_balanced_qe_raw.csv", index=False)

    variant_rng_dir = tmp_path / "default_aware" / "tuned_vs_default_rng"
    variant_rng_tables_dir = variant_rng_dir / "tables"
    variant_rng_tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"pct_improvement": 30.0},
            {"pct_improvement": 20.0},
        ]
    ).to_csv(variant_rng_tables_dir / "tuned_vs_default_pairs_balanced_qe_raw.csv", index=False)

    summary = publication_figures_module._sync_default_aware_stats_to_manuscript(
        {
            "enabled": True,
            "tuned_vs_default_variants": [
                {
                    "key": "selected_source",
                    "output_dir": str(variant_dir),
                },
                {
                    "key": "selected_source_hex",
                    "output_dir": str(variant_hex_dir),
                },
                {
                    "key": "selected_source_mst",
                    "output_dir": str(variant_mst_dir),
                },
                {
                    "key": "selected_source_rng",
                    "output_dir": str(variant_rng_dir),
                },
            ],
        }
    )

    updated_text = manuscript_path.read_text(encoding="utf-8")
    assert summary["updated"] is True
    assert (
        "The tuned-versus-default pairing results in Fig. 6 show the same direction across the $QE$ endpoints, "
        "based on n=2 paired comparisons per $QE$ endpoint (n=6 total across all $QE$ variants "
        "$QE_B$/$QE_H$/$QE_T$) from 2 datasets, 2 seeds, and the full and random sampling modes."
    ) in updated_text
    assert "The current matched Figure 6 source spans topologies hexagonal and rng." in updated_text
    assert (
        "Across the matched pairs, tuned settings improve Balanced QE in 1/2 pairs (1 ties), "
        "with median and mean improvements of 10.00% and 10.00%; Holdout QE in 1/2 pairs (1 worse), "
        "with median and mean improvements of 5.00% and 5.00%; and Train QE in 2/2 pairs, "
        "with median and mean improvements of 25.00% and 25.00%."
    ) in updated_text
    assert "Holdout QE in 1/2 pairs" in updated_text
    assert "default-aware analyses over n=2 paired comparisons per $QE$ endpoint (n=6 total across all $QE$ variants $QE_B$/$QE_H$/$QE_T$)" in updated_text
    assert (
        "The same tuning pattern is observed across topologies: mean Balanced-QE improvement is positive for "
        "hexagonal (15.00%), MST (15.00%), and RNG (25.00%), indicating that tuning affects all topology families "
        "rather than a single-architecture artifact."
    ) in updated_text
    assert (
        "Across the matched Fig. 6 comparisons, tuned defaults consistently produce better QE results than "
        "untuned defaults. This suggests that tuning should be treated as part of the method configuration "
        "rather than as optional post-processing."
    ) in updated_text
    assert "`n=2`" not in updated_text
    assert "`1/2`" not in updated_text
    assert "`15.00%`" not in updated_text


def test_sync_topology_pvalue_summary_to_paper_and_manuscript(
    tmp_path,
    monkeypatch,
    publication_figures_module,
):
    assets_dir = tmp_path / "paper_assets"
    (assets_dir / "tables").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(publication_figures_module, "_resolve_paper_assets_dir", lambda: assets_dir)

    manuscript_path = tmp_path / "manuscript.md"
    manuscript_path.write_text(
        "\n".join(
            [
                "### 5.3 MST Results",
                "<!-- AUTO-TOPOLOGY-MST-PVALUES:START -->",
                "stale mst block",
                "<!-- AUTO-TOPOLOGY-MST-PVALUES:END -->",
                "![Figure 4](assets_manual/figures/fig_4.svg)",
                "### 5.4 RNG Results",
                "<!-- AUTO-TOPOLOGY-RNG-PVALUES:START -->",
                "stale rng block",
                "<!-- AUTO-TOPOLOGY-RNG-PVALUES:END -->",
                "![Figure 5](assets_manual/figures/fig_5.svg)",
                "## Supplementary Tables (End Matter)",
                "**Supplementary Table S4. Existing entry.**",
                "## Supplementary Figures (End Matter)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    raw_root = tmp_path / "raw"
    tables_dir = raw_root / "sampling" / "full" / "topology_main" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    publication_dir = raw_root / "publication_figures"
    publication_dir.mkdir(parents=True, exist_ok=True)

    for comparison_slug, rows_by_metric in {
        "hex_vs_mst": {
            "balanced_qe_raw": [
                {"dataset": "iris", "p_value": 0.0123},
                {"dataset": "wine", "p_value": 0.0456},
                {"dataset": "GLOBAL_OVERALL", "p_value": 0.00091},
                {"dataset": "GLOBAL_REAL", "p_value": 0.22},
            ],
            "quantization_error_holdout": [
                {"dataset": "iris", "p_value": 0.11},
                {"dataset": "wine", "p_value": 0.22},
                {"dataset": "GLOBAL_OVERALL", "p_value": 0.0032},
            ],
            "quantization_error_train": [
                {"dataset": "iris", "p_value": 0.21},
                {"dataset": "wine", "p_value": 0.31},
                {"dataset": "GLOBAL_OVERALL", "p_value": 0.0045},
            ],
        },
        "hex_vs_rng": {
            "balanced_qe_raw": [
                {"dataset": "iris", "p_value": 0.0012},
                {"dataset": "wine", "p_value": 0.0023},
                {"dataset": "GLOBAL_OVERALL", "p_value": 0.00012},
            ],
            "quantization_error_holdout": [
                {"dataset": "iris", "p_value": 0.015},
                {"dataset": "wine", "p_value": 0.025},
                {"dataset": "GLOBAL_OVERALL", "p_value": 0.00034},
            ],
            "quantization_error_train": [
                {"dataset": "iris", "p_value": 0.018},
                {"dataset": "wine", "p_value": 0.028},
                {"dataset": "GLOBAL_OVERALL", "p_value": 0.00056},
            ],
        },
    }.items():
        for metric_slug, rows in rows_by_metric.items():
            pd.DataFrame(rows).to_csv(tables_dir / f"{comparison_slug}_main_{metric_slug}.csv", index=False)

    generated_files: list[str] = []
    summary = publication_figures_module._sync_topology_pvalue_summary_to_paper_and_manuscript(
        suite_outputs={
            "raw": {
                "root": str(raw_root),
                "publication_figures": str(publication_dir),
            }
        },
        generated_files=generated_files,
    )

    updated_text = manuscript_path.read_text(encoding="utf-8")
    assert summary["generated"] is True
    assert "Balanced QE (p=0.00091); Holdout QE (p=0.0032); and Train QE (p=0.0045)." in updated_text
    assert "Balanced QE (p=0.00012); Holdout QE (p=0.00034); and Train QE (p=0.00056)." in updated_text
    assert "Supplementary Table S5" in updated_text
    assert "`assets/tables/supp_table_topology_hex_vs_mst_rng_pvalues.tsv`" in updated_text

    paper_table_path = assets_dir / "tables" / "supp_table_topology_hex_vs_mst_rng_pvalues.tsv"
    assert paper_table_path.exists()
    table_df = pd.read_csv(paper_table_path, sep="\t")
    assert list(table_df.columns) == ["metric", "dataset", "MST", "RNG"]
    overall_balanced = table_df[(table_df["metric"] == "Balanced QE") & (table_df["dataset"] == "OVERALL")].iloc[0]
    assert overall_balanced["MST"] == "p=0.00091"
    assert overall_balanced["RNG"] == "p=0.00012"
    assert "GLOBAL_REAL" not in set(table_df["dataset"].astype(str))


def test_sync_systems_scaling_stats_to_manuscript(
    tmp_path,
    monkeypatch,
    publication_figures_module,
):
    assets_dir = tmp_path / "paper_assets"
    (assets_dir / "tables").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(publication_figures_module, "_resolve_paper_assets_dir", lambda: assets_dir)

    manuscript_path = tmp_path / "manuscript.md"
    manuscript_path.write_text(
        "\n".join(
            [
                "### 6.2 Multi-GPU topology scaling and OOM context",
                "![Figure 9](assets_manual/figures/fig_9.svg)",
                "*Figure 9. Multi-GPU full-batch scaling across $G\\in\\{1,2,4,8\\}$ GPUs. Panels A-C show runtime (s) for dimension-, sample-, and grid-size-scaling workloads, respectively. Panels D-F show scaling efficiency for the same workloads, computed from the single-GPU baseline and the corresponding $G$-GPU runtime. Runtime error bars denote $\\pm 1$ standard deviation across $n=3$ repeated runs per configuration; the 100\\% efficiency reference line indicates ideal linear scaling.*",
                "<!-- AUTO-SYSTEMS-SCALING-STATS:START -->",
                "stale systems block",
                "<!-- AUTO-SYSTEMS-SCALING-STATS:END -->",
                "We interpret scaling efficiency using the standard single-GPU baseline-over-observed speedup definition.",
                "At fixed $G=8$ under the same harmonized full-batch scaling conditions, topology runtime is compared across hexagonal, MST, and RNG (Fig. 10).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    diagnostics_path = assets_dir / "tables" / "supp_table_figure_9_rng_scaling_diagnostics.tsv"
    pd.DataFrame(
        [
            {"mode_name": "dimension_scaling", "mode_label": "Dimension Scaling", "topology": "rng", "method": "batch", "axis_value": 500, "gpu_count": 1, "runtime_mean_s": 101.0, "runtime_std_s": 1.0, "n_repeats": 3, "log_count": 1, "staging_mode": "ram", "staging_modes": "ram", "any_repeat_disk": False, "all_repeats_disk": False},
            {"mode_name": "dimension_scaling", "mode_label": "Dimension Scaling", "topology": "rng", "method": "batch", "axis_value": 1000, "gpu_count": 1, "runtime_mean_s": 111.0, "runtime_std_s": 1.0, "n_repeats": 3, "log_count": 1, "staging_mode": "disk", "staging_modes": "disk", "any_repeat_disk": True, "all_repeats_disk": True},
            {"mode_name": "dimension_scaling", "mode_label": "Dimension Scaling", "topology": "rng", "method": "batch", "axis_value": 1000, "gpu_count": 2, "runtime_mean_s": 91.0, "runtime_std_s": 1.0, "n_repeats": 3, "log_count": 1, "staging_mode": "ram", "staging_modes": "ram", "any_repeat_disk": False, "all_repeats_disk": False},
            {"mode_name": "dimension_scaling", "mode_label": "Dimension Scaling", "topology": "rng", "method": "batch", "axis_value": 2000, "gpu_count": 2, "runtime_mean_s": 95.0, "runtime_std_s": 1.0, "n_repeats": 3, "log_count": 1, "staging_mode": "disk", "staging_modes": "disk", "any_repeat_disk": True, "all_repeats_disk": True},
            {"mode_name": "dimension_scaling", "mode_label": "Dimension Scaling", "topology": "rng", "method": "batch", "axis_value": 2000, "gpu_count": 4, "runtime_mean_s": 81.0, "runtime_std_s": 1.0, "n_repeats": 3, "log_count": 1, "staging_mode": "ram", "staging_modes": "ram", "any_repeat_disk": False, "all_repeats_disk": False},
            {"mode_name": "dimension_scaling", "mode_label": "Dimension Scaling", "topology": "rng", "method": "batch", "axis_value": 5000, "gpu_count": 4, "runtime_mean_s": 88.0, "runtime_std_s": 1.0, "n_repeats": 3, "log_count": 1, "staging_mode": "disk", "staging_modes": "disk", "any_repeat_disk": True, "all_repeats_disk": True},
            {"mode_name": "dimension_scaling", "mode_label": "Dimension Scaling", "topology": "rng", "method": "batch", "axis_value": 5000, "gpu_count": 8, "runtime_mean_s": 77.0, "runtime_std_s": 1.0, "n_repeats": 3, "log_count": 1, "staging_mode": "disk", "staging_modes": "disk", "any_repeat_disk": True, "all_repeats_disk": True},
            {"mode_name": "sample_scaling", "mode_label": "Sample Scaling", "topology": "rng", "method": "batch", "axis_value": 50000000, "gpu_count": 1, "runtime_mean_s": 201.0, "runtime_std_s": 2.0, "n_repeats": 3, "log_count": 1, "staging_mode": "ram", "staging_modes": "ram", "any_repeat_disk": False, "all_repeats_disk": False},
            {"mode_name": "sample_scaling", "mode_label": "Sample Scaling", "topology": "rng", "method": "batch", "axis_value": 100000000, "gpu_count": 1, "runtime_mean_s": 221.0, "runtime_std_s": 2.0, "n_repeats": 3, "log_count": 1, "staging_mode": "disk", "staging_modes": "disk", "any_repeat_disk": True, "all_repeats_disk": True},
            {"mode_name": "sample_scaling", "mode_label": "Sample Scaling", "topology": "rng", "method": "batch", "axis_value": 100000000, "gpu_count": 2, "runtime_mean_s": 171.0, "runtime_std_s": 2.0, "n_repeats": 3, "log_count": 1, "staging_mode": "ram", "staging_modes": "ram", "any_repeat_disk": False, "all_repeats_disk": False},
            {"mode_name": "sample_scaling", "mode_label": "Sample Scaling", "topology": "rng", "method": "batch", "axis_value": 500000000, "gpu_count": 2, "runtime_mean_s": 181.0, "runtime_std_s": 2.0, "n_repeats": 3, "log_count": 1, "staging_mode": "disk", "staging_modes": "disk", "any_repeat_disk": True, "all_repeats_disk": True},
            {"mode_name": "sample_scaling", "mode_label": "Sample Scaling", "topology": "rng", "method": "batch", "axis_value": 500000000, "gpu_count": 4, "runtime_mean_s": 141.0, "runtime_std_s": 2.0, "n_repeats": 3, "log_count": 1, "staging_mode": "disk", "staging_modes": "disk", "any_repeat_disk": True, "all_repeats_disk": True},
            {"mode_name": "sample_scaling", "mode_label": "Sample Scaling", "topology": "rng", "method": "batch", "axis_value": 1000000000, "gpu_count": 8, "runtime_mean_s": 543.21, "runtime_std_s": 2.0, "n_repeats": 3, "log_count": 1, "staging_mode": "disk", "staging_modes": "disk", "any_repeat_disk": True, "all_repeats_disk": True},
        ]
    ).to_csv(diagnostics_path, sep="\t", index=False)

    summary = publication_figures_module._sync_systems_scaling_stats_to_manuscript()

    updated_text = manuscript_path.read_text(encoding="utf-8")
    assert summary["updated"] is True
    assert "all tested GPU counts eventually transition into disk mode" in updated_text
    assert "1 GPU at 1,000 dimensions to 8 GPUs at 5,000 dimensions" in updated_text
    assert "1 GPU at 100,000,000 samples to 8 GPUs at 1,000,000,000 samples" in updated_text
    assert (
        "The 8-GPU RNG configuration processes 1,000,000,000 samples in 543.21 s (9.05 min), "
        "demonstrating billion-sample training at a runtime measured in minutes rather than hours "
        "even for a 50-feature, 1024-node network ($32 \\times 32 = 1024$) under multi-node "
        "distributed execution with data staged from shared non-local storage to node-local shards "
        "before training."
    ) in updated_text
    assert "stale systems block" not in updated_text


def test_sync_sampling_regression_stats_to_manuscript_uses_plain_text(
    tmp_path,
    monkeypatch,
    publication_figures_module,
):
    assets_dir = tmp_path / "paper_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(publication_figures_module, "_resolve_paper_assets_dir", lambda: assets_dir)

    manuscript_path = tmp_path / "manuscript.md"
    manuscript_path.write_text(
        "\n".join(
            [
                "Sampling intro.",
                "*Figure 3. Base caption. [[AUTO-SAMPLING-REGRESSION-STATS]]*",
                "<!-- AUTO-SAMPLING-REGRESSION-STATS:START -->",
                "stale sampling block",
                "<!-- AUTO-SAMPLING-REGRESSION-STATS:END -->",
                "### 5.3 MST Results",
                "",
            ]
        ),
        encoding="utf-8",
    )

    sampling_dir = tmp_path / "sampling_comparison"
    tables_dir = sampling_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = [
        ("balanced_qe_raw", -0.761, 0.00158, 14),
        ("quantization_error_holdout", -0.743, 0.00235, 14),
        ("quantization_error_train", -0.653, 0.0113, 14),
    ]
    for metric_slug, pearson_r, pearson_p, n_datasets in stats_rows:
        pd.DataFrame(
            [
                {
                    "pearson_r": pearson_r,
                    "pearson_r_p_value": pearson_p,
                    "n_datasets": n_datasets,
                }
            ]
        ).to_csv(
            tables_dir
            / f"sampling_mode_hexagonal_full_batch_full_vs_random_qe_vs_sample_size_regression_stats_{metric_slug}.csv",
            index=False,
        )

    pd.DataFrame(
        [
            {
                "dataset_index": 1,
                "dataset": "iris",
                "dimension_count": 4,
                "sample_size": 150,
                "dataset_type": "real",
            }
        ]
    ).to_csv(tables_dir / "figure_3_full_vs_random_dataset_metadata.csv", index=False)

    summary = publication_figures_module._sync_sampling_regression_stats_to_manuscript(
        {
            "raw": {
                "sampling_comparison": str(sampling_dir),
            }
        }
    )

    updated_text = manuscript_path.read_text(encoding="utf-8")
    assert summary["updated"] is True
    assert "Differences in QE between random and full stratified by dataset size (panels D-F):" in updated_text
    assert "Balanced QE (Pearson R=-0.761, p=0.00158, n=14)" in updated_text
    assert "Holdout QE (Pearson R=-0.743, p=0.00235, n=14)" in updated_text
    assert "Train QE (Pearson R=-0.653, p=0.0113, n=14)" in updated_text
    assert (
        "The corresponding dataset metadata table is listed in Supplementary Table S1."
    ) in updated_text
    assert "[[AUTO-SAMPLING-REGRESSION-STATS]]" not in updated_text
    assert "stale sampling block" not in updated_text
    assert "`R=" not in updated_text
    assert "assets/tables/supp_table_figure_2_sampling_dataset_metadata.csv" not in updated_text


def test_sync_figure10_topology_runtime_stats_to_manuscript_uses_generated_table(
    tmp_path,
    monkeypatch,
    publication_figures_module,
):
    assets_dir = tmp_path / "paper_assets"
    tables_dir = assets_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(publication_figures_module, "_resolve_paper_assets_dir", lambda: assets_dir)

    manuscript_path = tmp_path / "manuscript.md"
    manuscript_path.write_text(
        "\n".join(
            [
                "Figure 10 intro.",
                "<!-- AUTO-FIGURE10-TOPOLOGY-RUNTIME-STATS:START -->",
                "stale figure10 block",
                "<!-- AUTO-FIGURE10-TOPOLOGY-RUNTIME-STATS:END -->",
                "<!-- AUTO-FIGURE10-GRID-SIZE-DISCUSSION:START -->",
                "stale systems discussion block",
                "<!-- AUTO-FIGURE10-GRID-SIZE-DISCUSSION:END -->",
                "RNG runtime traces are interpreted jointly with the quality outcomes in Section 5.4, using the same harmonized benchmark construction described in Section 4.4.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "mode_name": "dimension_scaling",
                "mode_label": "Dimension Scaling",
                "gpu_count": 8,
                "method": "batch",
                "axis_value": 5000,
                "hexagonal_runtime_mean_s": 120.0,
                "mst_runtime_mean_s": 120.8,
                "rng_runtime_mean_s": 121.1,
                "fastest_topology": "hexagonal",
                "fastest_runtime_mean_s": 120.0,
                "slowest_topology": "rng",
                "slowest_runtime_mean_s": 121.1,
                "max_pairwise_runtime_spread_pct": 0.9166666667,
            },
            {
                "mode_name": "sample_scaling",
                "mode_label": "Sample Scaling",
                "gpu_count": 8,
                "method": "batch",
                "axis_value": 1000000000,
                "hexagonal_runtime_mean_s": 360.0,
                "mst_runtime_mean_s": 362.5,
                "rng_runtime_mean_s": 363.2,
                "fastest_topology": "hexagonal",
                "fastest_runtime_mean_s": 360.0,
                "slowest_topology": "rng",
                "slowest_runtime_mean_s": 363.2,
                "max_pairwise_runtime_spread_pct": 0.8888888889,
            },
            {
                "mode_name": "grid_size_scaling",
                "mode_label": "Grid-Size Scaling",
                "gpu_count": 8,
                "method": "batch",
                "axis_value": 64,
                "hexagonal_runtime_mean_s": 40.0,
                "mst_runtime_mean_s": 65.0,
                "rng_runtime_mean_s": 74.0,
                "fastest_topology": "hexagonal",
                "fastest_runtime_mean_s": 40.0,
                "slowest_topology": "rng",
                "slowest_runtime_mean_s": 74.0,
                "max_pairwise_runtime_spread_pct": 85.0,
            },
        ]
    ).to_csv(
        tables_dir / "supp_table_figure_10_topology_runtime_summary.tsv",
        index=False,
        sep="\t",
    )

    summary = publication_figures_module._sync_figure10_topology_runtime_stats_to_manuscript()

    updated_text = manuscript_path.read_text(encoding="utf-8")
    assert summary["updated"] is True
    assert (
        "In Fig. 10A-B, the topologies scale similarly as input complexity and data volume increase: "
        "even at the largest tested axis values, the maximum pairwise runtime spread remains modest at "
        "dimension scaling (0.92% at 5,000 dimensions); sample scaling (0.89% at "
        "1,000,000,000 samples)."
    ) in updated_text
    assert (
        "However, when the grid itself is enlarged in Fig. 10C, topology-dependent runtime differences "
        "become readily evident. At the largest tested grid size (grid size 64), the 8-GPU mean runtimes are "
        "40.00 s (0.67 min) for hexagonal, 65.00 s (1.08 min) for MST, and 74.00 s (1.23 min) for "
        "RNG, corresponding to 8-GPU MST and RNG runtimes that are 1.62x and 1.85x the hexagonal "
        "runtime, respectively."
    ) in updated_text
    assert "`0.92%`" not in updated_text
    assert "`5,000`" not in updated_text


def test_sync_figure11_deployment_runtime_stats_to_manuscript_uses_generated_table(
    tmp_path,
    monkeypatch,
    publication_figures_module,
):
    assets_dir = tmp_path / "paper_assets"
    tables_dir = assets_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(publication_figures_module, "_resolve_paper_assets_dir", lambda: assets_dir)

    manuscript_path = tmp_path / "manuscript.md"
    manuscript_path.write_text(
        "\n".join(
            [
                "*Figure 11. Integrated deployment comparison of default XPySOM versus tuned FloatSOM RNG. Panels A-C compare $QE_B$, $QE_H$, and $QE_T$ using the untuned XPySOM baseline against matched tuned FloatSOM RNG full-sampling runs. Panel D provides the scaling/runtime context for the same comparison, with the separately executed targeted 1B-sample runs discussed in the text rather than plotted directly. Taken together, this integrated figure summarizes the operating point observed for tuned FloatSOM RNG once workload size is large enough for steady-state execution to dominate startup overhead. Per-dataset and `GLOBAL_OVERALL` panel summaries are listed in Supplementary Table S6.*",
                "<!-- AUTO-FIG11-DEPLOYMENT-QE-STATS:START -->",
                "stale figure11 qe block",
                "<!-- AUTO-FIG11-DEPLOYMENT-QE-STATS:END -->",
                "For the XPySOM reference in Fig. 11, workloads beyond the $10^8$-sample case were not processed under this benchmark setup because the implementation ran out of memory.",
                "<!-- AUTO-FIG11-DEPLOYMENT-RUNTIME-STATS:START -->",
                "stale figure11 runtime block",
                "<!-- AUTO-FIG11-DEPLOYMENT-RUNTIME-STATS:END -->",
                "Alongside these figure-level scaling outputs, we use log-derived systems diagnostics as supporting evidence (not additional main figures): staging mode and worker throughput summaries, per-iteration timing breakdowns (submit/get/collective components), and OOM-avoidance stability notes under the largest workloads.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "mode_name": "dimension_scaling",
                "mode_label": "Dimension Scaling",
                "gpu_count": 8,
                "method": "batch",
                "axis_value": 5000,
                "hexagonal_runtime_mean_s": 404.75,
                "mst_runtime_mean_s": 423.71,
                "rng_runtime_mean_s": 410.50,
                "fastest_topology": "hexagonal",
                "fastest_runtime_mean_s": 404.75,
                "slowest_topology": "mst",
                "slowest_runtime_mean_s": 423.71,
                "max_pairwise_runtime_spread_pct": 4.68582615698,
            },
            {
                "mode_name": "sample_scaling",
                "mode_label": "Sample Scaling",
                "gpu_count": 8,
                "method": "batch",
                "axis_value": 1000000000,
                "hexagonal_runtime_mean_s": 364.01,
                "mst_runtime_mean_s": 375.88,
                "rng_runtime_mean_s": 369.84,
                "fastest_topology": "hexagonal",
                "fastest_runtime_mean_s": 364.01,
                "slowest_topology": "mst",
                "slowest_runtime_mean_s": 375.88,
                "max_pairwise_runtime_spread_pct": 3.2608578823,
            },
        ]
    ).to_csv(
        tables_dir / "supp_table_figure_10_topology_runtime_summary.tsv",
        index=False,
        sep="\t",
    )
    pd.DataFrame(
        [
            {"figure": "Figure 11", "topology": "rng", "metric": "QE_B", "dataset": "GLOBAL_OVERALL", "median_pct_change": 14.4850, "ci_low_pct": 12.7836, "ci_high_pct": 16.1865},
            {"figure": "Figure 11", "topology": "rng", "metric": "QE_H", "dataset": "GLOBAL_OVERALL", "median_pct_change": 9.0765, "ci_low_pct": 7.1203, "ci_high_pct": 11.0326},
            {"figure": "Figure 11", "topology": "rng", "metric": "QE_T", "dataset": "GLOBAL_OVERALL", "median_pct_change": 22.4609, "ci_low_pct": 19.4613, "ci_high_pct": 25.4605},
        ]
    ).to_csv(
        tables_dir / "supp_table_figure_11_xpysom_rng_deployment_summary.tsv",
        index=False,
        sep="\t",
    )

    summary = publication_figures_module._sync_figure11_deployment_runtime_stats_to_manuscript()

    updated_text = manuscript_path.read_text(encoding="utf-8")
    assert summary["updated"] is True
    assert (
        "At the overall level, Fig. 11 shows median percentage improvements of $QE_B$ (14.5%); "
        "$QE_H$ (9.1%); and $QE_T$ (22.5%) for tuned FloatSOM RNG relative to default XPySOM, "
        "capturing the combined deployment effect of topology choice and tuning on $QE$."
    ) in updated_text
    assert (
        "The targeted deployment-scale runs indicate that these quality gains are not purchased at the "
        "cost of a qualitatively different runtime profile: applying the tuned defaults used in Fig. 11 "
        "does not strongly alter the distributed scaling behavior established by the earlier results "
        "(Supplementary Table S9)."
    ) in updated_text
    assert "stale figure11 qe block" not in updated_text
    assert "stale figure11 runtime block" not in updated_text


def test_sync_default_aware_stability_regression_stats_to_manuscript_uses_plain_text(
    tmp_path,
    monkeypatch,
    publication_figures_module,
):
    assets_dir = tmp_path / "paper_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(publication_figures_module, "_resolve_paper_assets_dir", lambda: assets_dir)

    manuscript_path = tmp_path / "manuscript.md"
    manuscript_path.write_text(
        "\n".join(
            [
                "Stability intro.",
                "<!-- AUTO-DEFAULT-AWARE-STABILITY-REGRESSION:START -->",
                "stale stability block",
                "<!-- AUTO-DEFAULT-AWARE-STABILITY-REGRESSION:END -->",
                "![Figure 7](assets_manual/figures/fig_7.svg)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    publication_dir = tmp_path / "default_aware" / "publication_figures"
    publication_dir.mkdir(parents=True, exist_ok=True)
    metric_slug = publication_figures_module.DEFAULT_AWARE_STABILITY_FIGURE7_METRIC_SLUG

    pd.DataFrame(
        [
            {"topology": "hexagonal", "pearson_r": 0.082, "pearson_r_p_value": 0.781, "n_datasets": 14},
            {"topology": "mst", "pearson_r": 0.198, "pearson_r_p_value": 0.497, "n_datasets": 14},
            {"topology": "rng", "pearson_r": -0.192, "pearson_r_p_value": 0.511, "n_datasets": 14},
        ]
    ).to_csv(
        publication_dir / f"stats_stability_selected_params_vs_sample_size_full_{metric_slug}.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {"topology": "hexagonal", "pearson_r": -0.820, "pearson_r_p_value": 0.000326, "n_datasets": 14},
            {"topology": "mst", "pearson_r": -0.839, "pearson_r_p_value": 0.000176, "n_datasets": 14},
            {"topology": "rng", "pearson_r": -0.512, "pearson_r_p_value": 0.0613, "n_datasets": 14},
        ]
    ).to_csv(
        publication_dir / f"stats_stability_selected_params_vs_sample_size_random_{metric_slug}.csv",
        index=False,
    )

    summary = publication_figures_module._sync_default_aware_stability_regression_stats_to_manuscript(
        {
            "enabled": True,
            "output_dir": str(publication_dir.parent),
        }
    )

    updated_text = manuscript_path.read_text(encoding="utf-8")
    assert summary["updated"] is True
    assert "hexagonal (Pearson R=0.082, p=0.781, n=14); MST (Pearson R=0.198, p=0.497, n=14); and RNG (Pearson R=-0.192, p=0.511, n=14)." in updated_text
    assert "hexagonal (Pearson R=-0.820, p=0.000326, n=14); MST (Pearson R=-0.839, p=0.000176, n=14); and RNG (Pearson R=-0.512, p=0.0613, n=14)." in updated_text
    assert "`R=" not in updated_text
    assert "`p=" not in updated_text
    assert "`n=" not in updated_text


def test_render_default_aware_stability_figure8_uses_full_random_and_size_panels(
    tmp_path,
    monkeypatch,
    publication_figures_module,
):
    metric_slug = "balanced_qe_raw"
    stability_dir = tmp_path / "hyperparameter_stability"
    selected_parameters = [
        "param_initial_radius",
        "param_initialization_method",
        "param_radius_decay_type",
        "param_use_momentum",
    ]

    for sampling_mode, offset in [("full", 0.0), ("random", 0.1)]:
        tables_dir = stability_dir / metric_slug / f"sampling_{sampling_mode}" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        global_rows = []
        dataset_rows = []
        for idx, param in enumerate(selected_parameters):
            global_rows.append(
                {
                    "parameter": param,
                    "stability_score_mean_left": 0.30 + offset + idx * 0.01,
                    "stability_score_mean_right": 0.20 + offset + idx * 0.01,
                }
            )
            for dataset_idx, dataset in enumerate(["iris", "wine"]):
                dataset_rows.append(
                    {
                        "dataset": dataset,
                        "parameter": param,
                        "stability_score_mean_left": 0.30 + offset + idx * 0.01 + dataset_idx * 0.02,
                        "stability_score_mean_right": 0.20 + offset + idx * 0.01 + dataset_idx * 0.02,
                    }
                )
        pd.DataFrame(global_rows).to_csv(
            tables_dir / publication_figures_module.stability_table_filename("global", "hexagonal", "mst", metric_slug),
            index=False,
        )
        pd.DataFrame(global_rows).to_csv(
            tables_dir / publication_figures_module.stability_table_filename("global", "hexagonal", "rng", metric_slug),
            index=False,
        )
        pd.DataFrame(dataset_rows).to_csv(
            tables_dir / publication_figures_module.stability_table_filename("dataset", "hexagonal", "mst", metric_slug),
            index=False,
        )
        pd.DataFrame(dataset_rows).to_csv(
            tables_dir / publication_figures_module.stability_table_filename("dataset", "hexagonal", "rng", metric_slug),
            index=False,
        )

    captured: dict[str, object] = {"figure8_panel_rows": None}

    def _fake_plot(*, sampling_tables, output_path, title, subtitle, dpi, show_titles=False, show_mode_titles=True):
        del sampling_tables, title, subtitle, dpi, show_titles, show_mode_titles
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<svg/>", encoding="utf-8")

    def _fake_size_regression(
        *,
        dataset_summary_df,
        sampling_mode,
        output_figure_path,
        output_table_path,
        output_stats_path,
        dpi,
        show_mode_title=True,
    ):
        del dataset_summary_df, sampling_mode, dpi, show_mode_title
        output_table_path.parent.mkdir(parents=True, exist_ok=True)
        output_table_path.write_text("dataset\niris\n", encoding="utf-8")
        output_stats_path.write_text("topology\nhexagonal\n", encoding="utf-8")
        output_figure_path.write_text("<svg/>", encoding="utf-8")
        return {"generated": True}

    def _fake_legend(*, output_path, topology_values, dpi):
        del output_path, topology_values, dpi
        return None

    def _fake_compose(*, panel_rows, title, output_path, dpi, legend_path=None, **kwargs):
        del title, dpi, legend_path, kwargs
        if str(output_path.name) == publication_figures_module.DEFAULT_AWARE_STABILITY_FIGURE8_ASSET_FILENAME:
            captured["figure8_panel_rows"] = panel_rows
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<svg/>", encoding="utf-8")
        return True

    monkeypatch.setattr(
        publication_figures_module,
        "_plot_selected_parameter_stability_stratified_figure",
        _fake_plot,
    )
    monkeypatch.setattr(
        publication_figures_module,
        "_render_selected_parameter_stability_vs_sample_size_regression",
        _fake_size_regression,
    )
    monkeypatch.setattr(
        publication_figures_module,
        "_render_selected_parameter_stability_legend",
        _fake_legend,
    )
    monkeypatch.setattr(
        publication_figures_module,
        "_compose_panel_matrix_from_images",
        _fake_compose,
    )

    generated_files: list[str] = []
    summary = publication_figures_module._render_default_aware_stability_publication_figures(
        stability_dir=stability_dir,
        metric_specs=[("balanced_qe_raw", "Balanced QE", metric_slug)],
        selected_parameters=selected_parameters,
        output_dir=tmp_path / "publication_figures",
        generated_files=generated_files,
        dpi=72,
    )

    assert summary["figure_8"]
    assert summary["sampling_mode_size_figures"][metric_slug]["full"].endswith(
        f"fig_stability_selected_params_vs_sample_size_full_{metric_slug}.svg"
    )
    assert summary["sampling_mode_size_figures"][metric_slug]["random"].endswith(
        f"fig_stability_selected_params_vs_sample_size_random_{metric_slug}.svg"
    )
    panel_rows = captured["figure8_panel_rows"]
    assert isinstance(panel_rows, list)
    assert len(panel_rows) == 2
    assert len(panel_rows[0]) == 2
    assert len(panel_rows[1]) == 2
    assert panel_rows[0][0][0] == "A"
    assert panel_rows[0][1][0] == "B"
    assert panel_rows[1][0][0] == "C"
    assert panel_rows[1][1][0] == "D"
    assert "full sampling" in str(panel_rows[0][0][1]).lower()
    assert "random sampling" in str(panel_rows[0][1][1]).lower()
    assert "_full_" in str(panel_rows[0][0][2])
    assert "_random_" in str(panel_rows[0][1][2])
    assert "dataset size" in str(panel_rows[1][0][1]).lower()
    assert "dataset size" in str(panel_rows[1][1][1]).lower()
    assert "vs_sample_size_full" in str(panel_rows[1][0][2])
    assert "vs_sample_size_random" in str(panel_rows[1][1][2])


def test_filter_full_batch_rows_for_stability_excludes_minibatch(publication_figures_module):
    source = pd.DataFrame(
        [
            {"pair_batch_mode": "full_batch", "value": 1},
            {"pair_batch_mode": "minibatch", "value": 2},
            {"pair_batch_mode": "all", "value": 3},
        ]
    )
    filtered, removed = publication_figures_module._filter_full_batch_rows_for_stability(source)

    assert removed == 1
    assert filtered["value"].tolist() == [1, 3]
    assert "minibatch" not in set(filtered["pair_batch_mode"].astype(str).str.lower())


def test_validate_expected_pairs_flags_dataset_mismatch(publication_figures_module):
    summary = pd.DataFrame(
        {
            "dataset": ["d1", "d2", "GLOBAL_OVERALL"],
            "n_pairs": [3, 2, 5],
        }
    )

    with pytest.raises(ValueError, match="Expected 3 pairs per dataset"):
        publication_figures_module._validate_expected_pairs(
            summary_df=summary,
            dataset_col="dataset",
            expected_pairs=3,
            context="mst-vs-hex",
        )


def test_add_global_row_uses_dataset_macro_average(publication_figures_module):
    summary = pd.DataFrame(
        {
            "dataset": ["d1", "d2", "d3"],
            "n_pairs": [30, 2, 2],
            "median_delta_raw": [-8.0, 1.0, 1.0],
            "median_pct": [100.0, 0.0, 0.0],
            "mean_pct": [95.0, 0.0, 0.0],
            "p_value": [0.01, 0.9, 0.9],
            "effect_size_signed": [1.0, -1.0, -1.0],
        }
    )
    # This strongly skewed pooled input should not affect GLOBAL once macro-averaging
    # is performed from per-dataset summary rows.
    pairs = pd.DataFrame(
        {
            "delta_a_minus_b": [-8.0] * 30 + [1.0, 1.0, 1.0, 1.0],
            "pct_improvement_a_over_b": [100.0] * 30 + [0.0, 0.0, 0.0, 0.0],
        }
    )

    out = publication_figures_module._add_global_row(
        summary_df=summary,
        pairs=pairs,
        dataset_col="dataset",
        bootstrap_iterations=64,
        seed=7,
    )
    global_overall = out[out["dataset"] == "GLOBAL_OVERALL"].iloc[0]
    global_real = out[out["dataset"] == "GLOBAL_REAL"].iloc[0]

    for global_row in (global_overall, global_real):
        assert int(global_row["n_pairs"]) == 3
        assert global_row["median_pct"] == pytest.approx((100.0 + 0.0 + 0.0) / 3.0)
        assert global_row["mean_pct"] == pytest.approx((100.0 + 0.0 + 0.0) / 3.0)


def test_main_requires_at_least_one_enabled_suite(tmp_path, monkeypatch, publication_figures_module):
    data_file = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "dataset": ["d1"],
            "algorithm": ["colors"],
            "architecture": ["hexagonal"],
            "quantization_error_holdout": [1.0],
        }
    ).to_csv(data_file, index=False)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_publication_figures.py",
            "--data-file",
            str(data_file),
            "--output-dir",
            str(output_dir),
            "--no-include-normalized-set",
            "--no-include-raw-set",
        ],
    )

    with pytest.raises(ValueError, match="At least one suite must be enabled"):
        publication_figures_module.main()


def test_method_and_topology_palette_matches_benchmark_scheme(publication_figures_module):
    assert publication_figures_module._method_color_for_algorithm_label("batch") == "#00C853"
    assert publication_figures_module._method_color_for_algorithm_label("colors") == "#FF6F00"
    assert publication_figures_module._method_color_for_algorithm_label("Batch (Mini-batch)") == "#CC79A7"
    assert publication_figures_module._comparison_series_color("architecture", "mst") == "#00E5FF"
    assert publication_figures_module._comparison_series_color("algorithm", "colors") == "#FF6F00"


def test_topology_main_scope_includes_minibatch_and_supplementary_keeps_it(tmp_path, publication_figures_module):
    df = pd.DataFrame(
        [
            {
                "dataset": "d1",
                "architecture": "mst",
                "algorithm": "colors",
                "pair_batch_mode": "all",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 8.0,
            },
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "colors",
                "pair_batch_mode": "all",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 10.0,
            },
            {
                "dataset": "d1",
                "architecture": "mst",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 9.0,
            },
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 11.0,
            },
            {
                "dataset": "d1",
                "architecture": "mst",
                "algorithm": "batch",
                "pair_batch_mode": "minibatch",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 9.5,
            },
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "minibatch",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 11.5,
            },
        ]
    )

    generated_files: list[str] = []
    diagnostics: dict[str, object] = {}

    main_output_dir = tmp_path / "topology_main"
    publication_figures_module._analyze_topology(
        df=df,
        metric="metric_x",
        metric_label="Metric X",
        output_dir=main_output_dir,
        top_k=1,
        bootstrap_iterations=16,
        seed=42,
        dpi=72,
        alpha=0.05,
        strict_pairing=True,
        expected_pairs_per_dataset=None,
        generated_files=generated_files,
        diagnostics=diagnostics,
        diagnostics_prefix="test.main.",
        figure_scope=publication_figures_module.TOPOLOGY_SCOPE_MAIN,
        inline_legends=False,
        legend_registry=None,
    )

    supplementary_output_dir = tmp_path / "topology_supplementary"
    publication_figures_module._analyze_topology(
        df=df,
        metric="metric_x",
        metric_label="Metric X",
        output_dir=supplementary_output_dir,
        top_k=1,
        bootstrap_iterations=16,
        seed=42,
        dpi=72,
        alpha=0.05,
        strict_pairing=True,
        expected_pairs_per_dataset=None,
        generated_files=generated_files,
        diagnostics=diagnostics,
        diagnostics_prefix="test.supplementary.",
        figure_scope=publication_figures_module.TOPOLOGY_SCOPE_SUPPLEMENTARY,
        inline_legends=False,
        legend_registry=None,
    )

    main_strata_path = main_output_dir / "tables" / "mst_vs_hex_strata_algorithm_mode_main_metric_x.csv"
    supplementary_strata_path = (
        supplementary_output_dir / "tables" / "mst_vs_hex_strata_algorithm_mode_supplementary_metric_x.csv"
    )
    assert main_strata_path.exists()
    assert supplementary_strata_path.exists()

    main_strata = pd.read_csv(main_strata_path)
    supplementary_strata = pd.read_csv(supplementary_strata_path)

    assert main_strata["algorithm_batch_mode"].astype(str).str.contains("Mini-batch", regex=False).any()
    assert supplementary_strata["algorithm_batch_mode"].astype(str).str.contains("Mini-batch", regex=False).any()

    main_diag = diagnostics["test.main.topology_main_metric_x"]
    assert main_diag["uses_stratified_sensitivity"] is True
    assert "Colors" in set(main_diag["sensitivity_strata"])
    assert any(str(label).startswith("Batch") for label in main_diag["sensitivity_strata"])


def test_topology_pair_basic_rng_outputs_algorithm_mode_strata(tmp_path, publication_figures_module):
    df = pd.DataFrame(
        [
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "colors",
                "pair_batch_mode": "all",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 10.0,
            },
            {
                "dataset": "d1",
                "architecture": "rng",
                "algorithm": "colors",
                "pair_batch_mode": "all",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 9.0,
            },
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 11.0,
            },
            {
                "dataset": "d1",
                "architecture": "rng",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 12.0,
            },
        ]
    )

    generated_files: list[str] = []
    diagnostics: dict[str, object] = {}
    output_dir = tmp_path / "topology_main"

    publication_figures_module._analyze_topology_pair_basic(
        df=df,
        metric="metric_x",
        metric_label="Metric X",
        output_dir=output_dir,
        top_k=1,
        bootstrap_iterations=16,
        seed=42,
        dpi=72,
        alpha=0.05,
        strict_pairing=True,
        expected_pairs_per_dataset=None,
        generated_files=generated_files,
        diagnostics=diagnostics,
        diagnostics_prefix="test.topology_pair.",
        value_a="hexagonal",
        value_b="rng",
        pair_slug="hex_vs_rng",
        reference_value="hexagonal",
        inline_legends=False,
        legend_registry=None,
    )

    strata_path = output_dir / "tables" / "hex_vs_rng_strata_algorithm_mode_main_metric_x.csv"
    strata_fig_path = output_dir / "figures" / "fig_hex_vs_rng_by_algorithm_mode_main_metric_x.svg"

    assert strata_path.exists()
    assert strata_fig_path.exists()

    strata_df = pd.read_csv(strata_path)
    labels = set(strata_df["algorithm_batch_mode"].astype(str))
    assert any("Colors" in label for label in labels)
    assert any("Batch" in label for label in labels)

    diag_key = "test.topology_pair.topology_pair.hex_vs_rng.metric_x"
    assert diagnostics[diag_key]["uses_stratified_sensitivity"] is True
    assert "Colors" in set(diagnostics[diag_key]["sensitivity_strata"])
    assert "Batch (Full batch)" in set(diagnostics[diag_key]["sensitivity_strata"])


def test_sampling_mode_disaggregated_grid_uses_expected_anchors_and_outputs(tmp_path, publication_figures_module):
    df = pd.DataFrame(
        [
            # Batch / full_batch
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 10.0,
            },
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_sampling": "random",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 9.0,
            },
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_sampling": "hdsssom",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 11.0,
            },
            # Batch / minibatch
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "minibatch",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 12.0,
            },
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "minibatch",
                "pair_sampling": "random",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 10.0,
            },
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "minibatch",
                "pair_sampling": "hdsssom",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 14.0,
            },
            # Colors
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "colors",
                "pair_batch_mode": "all",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 8.0,
            },
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "colors",
                "pair_batch_mode": "all",
                "pair_sampling": "random",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 7.0,
            },
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "colors",
                "pair_batch_mode": "all",
                "pair_sampling": "hdsssom",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 9.0,
            },
            # MST / full_batch
            {
                "dataset": "d1",
                "architecture": "mst",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 11.0,
            },
            {
                "dataset": "d1",
                "architecture": "mst",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_sampling": "random",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 12.0,
            },
            {
                "dataset": "d1",
                "architecture": "mst",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_sampling": "hdsssom",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 13.0,
            },
            # MST / minibatch
            {
                "dataset": "d1",
                "architecture": "mst",
                "algorithm": "batch",
                "pair_batch_mode": "minibatch",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 9.0,
            },
            {
                "dataset": "d1",
                "architecture": "mst",
                "algorithm": "batch",
                "pair_batch_mode": "minibatch",
                "pair_sampling": "random",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 10.0,
            },
            {
                "dataset": "d1",
                "architecture": "mst",
                "algorithm": "batch",
                "pair_batch_mode": "minibatch",
                "pair_sampling": "hdsssom",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 10.5,
            },
            # MST / colors
            {
                "dataset": "d1",
                "architecture": "mst",
                "algorithm": "colors",
                "pair_batch_mode": "all",
                "pair_sampling": "full",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 8.5,
            },
            {
                "dataset": "d1",
                "architecture": "mst",
                "algorithm": "colors",
                "pair_batch_mode": "all",
                "pair_sampling": "random",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 7.5,
            },
            {
                "dataset": "d1",
                "architecture": "mst",
                "algorithm": "colors",
                "pair_batch_mode": "all",
                "pair_sampling": "hdsssom",
                "pair_seed": 7,
                "pair_split": "holdout",
                "metric_x": 9.5,
            },
        ]
    )

    output_dir = tmp_path / "sampling_comparison"
    generated_files: list[str] = []
    diagnostics: dict[str, object] = {}

    publication_figures_module._analyze_sampling_mode_outcomes(
        df=df,
        metric="metric_x",
        metric_label="Metric X",
        output_dir=output_dir,
        top_k=1,
        bootstrap_iterations=16,
        seed=42,
        dpi=72,
        alpha=0.05,
        generated_files=generated_files,
        diagnostics=diagnostics,
        diagnostics_prefix="test.sampling.",
    )

    combined_figure = output_dir / "figures" / "fig_sampling_mode_disaggregated_hex_mst_metric_x.svg"
    assert combined_figure.exists()

    full_batch_pairs = (
        output_dir / "tables" / "sampling_mode_pairs_hexagonal_full_batch_random_vs_hdsssom_metric_x.csv"
    )
    minibatch_pairs = (
        output_dir / "tables" / "sampling_mode_pairs_hexagonal_minibatch_random_vs_hdsssom_metric_x.csv"
    )
    assert full_batch_pairs.exists()
    assert minibatch_pairs.exists()

    full_batch_pairs_df = pd.read_csv(full_batch_pairs)
    minibatch_pairs_df = pd.read_csv(minibatch_pairs)
    assert sorted(full_batch_pairs_df["pair_batch_mode"].astype(str).unique().tolist()) == ["full_batch"]
    assert sorted(minibatch_pairs_df["pair_batch_mode"].astype(str).unique().tolist()) == ["minibatch"]
    assert full_batch_pairs_df["pct_improvement_a_over_b_reference"].to_list() == pytest.approx([(2.0 / 9.0) * 100.0])
    assert minibatch_pairs_df["pct_improvement_a_over_b_reference"].to_list() == pytest.approx([(4.0 / 10.0) * 100.0])

    mst_pairs = output_dir / "tables" / "sampling_mode_pairs_mst_colors_full_vs_random_metric_x.csv"
    assert mst_pairs.exists()


def test_analyze_sampling_mode_outcomes_uses_full_random_only_filenames_without_hdsssom(
    tmp_path,
    publication_figures_module,
):
    df = pd.DataFrame(
        [
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_sampling": "full",
                "pair_seed": 1,
                "pair_split": "holdout",
                "metric_x": 8.0,
            },
            {
                "dataset": "d1",
                "architecture": "hexagonal",
                "algorithm": "batch",
                "pair_batch_mode": "full_batch",
                "pair_sampling": "random",
                "pair_seed": 1,
                "pair_split": "holdout",
                "metric_x": 10.0,
            },
            {
                "dataset": "d1",
                "architecture": "mst",
                "algorithm": "colors",
                "pair_batch_mode": "all",
                "pair_sampling": "full",
                "pair_seed": 2,
                "pair_split": "holdout",
                "metric_x": 4.0,
            },
            {
                "dataset": "d1",
                "architecture": "mst",
                "algorithm": "colors",
                "pair_batch_mode": "all",
                "pair_sampling": "random",
                "pair_seed": 2,
                "pair_split": "holdout",
                "metric_x": 5.0,
            },
        ]
    )

    output_dir = tmp_path / "sampling_comparison"
    generated_files: list[str] = []
    diagnostics: dict[str, object] = {}

    publication_figures_module._analyze_sampling_mode_outcomes(
        df=df,
        metric="metric_x",
        metric_label="Metric X",
        output_dir=output_dir,
        top_k=1,
        bootstrap_iterations=16,
        seed=42,
        dpi=72,
        alpha=0.05,
        generated_files=generated_files,
        diagnostics=diagnostics,
        diagnostics_prefix="test.sampling.",
    )

    assert (
        output_dir / "figures" / "fig_sampling_mode_disaggregated_hex_mst_full_random_only_metric_x.svg"
    ).exists()
    assert (
        output_dir
        / "figures"
        / "fig_sampling_mode_hexagonal_full_batch_full_random_only_full_vs_random_metric_x.svg"
    ).exists()
    assert (
        output_dir
        / "tables"
        / "sampling_mode_pairs_hexagonal_full_batch_full_random_only_full_vs_random_metric_x.csv"
    ).exists()
    assert not (
        output_dir / "figures" / "fig_sampling_mode_hexagonal_full_batch_full_vs_hdsssom_metric_x.svg"
    ).exists()


def test_build_sensitivity_summary_uses_dataset_macro_average(publication_figures_module):
    df = pd.DataFrame(
        [
            {
                "dataset": "d1",
                "pair_sampling": "full",
                "pair_seed": 1,
                "pair_split": "holdout",
                "algorithm": "colors",
                "metric_x": 4.0,
            },
            {
                "dataset": "d1",
                "pair_sampling": "full",
                "pair_seed": 1,
                "pair_split": "holdout",
                "algorithm": "batch",
                "metric_x": 10.0,
            },
            {
                "dataset": "d1",
                "pair_sampling": "random",
                "pair_seed": 2,
                "pair_split": "holdout",
                "algorithm": "colors",
                "metric_x": 4.0,
            },
            {
                "dataset": "d1",
                "pair_sampling": "random",
                "pair_seed": 2,
                "pair_split": "holdout",
                "algorithm": "batch",
                "metric_x": 10.0,
            },
            {
                "dataset": "d1",
                "pair_sampling": "hdsssom",
                "pair_seed": 3,
                "pair_split": "holdout",
                "algorithm": "colors",
                "metric_x": 4.0,
            },
            {
                "dataset": "d1",
                "pair_sampling": "hdsssom",
                "pair_seed": 3,
                "pair_split": "holdout",
                "algorithm": "batch",
                "metric_x": 10.0,
            },
            {
                "dataset": "d2",
                "pair_sampling": "full",
                "pair_seed": 1,
                "pair_split": "holdout",
                "algorithm": "colors",
                "metric_x": 10.0,
            },
            {
                "dataset": "d2",
                "pair_sampling": "full",
                "pair_seed": 1,
                "pair_split": "holdout",
                "algorithm": "batch",
                "metric_x": 10.0,
            },
        ]
    )

    summary = publication_figures_module._build_sensitivity_summary(
        df=df,
        metric="metric_x",
        compare_col="algorithm",
        value_a="colors",
        value_b="batch",
        key_cols=["dataset", "pair_sampling", "pair_seed", "pair_split"],
        top_k_values=[1],
        bootstrap_iterations=16,
        seed=42,
        pct_col="pct_improvement_a_over_b_reference",
        reference_value="batch",
    )

    assert not summary.empty
    row = summary.iloc[0]
    # d1 contributes 60% (three paired units, dataset-level location shift),
    # d2 contributes 0% (one paired unit). Macro average is equal-weighted:
    # (60 + 0) / 2 = 30.
    assert int(row["n_pairs"]) == 2
    assert float(row["median_pct"]) == pytest.approx(30.0)
    assert float(row["mean_pct"]) == pytest.approx(30.0)


def test_wilcoxon_location_ci_degenerate_for_constant_nonzero(publication_figures_module):
    values = np.array([2.5, 2.5, 2.5], dtype=float)
    location, ci_low, ci_high = publication_figures_module._wilcoxon_location_ci(values)

    assert location == pytest.approx(2.5)
    assert ci_low == pytest.approx(2.5)
    assert ci_high == pytest.approx(2.5)


def test_summarize_pairs_aligns_signed_test_and_ci_to_percent_series(monkeypatch, publication_figures_module):
    pairs = pd.DataFrame(
        {
            "dataset": ["d1", "d1", "d2", "d2"],
            "delta_a_minus_b": [-4.0, -2.0, 1.0, 3.0],
            "pct_improvement_a_over_b_reference": [40.0, 20.0, -10.0, -30.0],
        }
    )

    signed_inputs: list[np.ndarray] = []
    ci_inputs: list[np.ndarray] = []
    original_signed_test_stats = publication_figures_module._signed_test_stats
    original_wilcoxon_location_ci = publication_figures_module._wilcoxon_location_ci

    def _capture_signed(values):
        signed_inputs.append(np.asarray(values, dtype=float).copy())
        return original_signed_test_stats(values)

    def _capture_ci(values, alpha=0.05):
        ci_inputs.append(np.asarray(values, dtype=float).copy())
        return original_wilcoxon_location_ci(values, alpha=alpha)

    monkeypatch.setattr(publication_figures_module, "_signed_test_stats", _capture_signed)
    monkeypatch.setattr(publication_figures_module, "_wilcoxon_location_ci", _capture_ci)

    summary = publication_figures_module._summarize_pairs(
        pairs=pairs,
        group_cols=["dataset"],
        bootstrap_iterations=16,
        seed=42,
        pct_col="pct_improvement_a_over_b_reference",
    )

    assert not summary.empty
    assert len(signed_inputs) == 2
    assert len(ci_inputs) == 2

    observed_signed = sorted(tuple(np.sort(x)) for x in signed_inputs)
    observed_ci = sorted(tuple(np.sort(x)) for x in ci_inputs)
    expected_signed = sorted(
        tuple(np.sort(-pairs[pairs["dataset"] == dataset]["pct_improvement_a_over_b_reference"].to_numpy(dtype=float)))
        for dataset in ["d1", "d2"]
    )
    expected_ci = sorted(
        tuple(np.sort(pairs[pairs["dataset"] == dataset]["pct_improvement_a_over_b_reference"].to_numpy(dtype=float)))
        for dataset in ["d1", "d2"]
    )

    assert observed_signed == expected_signed
    assert observed_ci == expected_ci


def test_add_global_row_aligns_signed_test_and_ci_to_dataset_percent_series(monkeypatch, publication_figures_module):
    summary_df = pd.DataFrame(
        {
            "dataset": ["d1", "d2"],
            "median_delta_raw": [-2.0, 1.0],
            "median_pct": [20.0, -10.0],
            "mean_pct": [22.0, -8.0],
            "n_pairs": [3, 3],
            "wins_a": [2, 1],
            "wins_b": [1, 2],
            "ties": [0, 0],
            "win_rate_a": [2.0 / 3.0, 1.0 / 3.0],
            "p_value": [0.2, 0.4],
            "effect_size_signed": [0.33, -0.33],
        }
    )

    signed_inputs: list[np.ndarray] = []
    ci_inputs: list[np.ndarray] = []
    original_signed_test_stats = publication_figures_module._signed_test_stats
    original_wilcoxon_location_ci = publication_figures_module._wilcoxon_location_ci

    def _capture_signed(values):
        signed_inputs.append(np.asarray(values, dtype=float).copy())
        return original_signed_test_stats(values)

    def _capture_ci(values, alpha=0.05):
        ci_inputs.append(np.asarray(values, dtype=float).copy())
        return original_wilcoxon_location_ci(values, alpha=alpha)

    monkeypatch.setattr(publication_figures_module, "_signed_test_stats", _capture_signed)
    monkeypatch.setattr(publication_figures_module, "_wilcoxon_location_ci", _capture_ci)

    out = publication_figures_module._add_global_row(
        summary_df=summary_df,
        pairs=pd.DataFrame(),
        dataset_col="dataset",
        bootstrap_iterations=16,
        seed=42,
        pct_col="pct_improvement_a_over_b_reference",
    )

    assert len(out) == 4
    assert out.iloc[-2]["dataset"] == "GLOBAL_REAL"
    assert out.iloc[-1]["dataset"] == "GLOBAL_OVERALL"
    assert len(signed_inputs) == 2
    assert len(ci_inputs) == 0
    assert signed_inputs[0] == pytest.approx(np.array([-20.0, 10.0], dtype=float))
    assert signed_inputs[1] == pytest.approx(np.array([-20.0, 10.0], dtype=float))


def test_render_composite_legend_key_supports_multiseries_dotted_ci(tmp_path, publication_figures_module):
    legend_path = tmp_path / "legend.svg"
    result = publication_figures_module._render_composite_legend_key(
        output_path=legend_path,
        dpi=120,
        alpha=0.05,
        include_forest=False,
        include_sensitivity=True,
        sensitivity_series_specs=[
            ("Colors vs Batch (Pooled)", "#FF6F00", "o"),
            ("Colors vs Batch (Full Batch)", "#00C853", "s"),
            ("Colors vs Batch (Mini-batch)", "#CC79A7", "^"),
        ],
        sensitivity_dotted_bounds=True,
        compact=True,
    )

    assert result == legend_path
    assert legend_path.exists()
    text = legend_path.read_text(encoding="utf-8")
    assert "Colors vs Batch (Pooled)" in text
    assert "Dotted bounds: 95% CI" in text


def test_plot_sensitivity_curve_multiseries_writes_output(tmp_path, publication_figures_module):
    pooled = pd.DataFrame(
        {
            "top_k": [1, 3, 5],
            "median_pct": [8.0, 6.0, 5.0],
            "ci_low_pct": [4.0, 2.0, 1.0],
            "ci_high_pct": [11.0, 9.0, 8.0],
            "p_value": [0.03, 0.07, 0.12],
        }
    )
    full_batch = pd.DataFrame(
        {
            "top_k": [1, 3, 5],
            "median_pct": [5.0, 4.0, 3.0],
            "ci_low_pct": [2.0, 1.0, 0.0],
            "ci_high_pct": [7.0, 6.0, 5.0],
            "p_value": [0.04, 0.09, 0.2],
        }
    )
    minibatch = pd.DataFrame(
        {
            "top_k": [1, 3, 5],
            "median_pct": [1.0, 0.5, 0.3],
            "ci_low_pct": [-1.5, -2.0, -2.2],
            "ci_high_pct": [3.2, 2.5, 2.0],
            "p_value": [0.4, 0.5, 0.6],
        }
    )

    output_path = tmp_path / "hex_sensitivity.svg"
    publication_figures_module._plot_sensitivity_curve_multiseries(
        sensitivity_series_by_mode={
            "pooled_batch_modes": pooled,
            "full_batch": full_batch,
            "minibatch": minibatch,
        },
        mode_styles=publication_figures_module.HEX_SENSITIVITY_MODE_STYLES,
        primary_mode="full_batch",
        title="Test",
        subtitle="Test subtitle",
        output_path=output_path,
        dpi=120,
        inline_legends=False,
        legend_registry=None,
    )

    assert output_path.exists()


def test_resolve_hex_comparison_scheme_prefers_full_vs_minibatch_without_colors(publication_figures_module):
    scheme = publication_figures_module._resolve_hex_comparison_scheme(
        available_algorithms={"batch"},
        batch_mode_values=["full_batch", "minibatch"],
    )
    assert scheme == "full_vs_minibatch"


def test_infer_hex_comparison_scheme_from_diagnostics(publication_figures_module):
    diagnostics = {
        "normalized.full_plus_random_no_hdsssom.hex_full_batch_balanced_qe_raw": {
            "comparison_scheme": "full_vs_minibatch",
        },
        "normalized.other": {"anything": 1},
    }

    inferred = publication_figures_module._infer_hex_comparison_scheme_from_diagnostics(
        diagnostics=diagnostics,
        diagnostics_prefix="normalized.",
    )
    assert inferred == "full_vs_minibatch"


def test_sync_sampling_comparison_assets_to_paper_uses_reduced_scope_names(
    tmp_path,
    monkeypatch,
    publication_figures_module,
):
    assets_dir = tmp_path / "paper_assets"
    (assets_dir / "figures").mkdir(parents=True, exist_ok=True)
    (assets_dir / "tables").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(publication_figures_module, "_resolve_paper_assets_dir", lambda: assets_dir)

    sampling_comparison_dir = tmp_path / "raw" / "sampling_comparison"
    publication_figures_dir = tmp_path / "raw" / "publication_figures"
    (sampling_comparison_dir / "figures").mkdir(parents=True, exist_ok=True)
    (sampling_comparison_dir / "tables").mkdir(parents=True, exist_ok=True)
    publication_figures_dir.mkdir(parents=True, exist_ok=True)

    (
        sampling_comparison_dir / "figures" / "fig_sampling_mode_disaggregated_hex_mst_full_random_only_balanced_qe_raw.svg"
    ).write_text("<svg/>", encoding="utf-8")
    (
        sampling_comparison_dir
        / "figures"
        / "fig_sampling_mode_disaggregated_no_stack_hex_mst_full_random_only_balanced_qe_raw.svg"
    ).write_text("<svg/>", encoding="utf-8")
    (
        sampling_comparison_dir
        / "tables"
        / "sampling_mode_stats_hexagonal_full_batch_full_random_only_full_vs_random_balanced_qe_raw.csv"
    ).write_text("a,b\n1,2\n", encoding="utf-8")
    (
        publication_figures_dir
        / "figure_3_algorithm_sampling_stratified_full_random_only_all_metrics.svg"
    ).write_text("<svg/>", encoding="utf-8")

    generated_files: list[str] = []
    summary = publication_figures_module._sync_sampling_comparison_assets_to_paper(
        suite_outputs={
            "raw": {
                "sampling_comparison": str(sampling_comparison_dir),
                "publication_figures": str(publication_figures_dir),
                "sampling": {
                    "full_plus_random_no_hdsssom": str(
                        tmp_path / "raw" / "sampling" / "full_plus_random_no_hdsssom"
                    )
                },
            }
        },
        generated_files=generated_files,
    )

    assert summary["copied"] is True
    assert summary["sampling_comparison_scope"] == "full_random_only"
    assert (assets_dir / "figures" / "fig_sampling_hex_batch_balanced_qe_main.svg").exists()
    assert (assets_dir / "figures" / "fig_sampling_hex_batch_balanced_qe_main_full_random_only.svg").exists()
    assert (
        assets_dir / "figures" / "supp_fig_sampling_mode_disaggregated_hex_mst_full_random_only_balanced_qe_raw.svg"
    ).exists()
    assert (
        assets_dir
        / "tables"
        / "supp_full_random_only_sampling_mode_stats_hexagonal_full_batch_full_random_only_full_vs_random_balanced_qe_raw.csv"
    ).exists()


def test_darken_color_reduces_channel_values(publication_figures_module):
    original_rgb = np.asarray(publication_figures_module.mcolors.to_rgb("#00C853"))
    dark_hex = publication_figures_module._darken_color("#00C853", factor=0.5)
    dark_rgb = np.asarray(publication_figures_module.mcolors.to_rgb(dark_hex))

    assert np.all(dark_rgb <= original_rgb + 1e-12)
    assert np.any(dark_rgb < original_rgb - 1e-12)


def test_build_pairs_from_aggregated_reference_uses_hex_denominator(publication_figures_module):
    aggregated = pd.DataFrame(
        {
            "dataset": ["d1", "d1"],
            "pair_seed": ["7", "7"],
            "__pair_compare__": ["hexagonal", "mst"],
            "score": [2.0, 5.0],
            "trials": [3, 3],
        }
    )

    pairs = publication_figures_module._build_pairs_from_aggregated(
        aggregated=aggregated,
        key_cols=["dataset", "pair_seed"],
        value_a="hexagonal",
        value_b="mst",
        reference_value="hexagonal",
    )

    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert float(row["reference_score"]) == pytest.approx(2.0)
    assert float(row["pct_improvement_a_over_b_reference"]) == pytest.approx(150.0)


def test_summarize_pairs_supports_wilcoxon_methods(publication_figures_module):
    pairs = pd.DataFrame(
        {
            "dataset": ["d1", "d1", "d1"],
            "delta_a_minus_b": [-0.4, -0.2, 0.1],
            "pct_improvement_a_over_b_reference": [20.0, 10.0, -5.0],
        }
    )

    summary = publication_figures_module._summarize_pairs(
        pairs=pairs,
        group_cols=["dataset"],
        bootstrap_iterations=0,
        seed=0,
        pct_col="pct_improvement_a_over_b_reference",
        test_method="wilcoxon",
        location_ci_method="wilcoxon",
    )

    assert len(summary) == 1
    row = summary.iloc[0]
    expected_location, expected_low, expected_high = publication_figures_module._wilcoxon_location_ci(
        np.asarray([20.0, 10.0, -5.0], dtype=float)
    )
    expected_p = publication_figures_module._wilcoxon_two_sided_pvalue(
        np.asarray([20.0, 10.0, -5.0], dtype=float)
    )
    assert float(row["median_pct"]) == pytest.approx(expected_location)
    assert float(row["ci_low_pct"]) == pytest.approx(expected_low)
    assert float(row["ci_high_pct"]) == pytest.approx(expected_high)
    assert float(row["p_value"]) == pytest.approx(expected_p)


def test_add_global_row_supports_wilcoxon_methods(publication_figures_module):
    summary = pd.DataFrame(
        {
            "dataset": ["d1", "d2"],
            "n_pairs": [2, 2],
            "wins_a": [2, 1],
            "wins_b": [0, 1],
            "ties": [0, 0],
            "win_rate_a": [1.0, 0.5],
            "median_delta_raw": [-0.15, 0.05],
            "median_pct": [15.0, -5.0],
            "mean_pct": [15.0, -5.0],
            "ci_low_pct": [10.0, -10.0],
            "ci_high_pct": [20.0, 0.0],
            "p_value": [0.1, 1.0],
            "q_value": [0.2, 1.0],
            "effect_size_signed": [1.0, 0.0],
        }
    )
    pairs = pd.DataFrame(
        {
            "dataset": ["d1", "d1", "d2", "d2"],
            "pair_sampling": ["full", "full", "full", "full"],
            "delta_a_minus_b": [-0.2, -0.1, 0.1, 0.0],
            "pct_improvement_a_over_b_reference": [20.0, 10.0, -10.0, 0.0],
        }
    )

    out = publication_figures_module._add_global_row(
        summary_df=summary,
        pairs=pairs,
        dataset_col="dataset",
        bootstrap_iterations=0,
        seed=0,
        pct_col="pct_improvement_a_over_b_reference",
        test_method="wilcoxon",
        location_ci_method="wilcoxon",
    )

    assert len(out) == 3
    global_row = out[out["dataset"] == "GLOBAL_OVERALL"].iloc[0]
    expected_location, expected_low, expected_high = publication_figures_module._wilcoxon_location_ci(
        np.asarray([20.0, 10.0, -10.0, 0.0], dtype=float)
    )
    expected_p = publication_figures_module._wilcoxon_two_sided_pvalue(
        np.asarray([20.0, 10.0, -10.0, 0.0], dtype=float)
    )
    assert float(global_row["median_pct"]) == pytest.approx(expected_location)
    assert float(global_row["ci_low_pct"]) == pytest.approx(expected_low)
    assert float(global_row["ci_high_pct"]) == pytest.approx(expected_high)
    assert float(global_row["p_value"]) == pytest.approx(expected_p)


def test_add_global_row_pools_all_pairs_without_sampling_hardcode(publication_figures_module):
    summary = pd.DataFrame(
        {
            "dataset": ["d1", "d2"],
            "n_pairs": [2, 2],
            "wins_a": [1, 1],
            "wins_b": [1, 1],
            "ties": [0, 0],
            "win_rate_a": [0.5, 0.5],
            "median_delta_raw": [0.0, 0.0],
            "median_pct": [5.0, -5.0],
            "mean_pct": [5.0, -5.0],
            "ci_low_pct": [0.0, -10.0],
            "ci_high_pct": [10.0, 0.0],
            "p_value": [1.0, 1.0],
            "q_value": [1.0, 1.0],
            "effect_size_signed": [0.0, 0.0],
        }
    )
    pairs = pd.DataFrame(
        {
            "dataset": ["d1", "d1", "d2", "d2"],
            "pair_sampling": ["full", "random", "full", "random"],
            "delta_a_minus_b": [-0.2, -0.1, 0.1, -0.05],
            "pct_improvement_a_over_b_reference": [20.0, 10.0, -10.0, 5.0],
        }
    )

    out = publication_figures_module._add_global_row(
        summary_df=summary,
        pairs=pairs,
        dataset_col="dataset",
        bootstrap_iterations=0,
        seed=0,
        pct_col="pct_improvement_a_over_b_reference",
    )

    global_row = out[out["dataset"] == "GLOBAL_OVERALL"].iloc[0]
    expected_location, expected_low, expected_high = publication_figures_module._wilcoxon_location_ci(
        np.asarray([20.0, 10.0, -10.0, 5.0], dtype=float)
    )
    expected_p = publication_figures_module._wilcoxon_two_sided_pvalue(
        np.asarray([20.0, 10.0, -10.0, 5.0], dtype=float)
    )
    assert int(global_row["n_pairs"]) == 4
    assert float(global_row["median_pct"]) == pytest.approx(expected_location)
    assert float(global_row["ci_low_pct"]) == pytest.approx(expected_low)
    assert float(global_row["ci_high_pct"]) == pytest.approx(expected_high)
    assert float(global_row["p_value"]) == pytest.approx(expected_p)
