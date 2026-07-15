"""CPU-only tests for Figure 5 sampling, projection, and SVG assembly."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "run_topology_representative_figure.py"
SPEC = importlib.util.spec_from_file_location("topology_figure", SCRIPT)
assert SPEC and SPEC.loader
figure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figure)


def _panels(weights: np.ndarray):
    return [
        {"topology": topology, "weights_plot_np": weights + column * 0.01, "edges": [(0, 1)]}
        for column, topology in enumerate(figure.TOPOLOGIES)
    ]


def test_iterations_default_and_override_are_honored():
    assert figure.parse_args([]).iterations == 50
    assert figure.parse_args(["--iterations", "17"]).iterations == 17
    parsed = figure.parse_args(["--data-types", "sklearn_circles", "sklearn_covertype", "sklearn_iris"])
    assert parsed.data_types == ["sklearn_circles", "sklearn_covertype", "sklearn_iris"]
    base = Namespace()
    configured = figure._configure_topology_args(base, parsed, "sklearn_iris", "rng")
    assert configured.iterations == 50
    assert configured.data_type == "sklearn_iris"
    assert configured.sampling_method == "full"
    assert configured.initialization_method == "random"
    assert configured.normalization == "xpysom"
    assert configured.initial_radius == 5.0
    assert configured.radius_decay_type == "exponential"
    assert configured.learning_rate == 0.5
    assert configured.use_momentum is False
    assert configured.momentum_init == 0.5


def test_sampling_is_deterministic_and_order_preserving():
    data = np.arange(400, dtype=float).reshape(200, 2)
    first = figure._deterministic_display_sample(data, 30, seed=42)
    second = figure._deterministic_display_sample(data, 30, seed=42)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.diff(first[:, 0]) > 0)
    assert first.shape == (30, 2)


def test_shared_pca_projects_data_and_every_topology_with_one_basis():
    rng = np.random.default_rng(4)
    data = rng.normal(size=(200, 5))
    weights = {topology: rng.normal(size=(8, 5)) for topology in figure.TOPOLOGIES}
    projected_data, projected_weights, mode = figure._project_to_2d(data, weights)
    mean, basis = figure._stable_pca_basis(data)
    assert mode == "pca"
    np.testing.assert_allclose(projected_data, (data - mean) @ basis)
    for topology in figure.TOPOLOGIES:
        np.testing.assert_allclose(projected_weights[topology], (weights[topology] - mean) @ basis)


def test_raster_and_vector_mappers_have_the_same_scaled_coordinates():
    limits = (-3.0, 7.0, -2.0, 5.0)
    vector = figure._build_panel_mapper(limits, 20.0, 30.0, 400.0, 300.0)
    raster = figure._build_panel_mapper(limits, 0.0, 0.0, 800.0, 600.0)
    vx, vy = vector(1.25, 3.5)
    rx, ry = raster(1.25, 3.5)
    np.testing.assert_allclose((rx, ry), ((vx - 20.0) * 2, (vy - 30.0) * 2))


def test_svg_has_two_embedded_jpegs_reused_across_six_panels(tmp_path: Path):
    rng = np.random.default_rng(12)
    circles = rng.normal(size=(120, 2))
    covertype = rng.normal(size=(150, 2))
    weights = rng.normal(size=(100, 2))
    datasets = [
        {"data_type": "sklearn_circles", "data_plot_np": circles, "panels": _panels(weights), "axis_mode": "native"},
        {"data_type": "sklearn_covertype", "data_plot_np": covertype, "panels": _panels(weights), "axis_mode": "pca"},
    ]
    output = tmp_path / "fig_5.svg"
    backgrounds = figure._write_combined_svg(
        output,
        datasets,
        display_limit=30,
        seed=42,
        jpeg_quality=90,
        raster_scale=2,
        background_dir=tmp_path,
    )
    svg = output.read_text(encoding="utf-8")
    assert len(backgrounds) == 2 and all(path.exists() for path in backgrounds)
    assert svg.count("data:image/jpeg;base64,") == 2
    assert svg.count('<image id="cloud-row-') == 2
    assert 'xmlns:xlink="http://www.w3.org/1999/xlink"' in svg
    assert svg.count('<use xlink:href="#cloud-row-') == 6
    assert all(f">{label}</text>" in svg for label in "ABCDEF")
    assert "Circles" not in svg
    assert "Covertype" not in svg
    assert "KDD Cup 99" not in svg
    assert "native 2D display" not in svg
    assert "shared 2D PCA projection" not in svg
    assert "may distort graph geometry" not in svg
    # Only nodes and the single legend marker are vector circles, never observations.
    assert svg.count("<circle") == (6 * 100) + 2
