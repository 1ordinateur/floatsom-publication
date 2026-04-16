from __future__ import annotations

from typing import Any, Dict, Tuple

METRIC_LABELS: Dict[str, str] = {
    "quantization_error_holdout": "QE Holdout",
    "quantization_error_train": "QE Train",
    "distortion_measure_holdout": "Distortion Holdout",
    "balanced_qe_raw": "Balanced QE",
    "quantization_error_holdout_normalized": "QE Holdout (Normalized)",
    "quantization_error_train_normalized": "QE Train (Normalized)",
    "distortion_measure_holdout_normalized": "Distortion Holdout (Normalized)",
    "balanced_qe_normalized": "Balanced QE (Normalized)",
    "Normalized_Overall_Score": "Overall Score (Normalized)",
}

DEFAULT_METRICS: Tuple[str, ...] = (
    "quantization_error_holdout_normalized",
    "quantization_error_train_normalized",
    "distortion_measure_holdout_normalized",
    "balanced_qe_normalized",
    "Normalized_Overall_Score",
)

RAW_METRICS: Tuple[str, ...] = (
    "quantization_error_holdout",
    "quantization_error_train",
    "distortion_measure_holdout",
    "balanced_qe_raw",
)

CANONICAL_DATASET_ORDER: Tuple[str, ...] = (
    "blobs",
    "breast_cancer",
    "circles",
    "digits",
    "iris",
    "moons",
    "olivetti_faces",
    "s_curve",
    "swiss_roll",
    "wine",
)

DEFAULT_SENSITIVITY_TOP_K: Tuple[int, ...] = (1, 3, 5, 10)
PAIR_COMPARE_KEY = "__pair_compare"
PAIR_RANK_KEY = "__pair_rank"

METHOD_BASE_COLORS: Dict[str, str] = {
    "batch": "#00C853",
    "colors": "#FF6F00",
    "minibatch": "#CC79A7",
}

HEX_SENSITIVITY_MODE_STYLES: Dict[str, Dict[str, Any]] = {
    "pooled_batch_modes": {
        "label": "Colors vs Batch (Pooled)",
        "color": METHOD_BASE_COLORS["colors"],
        "marker": "o",
    },
    "full_batch": {
        "label": "Colors vs Batch (Full Batch)",
        "color": METHOD_BASE_COLORS["batch"],
        "marker": "s",
    },
    "minibatch": {
        "label": "Colors vs Batch (Mini-batch)",
        "color": METHOD_BASE_COLORS["minibatch"],
        "marker": "^",
    },
}

TOPOLOGY_BASE_COLORS: Dict[str, str] = {
    "hexagonal": "#FF4FA3",
    "mst": "#00E5FF",
    "rng": "#8A2BE2",
}

DEFAULT_SERIES_COLOR = "#4C78A8"
TOPOLOGY_SENSITIVITY_MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "P", "X")
TOPOLOGY_SCOPE_MAIN = "main"
TOPOLOGY_SCOPE_SUPPLEMENTARY = "supplementary"
SAMPLING_SIGNIFICANT_COLOR = "#d7301f"
SAMPLING_NON_SIGNIFICANT_COLOR = "#7a7a7a"
TEXT_SIZE_SCALE = 1.65
AXIS_LABEL_PADDING = 28.0
PUBLICATION_LEGEND_ROW_HEIGHT = 175
SAMPLING_LEGEND_FONT_SIZE = 20.0
SAMPLING_AXIS_LABEL_PAD = 34.0
SENSITIVITY_Y_AXIS_MIN = -0.1
SHOW_PANEL_PLOT_TITLES = False
MARKER_SIZE_SCALE = 1.35
SCATTER_SIZE_SCALE = 1.45
PUBLICATION_SAFE_LEFT_MARGIN = 0.31
PUBLICATION_SAFE_LEFT_MARGIN_NO_Y_LABELS = 0.03
PUBLICATION_PANEL_FIG_WIDTH = 13.8
PUBLICATION_PANEL_FIG_WIDTH_WITH_Y_LABELS = 15.4
TRIPANEL_FOREST_X_AXIS_LABEL_FONT_SIZE = 17.0
TRIPANEL_FOREST_X_TICK_LABEL_FONT_SIZE = 18.5
TRIPANEL_FOREST_Y_TICK_LABEL_FONT_SIZE = 26.0
TRIPANEL_FOREST_Y_TICK_LABEL_FONT_SIZE_NO_LABELS = 22.0
PUBLICATION_XLIM_RIGHT_PAD_FRACTION = 0.08
DATASET_GROUP_SEPARATOR_COLOR = "#c7c7c7"
DATASET_GROUP_SEPARATOR_LINEWIDTH = 1.15
DATASET_GROUP_SEPARATOR_ALPHA = 0.65
OPTIMAL_PARAMETER_DEFAULT_METRIC = "balanced_qe_raw"
OPTIMAL_PARAMETER_MARKDOWN_FILENAME = "OPTIMAL_PARAMETERS.md"
DEFAULT_AWARE_STABILITY_PUBLICATION_PARAMETERS: Tuple[str, ...] = (
    "param_initial_radius",
    "param_initialization_method",
    "param_radius_decay_type",
    "param_use_momentum",
)
DEFAULT_AWARE_STABILITY_PUBLICATION_SAMPLING_MODES: Tuple[str, ...] = ("full", "random")
DEFAULT_AWARE_STABILITY_PUBLICATION_TOPOLOGIES: Tuple[str, ...] = ("hexagonal", "mst", "rng")
DEFAULT_AWARE_STABILITY_FIGURE7_METRIC_SLUG = "balanced_qe_raw"
DEFAULT_AWARE_STABILITY_FIGURE7_ASSET_FILENAME = "fig_9.svg"
CANONICAL_SAMPLING_MODES: Tuple[str, ...] = ("full", "random", "hdsssom")
DATASET_SAMPLE_SIZE_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "n_train_samples",
    "dataset_n_samples",
    "n_samples",
    "sample_size",
    "dataset_size",
    "config_n_samples",
    "config_sample_size",
)
DATASET_DIMENSION_COUNT_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "n_features",
    "dataset_n_features",
    "feature_count",
    "dimension_count",
    "n_dimensions",
    "dataset_n_dimensions",
    "input_dim",
    "config_n_features",
)
DATASET_SAMPLE_SIZE_FALLBACK: Dict[str, int] = {
    "swiss_roll": 30000,
    "moons": 30000,
    "circles": 30000,
    "blobs": 30000,
    "s_curve": 30000,
    "breast_cancer": 569,
    "wine": 178,
    "iris": 150,
    "digits": 1797,
    "olivetti_faces": 400,
    "diabetes": 442,
    "california_housing": 20640,
    "covertype": 581012,
    "kddcup99": 494021,
    "lfw_people": 13233,
}
DATASET_DIMENSION_COUNT_FALLBACK: Dict[str, int] = {
    "swiss_roll": 3,
    "moons": 2,
    "circles": 2,
    "blobs": 2,
    "s_curve": 3,
    "breast_cancer": 30,
    "wine": 13,
    "iris": 4,
    "digits": 64,
    "olivetti_faces": 4096,
    "diabetes": 10,
    "california_housing": 8,
    "covertype": 54,
    "kddcup99": 41,
    "lfw_people": 2914,
}

__all__ = [name for name in globals() if name.isupper()]
