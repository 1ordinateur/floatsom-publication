"""Tests for Optuna helpers that separate true defaults from tuned overrides."""

from __future__ import annotations

import numpy as np
import pytest

from floatsom.benchmarks.optuna.core.objective import create_floatsom_params
from floatsom.benchmarks.optuna.core.single_benchmark import get_default_trial_params


def test_get_default_trial_params_uses_xpysom_style_defaults_for_forced_sampling_topology():
    defaults = get_default_trial_params(
        {
            "sampling_method": "random",
            "processing_method": "batch",
            "topology_type": "mst",
        }
    )

    assert defaults["initial_radius"] == pytest.approx(5.0)
    assert defaults["radius_decay_type"] == "exponential"
    assert defaults["use_momentum"] is False
    assert defaults["momentum_init"] == pytest.approx(0.5)
    assert defaults["initialization_method"] == "random"


def test_create_floatsom_params_uses_xpysom_style_defaults_when_values_are_omitted():
    data = np.zeros((8, 3), dtype=np.float32)

    params = create_floatsom_params(
        data,
        {
            "sampling_method": "full",
            "processing_method": "batch",
            "topology_type": "hexagonal",
        },
    )

    assert params.initial_radius == pytest.approx(5.0)
    assert params.radius_decay_type == "exponential"
    assert params.initialization_method == "random"
    assert params.processing_config.enable_momentum is False
    assert params.processing_config.initial_momentum == pytest.approx(0.5)
