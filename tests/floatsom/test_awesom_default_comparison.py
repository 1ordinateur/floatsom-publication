from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _load_table_module():
    path = ROOT / "benchmarks/optuna/build_awesom_default_vs_floatsom_hex_table.py"
    spec = importlib.util.spec_from_file_location("awesom_default_table_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_comparison_is_paired_by_seed_and_holm_adjusted():
    module = _load_table_module()
    rows = []
    for dataset_index, dataset in enumerate(module.DATASETS):
        for seed in range(10):
            awe = 2.0 + 0.1 * dataset_index + 0.01 * seed
            for implementation, value in (
                ("aweSOM default", awe),
                ("FloatSOM hex untuned", awe * 0.8),
            ):
                rows.append(
                    {
                        "implementation": implementation,
                        "dataset": dataset,
                        "seed": seed,
                        "balanced_qe_raw": value,
                        "quantization_error_holdout": value + 0.1,
                        "quantization_error_train": value - 0.1,
                    }
                )
    table = module.build_table(pd.DataFrame(rows))
    assert len(table) == 15
    assert set(table["n_pairs"]) == {10}
    balanced = table[table["metric"] == "balanced_qe"]
    np.testing.assert_allclose(
        balanced["floatsom_improvement_pct_mean"].to_numpy(dtype=float),
        20.0,
    )
    assert (table["paired_t_p_holm"] >= table["paired_t_p_raw"]).all()
