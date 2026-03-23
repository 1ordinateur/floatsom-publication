"""
Dataset format detection utilities.

This exists to keep file/dir format detection consistent across:
- DataSourceFactory (routing)
- FileDataSource (loading/streaming)
- Ray staging/distribution logic
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

DatasetFormat = Literal[
    "zarr",
    "fast_array",
    "npy",
    "npz",
    "csv",
    "parquet",
    "unknown",
]


def detect_dataset_format(path: Union[str, Path]) -> DatasetFormat:
    p = Path(path)
    suffix = p.suffix.lower()

    # Directory-backed stores (FastArrayStore, Zarr) need content inspection.
    if p.exists() and p.is_dir():
        if (p / "array_metadata.json").exists():
            return "fast_array"

        # Zarr v3 directory stores.
        if suffix == ".zarr":
            return "zarr"
        if (p / "zarr.json").exists():
            return "zarr"

        return "unknown"

    if suffix == ".zarr":
        return "zarr"
    if suffix == ".fast":
        # FastArrayStore is typically a directory; still treat the suffix as a hint.
        return "fast_array"
    if suffix == ".npy":
        return "npy"
    if suffix == ".npz":
        return "npz"
    if suffix in {".csv", ".txt"}:
        return "csv"
    if suffix == ".parquet":
        return "parquet"

    return "unknown"

