"""
Regression tests for worker-local Zarr shard staging.
"""

from __future__ import annotations

from collections import namedtuple
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from floatsom.data import shard_staging


def test_stage_zarr_slice_raises_when_disk_insufficient(tmp_path, monkeypatch):
    class _FakeZarrArray:
        shape = (10, 3)
        dtype = np.float32

    fake_zarr = SimpleNamespace(open_array=lambda _path, mode="r", zarr_format=3: _FakeZarrArray())
    monkeypatch.setitem(sys.modules, "zarr", fake_zarr)

    disk_usage = namedtuple("disk_usage", ["total", "used", "free"])
    monkeypatch.setattr(
        shard_staging.shutil,
        "disk_usage",
        lambda _path: disk_usage(total=1, used=1, free=0),
    )

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("Conversion should not run when disk space is insufficient")

    monkeypatch.setattr(shard_staging, "convert_zarr_slice_to_fast_shard_mp", _unexpected)

    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        shard_staging.stage_zarr_slice(
            zarr_path="dummy.zarr",
            local_store_path=str(tmp_path / "local_store"),
            start_idx=0,
            end_idx=1,
            shard_id=0,
            loader_chunk_size=1,
            assigned_cpus=1,
            mem_limit_bytes=None,
            mem_used_bytes=None,
            workers_on_node=1,
            progress_cb=None,
        )

