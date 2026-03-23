from __future__ import annotations

import numpy as np
import pytest

from floatsom.data import zarr_utils


def _require_zarr_v3():
    zarr = pytest.importorskip("zarr")
    major = int(str(getattr(zarr, "__version__", "0")).split(".", maxsplit=1)[0])
    if major < 3:
        pytest.skip("Requires zarr v3 for sharding helpers.")
    return zarr


def test_resolve_chunks_and_shards_defaults_are_chunk_aligned():
    chunks, shards = zarr_utils.resolve_chunks_and_shards(shape=(1000, 37))

    assert len(chunks) == 2
    assert len(shards) == 2
    for chunk, shard in zip(chunks, shards):
        assert chunk > 0
        assert shard >= chunk
        assert shard % chunk == 0


def test_resolve_chunks_and_shards_rejects_shard_smaller_than_chunk():
    with pytest.raises(ValueError, match="must be >="):
        zarr_utils.resolve_chunks_and_shards(
            shape=(128, 8),
            chunks=(32, 8),
            shards=(16, 8),
        )


def test_create_and_open_sharded_array_roundtrip(tmp_path):
    _require_zarr_v3()

    store_path = tmp_path / "assignments.zarr"
    data = np.arange(20, dtype=np.int32)

    handle = zarr_utils.create_sharded_array(
        store_path,
        array_key="assignments",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(10,),
        shards=(20,),
    )
    handle.array[:] = data

    reopened = zarr_utils.open_array_read(store_path)
    assert reopened.array_key == "assignments"
    np.testing.assert_array_equal(np.asarray(reopened.array), data)


def test_open_array_read_requires_key_for_multi_array_group(tmp_path):
    zarr = _require_zarr_v3()

    store_path = tmp_path / "multi_assignments.zarr"
    zarr_utils.create_sharded_array(
        store_path,
        array_key="assignments",
        shape=(4,),
        dtype=np.int32,
        chunks=(4,),
        shards=(4,),
    ).array[:] = np.arange(4, dtype=np.int32)

    group = zarr.open_group(str(store_path), mode="a", zarr_format=3)
    labels = group.create_array(
        "labels",
        shape=(4,),
        dtype=np.int32,
        chunks=(4,),
        shards=(4,),
        overwrite=True,
    )
    labels[:] = np.full((4,), 9, dtype=np.int32)

    with pytest.raises(ValueError, match="Multiple arrays"):
        zarr_utils.open_array_read(store_path)

    selected = zarr_utils.open_array_read(store_path, array_key="labels")
    assert selected.array_key == "labels"
    np.testing.assert_array_equal(np.asarray(selected.array), np.full((4,), 9, dtype=np.int32))
