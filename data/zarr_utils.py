"""Standalone hardened helpers for FloatSOM Zarr v3 array I/O."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ZarrArrayHandle:
    store_path: str
    array_key: Optional[str]
    array: Any


def _normalize_shape(shape: Sequence[int]) -> Tuple[int, ...]:
    dims = []
    for axis, raw in enumerate(shape):
        dim = int(raw)
        if dim < 0:
            raise ValueError(f"shape[{axis}] must be >= 0; got {raw!r}.")
        dims.append(dim)
    return tuple(dims)


def _coerce_layout(
    layout: Optional[Sequence[int]],
    *,
    shape: Tuple[int, ...],
    label: str,
) -> Optional[Tuple[int, ...]]:
    if layout is None:
        return None
    if len(layout) != len(shape):
        raise ValueError(
            f"{label} rank mismatch: expected {len(shape)} values for shape {shape}, got {len(layout)}."
        )
    out = []
    for axis, (raw, dim) in enumerate(zip(layout, shape)):
        parsed = int(raw)
        if parsed <= 0:
            raise ValueError(f"{label}[{axis}] must be positive; got {raw!r}.")
        if dim == 0:
            out.append(parsed)
        else:
            out.append(min(parsed, int(dim)))
    return tuple(out)


def default_chunks(shape: Sequence[int]) -> Tuple[int, ...]:
    dims = _normalize_shape(shape)
    if not dims:
        return tuple()
    if len(dims) == 1:
        return (min(1_000_000, max(1, dims[0])),)
    if len(dims) == 2:
        return (min(4096, max(1, dims[0])), min(256, max(1, dims[1])))
    out = []
    for idx, dim in enumerate(dims):
        if idx == 0:
            out.append(min(4096, max(1, dim)))
        elif idx == len(dims) - 1:
            out.append(min(256, max(1, dim)))
        else:
            out.append(min(32, max(1, dim)))
    return tuple(out)


def default_shards(shape: Sequence[int], chunks: Sequence[int]) -> Tuple[int, ...]:
    dims = _normalize_shape(shape)
    chunk_dims = _normalize_shape(chunks)
    if len(dims) != len(chunk_dims):
        raise ValueError("shape/chunks rank mismatch while inferring shards.")
    factors = (16,) + tuple(4 for _ in dims[1:])
    shards = []
    for dim, chunk, factor in zip(dims, chunk_dims, factors):
        target = int(chunk) * int(factor)
        if target <= int(dim):
            shards.append(target)
            continue
        aligned = (int(dim) // int(chunk)) * int(chunk)
        shards.append(max(int(chunk), aligned))
    return tuple(shards)


def resolve_chunks_and_shards(
    *,
    shape: Sequence[int],
    chunks: Optional[Sequence[int]] = None,
    shards: Optional[Sequence[int]] = None,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    dims = _normalize_shape(shape)
    chunk_layout = _coerce_layout(chunks, shape=dims, label="chunks")
    if chunk_layout is None:
        chunk_layout = default_chunks(dims)
    shard_layout = _coerce_layout(shards, shape=dims, label="shards")
    if shard_layout is None:
        shard_layout = default_shards(dims, chunk_layout)

    for axis, (chunk, shard) in enumerate(zip(chunk_layout, shard_layout)):
        if shard < chunk:
            raise ValueError(f"shards[{axis}]={shard} must be >= chunks[{axis}]={chunk}.")
        if shard % chunk != 0:
            raise ValueError(
                f"shards[{axis}]={shard} must be chunk-aligned to chunks[{axis}]={chunk}."
            )
    return chunk_layout, shard_layout


def _open_group(path: str, *, mode: str) -> Any:
    return _zarr_module().open_group(path, mode=mode, zarr_format=3)


def _open_root_array(path: str, *, mode: str) -> Any:
    return _zarr_module().open_array(path, mode=mode, zarr_format=3)


def _zarr_module() -> Any:
    try:
        import zarr  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "zarr is required for FloatSOM Zarr v3 I/O. Install with 'pip install zarr'."
        ) from exc
    return zarr


def _resolve_group_array(group: Any, *, array_key: Optional[str]) -> Tuple[str, Any]:
    if array_key:
        if array_key not in group:
            raise KeyError(f"Array key '{array_key}' not found in Zarr group.")
        return str(array_key), group[str(array_key)]

    keys = sorted(str(name) for name in group.array_keys())
    if not keys:
        raise ValueError("No arrays found in Zarr group.")
    if len(keys) > 1:
        raise ValueError(
            "Multiple arrays found in Zarr group; provide explicit array_key."
        )
    only = keys[0]
    return only, group[only]


def _open_array(path: str, *, mode: str, array_key: Optional[str]) -> ZarrArrayHandle:
    try:
        arr = _open_root_array(path, mode=mode)
        return ZarrArrayHandle(store_path=path, array_key=None, array=arr)
    except Exception:
        group = _open_group(path, mode=mode)
        resolved_key, arr = _resolve_group_array(group, array_key=array_key)
        return ZarrArrayHandle(store_path=path, array_key=resolved_key, array=arr)


def open_array_read(path: os.PathLike[str] | str, *, array_key: Optional[str] = None) -> ZarrArrayHandle:
    return _open_array(os.fspath(path), mode="r", array_key=array_key)


def open_array_update(path: os.PathLike[str] | str, *, array_key: Optional[str] = None) -> ZarrArrayHandle:
    return _open_array(os.fspath(path), mode="r+", array_key=array_key)


def create_sharded_array(
    path: os.PathLike[str] | str,
    *,
    array_key: str,
    shape: Sequence[int],
    dtype: Any,
    chunks: Optional[Sequence[int]] = None,
    shards: Optional[Sequence[int]] = None,
    overwrite: bool = True,
) -> ZarrArrayHandle:
    store_path = os.fspath(path)
    dims = _normalize_shape(shape)
    chunk_layout, shard_layout = resolve_chunks_and_shards(
        shape=dims,
        chunks=chunks,
        shards=shards,
    )

    if overwrite and os.path.exists(store_path):
        shutil.rmtree(store_path, ignore_errors=True)

    group = _open_group(store_path, mode="a")
    arr = group.create_array(
        str(array_key),
        shape=dims,
        dtype=dtype,
        chunks=chunk_layout,
        shards=shard_layout,
        overwrite=True,
    )
    return ZarrArrayHandle(store_path=store_path, array_key=str(array_key), array=arr)
