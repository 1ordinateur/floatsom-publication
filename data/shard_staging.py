"""
Worker-local shard staging helpers.

These utilities are intended to be called from Ray worker actors, but are kept
free of Ray/CuPy imports so they can be unit-tested in minimal environments.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np

from .fast_array_store import FastArrayStore
from .zarr_utils import open_array_read
from .zarr_to_fast_shard_mp import convert_zarr_slice_to_fast_shard_mp


def stage_fast_array_slice(
    *,
    source_path: str,
    local_store_path: str,
    start_idx: int,
    end_idx: int,
    loader_chunk_size: int,
    copy_workers: int,
    executor_cls: Optional[Type] = None,
) -> Dict[str, Any]:
    """
    Stage a slice of an existing FastArrayStore into a worker-local FastArrayStore.
    """
    source_store = FastArrayStore(source_path, mode="r")
    local_store = FastArrayStore(local_store_path, mode="w")

    try:
        shard_shape = source_store.mmap_array[start_idx:end_idx].shape
        shard_samples = int(shard_shape[0]) if shard_shape else 0
        n_features = shard_shape[1] if len(shard_shape) > 1 else 1

        chunk_rows = max(1, min(int(loader_chunk_size), shard_samples or 1))
        if len(shard_shape) > 1:
            chunk_shape = (chunk_rows, n_features)
        else:
            chunk_shape = (chunk_rows,)

        arr = local_store.create(
            shape=shard_shape,
            dtype=np.float32,
            chunks=chunk_shape,
        )

        target_blocks = max(1, int(copy_workers) * 4)
        block_size = max(1, min(int(loader_chunk_size), shard_samples or 1))
        if shard_samples > 0:
            min_block = int(math.ceil(shard_samples / target_blocks))
            if min_block < block_size:
                block_size = max(1, min(int(loader_chunk_size), min_block))

        if copy_workers <= 1 or shard_samples <= block_size:
            offset = 0
            while offset < shard_samples:
                block_end = min(offset + block_size, shard_samples)
                src_start = start_idx + offset
                src_end = start_idx + block_end
                arr[offset:block_end] = source_store.mmap_array[src_start:src_end]
                offset = block_end
        else:
            if executor_cls is None:
                import concurrent.futures

                executor_cls = concurrent.futures.ThreadPoolExecutor

            def _copy_block(bounds: Tuple[int, int]) -> None:
                dst_start, dst_end = bounds
                src_start = start_idx + dst_start
                src_end = start_idx + dst_end
                arr[dst_start:dst_end] = source_store.mmap_array[src_start:src_end]

            ranges = (
                (offset, min(offset + block_size, shard_samples))
                for offset in range(0, shard_samples, block_size)
            )

            with executor_cls(
                max_workers=int(copy_workers),
                thread_name_prefix="fast-stage",
            ) as executor:
                for _ in executor.map(_copy_block, ranges):
                    pass

        arr.flush()
    finally:
        source_store.close()
        local_store.close()

    return {
        "path": local_store_path,
        "shape": shard_shape,
        "n_samples": int(end_idx - start_idx),
        "n_features": int(n_features),
    }


def stage_zarr_slice(
    *,
    zarr_path: str,
    local_store_path: str,
    start_idx: int,
    end_idx: int,
    shard_id: int,
    loader_chunk_size: int,
    assigned_cpus: int,
    mem_limit_bytes: Optional[int],
    mem_used_bytes: Optional[int],
    workers_on_node: int,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> Dict[str, Any]:
    """
    Stage a slice of a Zarr store into a worker-local FastArrayStore shard.
    """
    os.makedirs(local_store_path, exist_ok=True)

    store = None
    arr = None
    data_file = None

    try:
        z = open_array_read(zarr_path).array
        shape = z.shape
        if not shape:
            raise ValueError(f"Zarr array has no shape: {zarr_path}")

        total_samples = int(shape[0])
        n_features = int(shape[1]) if len(shape) > 1 else 1

        if start_idx < 0 or end_idx < start_idx or end_idx > total_samples:
            raise ValueError(
                f"Invalid shard range {start_idx}:{end_idx} for total samples {total_samples}"
            )

        shard_rows = int(end_idx - start_idx)
        shard_shape = (shard_rows, n_features) if len(shape) > 1 else (shard_rows,)

        required_bytes = shard_rows * n_features * 4
        usage = shutil.disk_usage(local_store_path)
        if usage.free < required_bytes + (1 * 1024**3):
            raise RuntimeError(
                f"Insufficient disk space for shard {shard_id} "
                f"({required_bytes / (1024**3):.1f} GB required, {usage.free / (1024**3):.1f} GB free) "
                f"at {local_store_path}"
            )

        store = FastArrayStore(local_store_path, mode="w")
        chunk_rows = max(1, min(int(loader_chunk_size), shard_rows or 1))
        if len(shape) > 1:
            chunk_shape = (chunk_rows, n_features)
        else:
            chunk_shape = (chunk_rows,)
        arr = store.create(shape=shard_shape, dtype=np.float32, chunks=chunk_shape)
        arr.flush()
        data_file = store.data_file
        del arr
        arr = None
        store.close()
        store = None

        if shard_rows == 0:
            return {
                "path": local_store_path,
                "n_samples": shard_rows,
                "n_features": n_features,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "copy_workers_used": 0,
                "block_rows": 0,
                "elapsed_s": 0.0,
                "throughput_gb_s": 0.0,
            }

        bytes_per_row = n_features * 4
        target_bytes = 256 * 1024**2
        block_rows = max(1, min(shard_rows, target_bytes // max(1, bytes_per_row)))

        zarr_chunks = getattr(z, "chunks", None)
        zarr_chunk_rows = 0
        zarr_chunk_cols = 0
        if zarr_chunks:
            try:
                if len(zarr_chunks) > 0:
                    zarr_chunk_rows = int(zarr_chunks[0] or 0)
            except (TypeError, ValueError):
                zarr_chunk_rows = 0
            try:
                if len(zarr_chunks) > 1:
                    zarr_chunk_cols = int(zarr_chunks[1] or 0)
            except (TypeError, ValueError):
                zarr_chunk_cols = 0

        if zarr_chunk_rows > 0 and zarr_chunk_rows <= block_rows:
            aligned = (block_rows // zarr_chunk_rows) * zarr_chunk_rows
            if aligned <= 0:
                aligned = zarr_chunk_rows
            block_rows = max(1, aligned)
        block_rows = min(block_rows, shard_rows)

        if workers_on_node <= 0:
            workers_on_node = 1

        if mem_limit_bytes is not None and mem_used_bytes is not None:
            headroom_bytes = max(0, int(mem_limit_bytes) - int(mem_used_bytes))
        elif mem_limit_bytes is not None:
            reserve_bytes = max(2 * 1024**3, int(mem_limit_bytes) // 10)
            headroom_bytes = max(0, int(mem_limit_bytes) - reserve_bytes)
        else:
            headroom_bytes = None
            try:
                import psutil  # type: ignore

                headroom_bytes = int(psutil.virtual_memory().available)
            except Exception:
                headroom_bytes = None

        per_worker_headroom = None
        if headroom_bytes is not None:
            per_worker_headroom = max(0, int(headroom_bytes) * 9 // 10 // int(workers_on_node))

        zarr_dtype = getattr(z, "dtype", None)
        try:
            src_itemsize = int(np.dtype(zarr_dtype).itemsize)
        except Exception:
            src_itemsize = 4

        chunk_src_bytes = None
        if zarr_chunk_rows > 0:
            chunk_cols = zarr_chunk_cols if zarr_chunk_cols > 0 else n_features
            chunk_src_bytes = int(zarr_chunk_rows) * int(chunk_cols) * int(src_itemsize)

        def _estimate_peak_bytes(rows: int) -> int:
            if rows <= 0:
                return 0
            dst_bytes = int(rows) * int(bytes_per_row)
            sel_src_bytes = int(rows) * int(n_features) * int(src_itemsize)
            decode_bytes = int(chunk_src_bytes) if chunk_src_bytes is not None else int(sel_src_bytes)

            peak = dst_bytes + decode_bytes
            if chunk_src_bytes is not None:
                aligned_to_row_chunks = (start_idx % zarr_chunk_rows == 0) and (rows % zarr_chunk_rows == 0)
                if not aligned_to_row_chunks or rows < zarr_chunk_rows:
                    peak += int(chunk_src_bytes)

            peak = int(peak * 13 // 10) + 512 * 1024**2
            return int(peak)

        peak_bytes = _estimate_peak_bytes(block_rows)
        if per_worker_headroom is not None and per_worker_headroom > 0 and peak_bytes > per_worker_headroom:
            max_iters = 12
            while max_iters > 0 and block_rows > 1 and peak_bytes > per_worker_headroom:
                block_rows = max(1, block_rows // 2)
                peak_bytes = _estimate_peak_bytes(block_rows)
                max_iters -= 1

            if peak_bytes > per_worker_headroom:
                raise RuntimeError(
                    f"Insufficient memory headroom to stage shard {shard_id}: "
                    f"peak_est={peak_bytes / (1024**3):.1f} GB > "
                    f"per_worker_headroom={per_worker_headroom / (1024**3):.1f} GB. "
                    f"(job_limit={mem_limit_bytes} job_used={mem_used_bytes} workers_on_node={workers_on_node} "
                    f"zarr_chunks={zarr_chunks} zarr_dtype={zarr_dtype} block_rows={block_rows})"
                )

        tasks = [
            (offset, min(offset + block_rows, shard_rows))
            for offset in range(0, shard_rows, block_rows)
        ]
        num_blocks = len(tasks)

        cpu_cap = max(1, min(num_blocks, int(assigned_cpus) - 1))
        copy_procs = cpu_cap
        if per_worker_headroom is not None and per_worker_headroom > 0 and peak_bytes > 0:
            mem_cap = max(1, min(num_blocks, int(per_worker_headroom // max(1, peak_bytes))))
            copy_procs = max(1, min(copy_procs, int(mem_cap)))

        start_time = time.perf_counter()
        rows_done, elapsed = convert_zarr_slice_to_fast_shard_mp(
            zarr_path=zarr_path,
            out_data_file=str(data_file),
            shard_shape=shard_shape,
            global_start=start_idx,
            tasks=tasks,
            num_procs=copy_procs,
            progress_cb=progress_cb,
        )

        throughput_gb_s = (
            (rows_done * bytes_per_row / (1024**3)) / elapsed
            if elapsed > 0
            else 0.0
        )

        return {
            "path": local_store_path,
            "n_samples": shard_rows,
            "n_features": n_features,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "copy_workers_used": copy_procs,
            "block_rows": block_rows,
            "elapsed_s": float(elapsed),
            "throughput_gb_s": float(throughput_gb_s),
        }
    except Exception:
        if arr is not None:
            del arr
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        shutil.rmtree(local_store_path, ignore_errors=True)
        raise

