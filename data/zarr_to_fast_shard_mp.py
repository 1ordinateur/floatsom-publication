"""
Spawn-safe multi-process conversion helpers for Zarr -> FastArrayStore shards.

These utilities are designed to be called from inside Ray GPU actors:
- Uses multiprocessing "spawn" (never fork) to avoid CUDA/fork hazards.
- Keeps worker-process code free of CuPy imports.
- Forces common BLAS/OpenMP/codec thread pools to 1 thread per process to
  prevent oversubscription when using many processes.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Callable, Iterator, List, Optional, Tuple

from .zarr_utils import open_array_read

_WORKER_ZARR = None
_WORKER_OUT_FD = None
_WORKER_GLOBAL_START = 0
_WORKER_ROW_BYTES = 0
_WORKER_ZARR_DTYPE = None

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "TBB_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


def _force_thread_env_to_one() -> None:
    for key in _THREAD_ENV_VARS:
        os.environ[key] = "1"


def _apply_runtime_thread_limits_best_effort() -> None:
    # Best-effort runtime enforcement for already-imported thread pools.
    try:
        import threadpoolctl  # type: ignore

        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass

    # numcodecs/blosc is a common zarr compressor backend.
    try:
        from numcodecs import blosc  # type: ignore

        blosc.set_nthreads(1)
    except Exception:
        pass


def _write_all(fd: int, data: memoryview, offset: int) -> None:
    view = data
    if hasattr(os, "pwrite"):
        write_fn = os.pwrite
        current_offset = int(offset)
        while view:
            written = int(write_fn(fd, view, current_offset))
            if written <= 0:
                raise RuntimeError(f"pwrite failed (written={written}) at offset={current_offset}")
            current_offset += written
            view = view[written:]
        return

    os.lseek(fd, int(offset), os.SEEK_SET)
    while view:
        written = int(os.write(fd, view))
        if written <= 0:
            raise RuntimeError(f"write failed (written={written}) at offset={offset}")
        view = view[written:]


def _as_bytes_view(array: object) -> memoryview:
    mv = memoryview(array)
    try:
        return mv.cast("B")
    except TypeError:
        return mv.cast("B", shape=[mv.nbytes])


def _make_zarr_selection(z: object, src_start: int, src_end: int) -> object:
    selection: object = slice(src_start, src_end)
    try:
        ndim = getattr(z, "ndim", None)
        if ndim is None:
            ndim = len(getattr(z, "shape", ()))
        if ndim and ndim > 1:
            selection = (slice(src_start, src_end), slice(None))
    except Exception:
        selection = slice(src_start, src_end)
    return selection


def _is_c_contiguous(array: object) -> bool:
    flags = getattr(array, "flags", None)
    if flags is None:
        raise TypeError("Array-like object is missing .flags for contiguity check")
    if hasattr(flags, "c_contiguous"):
        return bool(getattr(flags, "c_contiguous"))
    try:
        return bool(flags["C_CONTIGUOUS"])
    except Exception as exc:
        raise TypeError("Unable to resolve C_CONTIGUOUS flag from array.flags") from exc


def _init_worker(
    zarr_path: str,
    out_data_file: str,
    shard_shape: Tuple[int, ...],
    global_start: int,
) -> None:
    _force_thread_env_to_one()
    _apply_runtime_thread_limits_best_effort()

    import numpy as np
    global _WORKER_ZARR, _WORKER_OUT_FD, _WORKER_GLOBAL_START, _WORKER_ROW_BYTES, _WORKER_ZARR_DTYPE
    _WORKER_ZARR = open_array_read(zarr_path).array
    _WORKER_OUT_FD = os.open(out_data_file, os.O_WRONLY)
    _WORKER_GLOBAL_START = int(global_start)
    n_features = int(shard_shape[1]) if len(shard_shape) > 1 else 1
    _WORKER_ROW_BYTES = n_features * 4
    zarr_dtype = getattr(_WORKER_ZARR, "dtype", None)
    if zarr_dtype is None:
        raise TypeError("Zarr array is missing dtype metadata")
    _WORKER_ZARR_DTYPE = np.dtype(zarr_dtype)


def _copy_block(bounds: Tuple[int, int]) -> int:
    dst_start, dst_end = bounds
    src_start = _WORKER_GLOBAL_START + dst_start
    src_end = _WORKER_GLOBAL_START + dst_end

    import numpy as np

    z = _WORKER_ZARR
    out_fd = _WORKER_OUT_FD
    row_bytes = int(_WORKER_ROW_BYTES)
    z_dtype = _WORKER_ZARR_DTYPE

    selection = _make_zarr_selection(z, src_start, src_end)

    block_rows = int(dst_end - dst_start)
    if block_rows <= 0:
        return 0

    dest: object
    if z_dtype == np.float32:
        # Prefer direct reads into a float32 buffer to avoid allocating huge temporaries.
        try:
            if isinstance(selection, tuple):
                dest = np.empty((block_rows, row_bytes // 4), dtype=np.float32)
            else:
                dest = np.empty((block_rows,), dtype=np.float32)
        except Exception:
            dest = None
    else:
        # Avoid allocating a large float32 buffer if the Zarr dtype doesn't match.
        # In that case, direct reads are likely to allocate a separate temporary
        # anyway (or fail), and we'd pay an extra peak allocation per worker proc.
        dest = None

    try:
        if dest is not None:
            if hasattr(z, "get_basic_selection"):
                z.get_basic_selection(selection, out=dest)
            elif hasattr(z, "get_orthogonal_selection"):
                z.get_orthogonal_selection(selection, out=dest)
            elif hasattr(z, "read_direct"):
                try:
                    z.read_direct(dest, source_sel=selection)
                except TypeError:
                    z.read_direct(dest, selection)
            else:
                dest = None
    except Exception:
        dest = None

    if dest is None:
        chunk = z[selection]
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32, copy=False)
        if not _is_c_contiguous(chunk):
            chunk = np.ascontiguousarray(chunk)
        dest = chunk

    data = _as_bytes_view(dest)
    _write_all(int(out_fd), data, int(dst_start) * row_bytes)
    return int(dst_end - dst_start)


def _iter_copy_tasks_single_process(
    *,
    zarr_path: str,
    out_data_file: str,
    shard_shape: Tuple[int, ...],
    global_start: int,
    tasks: List[Tuple[int, int]],
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
    total_rows: int = 0,
    start_time: float = 0.0,
) -> Iterator[int]:
    import numpy as np

    _force_thread_env_to_one()
    _apply_runtime_thread_limits_best_effort()

    z = open_array_read(zarr_path).array
    fd = os.open(out_data_file, os.O_WRONLY)
    n_features = int(shard_shape[1]) if len(shard_shape) > 1 else 1
    row_bytes = n_features * 4
    z_dtype = np.dtype(getattr(z, "dtype", np.float32))

    rows_done = 0
    try:
        for dst_start, dst_end in tasks:
            dst_start_i = int(dst_start)
            dst_end_i = int(dst_end)
            src_start = int(global_start) + dst_start_i
            src_end = int(global_start) + dst_end_i
            selection = _make_zarr_selection(z, src_start, src_end)

            block_rows = int(dst_end_i - dst_start_i)
            if block_rows <= 0:
                continue

            if z_dtype == np.float32:
                if isinstance(selection, tuple):
                    dest = np.empty((block_rows, n_features), dtype=np.float32)
                else:
                    dest = np.empty((block_rows,), dtype=np.float32)

                wrote_direct = False
                try:
                    if hasattr(z, "get_basic_selection"):
                        z.get_basic_selection(selection, out=dest)
                        wrote_direct = True
                    elif hasattr(z, "get_orthogonal_selection"):
                        z.get_orthogonal_selection(selection, out=dest)
                        wrote_direct = True
                    elif hasattr(z, "read_direct"):
                        try:
                            z.read_direct(dest, source_sel=selection)
                        except TypeError:
                            z.read_direct(dest, selection)
                        wrote_direct = True
                except Exception:
                    wrote_direct = False

                if not wrote_direct:
                    chunk = z[selection]
                    if not _is_c_contiguous(chunk):
                        chunk = np.ascontiguousarray(chunk)
                    dest[...] = chunk

                _write_all(fd, _as_bytes_view(dest), dst_start_i * row_bytes)
            else:
                chunk = z[selection]
                if chunk.dtype != np.float32:
                    chunk = chunk.astype(np.float32, copy=False)
                if not _is_c_contiguous(chunk):
                    chunk = np.ascontiguousarray(chunk)
                _write_all(fd, _as_bytes_view(chunk), dst_start_i * row_bytes)

            processed = int(dst_end_i - dst_start_i)
            rows_done += processed
            if progress_cb is not None:
                progress_cb(rows_done, total_rows, time.perf_counter() - float(start_time))
            yield processed
    finally:
        os.close(fd)


def iter_convert_zarr_to_fast_shard_mp(
    *,
    zarr_path: str,
    out_data_file: str,
    shard_shape: Tuple[int, ...],
    global_start: int,
    tasks: List[Tuple[int, int]],
    num_procs: int,
    chunksize: Optional[int] = None,
) -> Iterator[int]:
    """
    Yield processed row counts while copying Zarr slices into an output shard file.
    """
    if num_procs <= 1 or len(tasks) <= 1:
        yield from _iter_copy_tasks_single_process(
            zarr_path=zarr_path,
            out_data_file=out_data_file,
            shard_shape=shard_shape,
            global_start=global_start,
            tasks=tasks,
        )
        return

    ctx = mp.get_context("spawn")
    actual_chunksize = chunksize
    if actual_chunksize is None:
        actual_chunksize = max(1, len(tasks) // max(1, num_procs * 8))

    with ctx.Pool(
        processes=int(num_procs),
        initializer=_init_worker,
        initargs=(zarr_path, out_data_file, shard_shape, int(global_start)),
    ) as pool:
        for processed in pool.imap_unordered(_copy_block, tasks, chunksize=actual_chunksize):
            yield int(processed)


def convert_zarr_slice_to_fast_shard_mp(
    *,
    zarr_path: str,
    out_data_file: str,
    shard_shape: Tuple[int, ...],
    global_start: int,
    tasks: List[Tuple[int, int]],
    num_procs: int,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> Tuple[int, float]:
    """
    Copy a Zarr slice (defined by global_start and tasks) into a preallocated
    FastArrayStore shard file via bounded-memory writes.

    Returns:
        (rows_done, elapsed_seconds)
    """
    total_rows = int(shard_shape[0]) if shard_shape else 0
    rows_done = 0
    start_time = time.perf_counter()

    # Ensure spawned processes inherit 1-thread limits early in interpreter startup.
    previous_env = {key: os.environ.get(key) for key in _THREAD_ENV_VARS}
    for key in _THREAD_ENV_VARS:
        os.environ[key] = "1"

    try:
        if num_procs <= 1 or len(tasks) <= 1:
            for processed in _iter_copy_tasks_single_process(
                zarr_path=zarr_path,
                out_data_file=out_data_file,
                shard_shape=shard_shape,
                global_start=global_start,
                tasks=tasks,
                progress_cb=progress_cb,
                total_rows=total_rows,
                start_time=start_time,
            ):
                rows_done += int(processed)
        else:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(
                processes=int(num_procs),
                initializer=_init_worker,
                initargs=(zarr_path, out_data_file, shard_shape, int(global_start)),
            )
            try:
                async_results = [pool.apply_async(_copy_block, (task,)) for task in tasks]
                pending = list(async_results)

                last_progress = time.perf_counter()
                last_heartbeat = last_progress
                poll_interval_s = 0.5
                heartbeat_s = 30.0
                stall_timeout_s = 600.0

                while pending:
                    progressed = False
                    for result in list(pending):
                        if not result.ready():
                            continue
                        processed = int(result.get())
                        pending.remove(result)
                        rows_done += processed
                        progressed = True
                        now = time.perf_counter()
                        last_progress = now
                        if progress_cb is not None:
                            progress_cb(rows_done, total_rows, now - start_time)

                    now = time.perf_counter()
                    if not progressed:
                        dead_workers = []
                        for proc in getattr(pool, "_pool", []) or []:
                            exitcode = getattr(proc, "exitcode", None)
                            if exitcode not in (None, 0):
                                dead_workers.append((getattr(proc, "pid", None), exitcode))
                        if dead_workers:
                            raise RuntimeError(
                                "Zarr->FastArrayStore conversion worker(s) exited unexpectedly "
                                f"(pid, exitcode)={dead_workers}. This is commonly caused by OOM "
                                "kills or I/O errors when writing the shard output."
                            )

                        if now - last_progress > stall_timeout_s:
                            raise RuntimeError(
                                "Zarr->FastArrayStore conversion stalled: no blocks finished for "
                                f"{stall_timeout_s:.0f}s (rows_done={rows_done}/{total_rows})."
                            )

                        if progress_cb is not None and now - last_heartbeat >= heartbeat_s:
                            last_heartbeat = now
                            progress_cb(rows_done, total_rows, now - start_time)

                        time.sleep(poll_interval_s)

                pool.close()
                pool.join()
            finally:
                try:
                    pool.terminate()
                except Exception:
                    pass
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    elapsed = time.perf_counter() - start_time
    return rows_done, float(elapsed)
