"""
Ray color worker for color set processing
Implements color-specific SOM training logic for distributed multi-GPU training
"""

# Configure numexpr before numpy import to avoid thread limit errors
from .numexpr_config import configure_numexpr_threads

configure_numexpr_threads()

import ray
from ray.util import collective
from ray.util.collective.types import ReduceOp
import cupy as cp
import numpy as np
import os
from typing import Dict, Any, Tuple, Optional, List, Generator
from dataclasses import dataclass
import logging
import time
import concurrent.futures
from collections import defaultdict, deque
from threading import Lock

from .ray_pipeline_base_worker import RayPipelineBaseWorker

from ...utils import (
    find_bmus,
    update_weights_vectorized,
    apply_normalization,
    create_strided_indices,
    compute_weight_updates,
    apply_momentum,
)


@dataclass
class _PinnedBatch:
    samples_mem: cp.cuda.PinnedMemory
    samples_view: np.ndarray
    bmus_mem: cp.cuda.PinnedMemory
    bmus_view: np.ndarray
    indices_mem: cp.cuda.PinnedMemory
    indices_view: np.ndarray
    count: int


class _PinnedColorBatch:
    """Pinned-memory accumulator for colour-set samples."""

    def __init__(self, capacity: int, feature_dim: int):
        self.capacity = max(1, int(capacity))
        self.feature_dim = max(1, int(feature_dim))

        sample_elements = self.capacity * self.feature_dim
        self.samples_mem = cp.cuda.alloc_pinned_memory(sample_elements * 4)
        self.samples_view = np.frombuffer(
            self.samples_mem,
            dtype=np.float32,
            count=sample_elements
        ).reshape(self.capacity, self.feature_dim)

        self.bmus_mem = cp.cuda.alloc_pinned_memory(self.capacity * 4)
        self.bmus_view = np.frombuffer(
            self.bmus_mem,
            dtype=np.int32,
            count=self.capacity
        )

        self.indices_mem = cp.cuda.alloc_pinned_memory(self.capacity * 4)
        self.indices_view = np.frombuffer(
            self.indices_mem,
            dtype=np.int32,
            count=self.capacity
        )

        self.size = 0

    def append(
        self,
        samples_np: np.ndarray,
        bmus_np: np.ndarray,
        indices_np: np.ndarray
    ) -> Generator['_PinnedBatch', None, None]:
        if samples_np.ndim == 1:
            samples_np = samples_np.reshape(-1, self.feature_dim)

        total = samples_np.shape[0]
        offset = 0

        while offset < total:
            if self.size == self.capacity:
                flushed = self.flush()
                if flushed is not None:
                    yield flushed

            space = self.capacity - self.size
            take = min(space, total - offset)
            end = offset + take

            self.samples_view[self.size:self.size + take] = samples_np[offset:end]
            self.bmus_view[self.size:self.size + take] = bmus_np[offset:end]
            self.indices_view[self.size:self.size + take] = indices_np[offset:end]

            self.size += take
            offset = end

    def flush(self) -> Optional['_PinnedBatch']:
        if self.size == 0:
            return None

        count = self.size

        samples_mem = cp.cuda.alloc_pinned_memory(count * self.feature_dim * 4)
        samples_view = np.frombuffer(
            samples_mem, dtype=np.float32, count=count * self.feature_dim
        ).reshape(count, self.feature_dim)
        samples_view[:] = self.samples_view[:count]

        bmus_mem = cp.cuda.alloc_pinned_memory(count * 4)
        bmus_view = np.frombuffer(bmus_mem, dtype=np.int32, count=count)
        bmus_view[:] = self.bmus_view[:count]

        indices_mem = cp.cuda.alloc_pinned_memory(count * 4)
        indices_view = np.frombuffer(indices_mem, dtype=np.int32, count=count)
        indices_view[:] = self.indices_view[:count]

        self.size = 0
        batch = _PinnedBatch(
            samples_mem=samples_mem,
            samples_view=samples_view,
            bmus_mem=bmus_mem,
            bmus_view=bmus_view,
            indices_mem=indices_mem,
            indices_view=indices_view,
            count=count,
        )
        return batch


logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class RayColorWorker(RayPipelineBaseWorker):
    """
    Color-specific worker that extends base worker
    Handles complex color set processing logic for distributed training
    """
    
    def __init__(self, worker_id: int, num_gpus: int, worker_config: Dict[str, Any], 
                 gpu_id: Optional[int] = None, chunk_size: Optional[int] = None):
        """
        Initialize color worker with extended state for color processing
        
        Args:
            worker_id: Worker ID for identification
            num_gpus: Total number of GPUs in cluster
            worker_config: Worker configuration dictionary
            gpu_id: Optional specific GPU ID to use (defaults to Ray-assigned GPU 0)
            chunk_size: Chunk size in number of samples
        """
        # Initialize base worker
        super().__init__(worker_id, num_gpus, worker_config, gpu_id, chunk_size)
        self.randomize_chunk_order = True
        
        # Color-specific state
        self.color_sets = None
        self.params = None
        self.bmu_scheduler_enabled = False
        self.bmu_scheduler = None  # BMU scheduler instance
        self.delta_weights = None

        # Local data and BMU storage
        self.local_samples = None
        self.local_bmus = None
        self.total_local_samples = 0
        self.processed_mask = None
        self.local_sample_order = None
        
        # Partition metadata for colour processing
        self.partition_seed = None
        self.color_order_seed = 0
        self.partitioned_chunks: List[np.ndarray] = []
        self.target_color_batch_size = chunk_size or 0
        self.num_chunks = 0
        self.neuron_to_color_cpu = None
        self._pending_color_jobs: List[Dict[str, Any]] = []
        self._selected_chunk_ids: Optional[np.ndarray] = None
        self._selected_chunk_local_positions: Optional[Dict[int, np.ndarray]] = None
        self._iteration_total_samples = 0
        # Dedicated stream for colour batch host->device transfers. Keep separate from the
        # double-buffering transfer stream to avoid cross-stream wait cycles.
        self.color_transfer_stream: Optional[cp.cuda.Stream] = None
        self._timing_lock = Lock()
        self._timing_stats = defaultdict(lambda: {'count': 0, 'total': 0.0, 'max': 0.0})
        self._timing_alert_threshold = 0.25  # seconds for immediate logging
        self._cached_color_stage_timeout_s: Optional[float] = None

        logger.info(f"RayColorWorker {worker_id} initialized on GPU {self.assigned_gpu} with color processing support")
        self.no_local_samples = False

    def _configure_selected_indices(self, selected_indices: Optional[np.ndarray]) -> None:
        """
        Prepare worker-local lookup tables for selector-provided global row indices.

        The resulting mapping enables chunk-local filtering so colour processing touches
        only explicitly selected rows.
        """
        self._selected_chunk_ids = None
        self._selected_chunk_local_positions = None

        if selected_indices is None:
            return

        indices = np.asarray(selected_indices)
        if indices.ndim == 0:
            indices = indices.reshape(1)
        elif indices.ndim != 1:
            raise ValueError("selected_indices must be a 1D array")

        if indices.size == 0:
            self._selected_chunk_ids = np.empty(0, dtype=np.int32)
            self._selected_chunk_local_positions = {}
            return

        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("selected_indices must contain integer values")

        global_start = int(getattr(self, 'data_start_idx', 0) or 0)
        global_end = int(getattr(self, 'data_end_idx', global_start) or global_start)
        if global_end <= global_start:
            global_end = global_start + int(max(0, getattr(self, 'total_local_samples', 0) or 0))

        indices_int = indices.astype(np.int64, copy=False)
        in_span = indices_int[(indices_int >= global_start) & (indices_int < global_end)]
        if in_span.size == 0:
            self._selected_chunk_ids = np.empty(0, dtype=np.int32)
            self._selected_chunk_local_positions = {}
            return

        local_offsets = np.unique((in_span - global_start).astype(np.int64, copy=False))
        loader_chunk = max(
            1,
            int(getattr(self, 'loader_chunk_size', 0) or getattr(self, 'chunk_size', 1) or 1),
        )

        chunk_ids = (local_offsets // loader_chunk).astype(np.int32, copy=False)
        local_in_chunk = (local_offsets % loader_chunk).astype(np.int32, copy=False)
        order = np.lexsort((local_in_chunk, chunk_ids))
        chunk_ids = chunk_ids[order]
        local_in_chunk = local_in_chunk[order]

        chunk_positions: Dict[int, np.ndarray] = {}
        unique_chunks, starts = np.unique(chunk_ids, return_index=True)
        for idx, chunk_id in enumerate(unique_chunks):
            start = int(starts[idx])
            end = int(starts[idx + 1]) if idx + 1 < len(starts) else int(len(chunk_ids))
            positions = np.unique(local_in_chunk[start:end]).astype(np.int32, copy=False)
            if positions.size:
                chunk_positions[int(chunk_id)] = positions

        if not chunk_positions:
            self._selected_chunk_ids = np.empty(0, dtype=np.int32)
            self._selected_chunk_local_positions = {}
            return

        self._selected_chunk_ids = np.array(sorted(chunk_positions.keys()), dtype=np.int32)
        self._selected_chunk_local_positions = chunk_positions

    def configure_chunk_sampling(
        self,
        loader_chunk_size: int,
        sampling_fraction: float,
        sampling_method: str = "full",
    ) -> None:
        """
        Configure chunk sampling once per run.

        For colour random subsampling, we keep async-loader chunks dense and perform
        the proportion cut immediately before BMU compute on GPU.
        """
        super().configure_chunk_sampling(loader_chunk_size, sampling_fraction, sampling_method)

        if float(self.sampling_fraction) < 1.0:
            self._async_loader_sampling_fraction_override = 1.0
            # Sentinel: 0 disables target_rows so async chunks stay unsampled.
            self._async_loader_target_rows_override = 0
        else:
            self._async_loader_sampling_fraction_override = None
            self._async_loader_target_rows_override = None

    def _resolve_iteration_total_samples(
        self,
        params: Any,
        selected_indices: Optional[np.ndarray],
    ) -> int:
        selected_total_param = getattr(params, "selected_total_samples", None)
        if selected_total_param is None:
            selected_total_param = getattr(params, "total_selected_samples", None)
        if selected_total_param is not None:
            try:
                selected_total = int(selected_total_param)
                if selected_total > 0:
                    return selected_total
            except Exception:
                pass

        params_total = getattr(params, "total_samples", None)
        try:
            params_total_int = int(params_total) if params_total is not None else 0
        except Exception:
            params_total_int = 0

        if params_total_int > 0:
            sampling_fraction = float(getattr(self, "sampling_fraction", 1.0) or 1.0)
            if selected_indices is None and 0.0 < sampling_fraction < 1.0:
                expected_total = int(round(params_total_int * sampling_fraction))
                return max(expected_total, 1)
            return params_total_int

        local_total = int(getattr(self, "total_local_samples", 0) or 0)
        gpu_count = max(int(getattr(self, "num_gpus", 1) or 1), 1)
        return max(local_total * gpu_count, 1)

    def _reset_timing_stats(self) -> None:
        with self._timing_lock:
            self._timing_stats = defaultdict(lambda: {'count': 0, 'total': 0.0, 'max': 0.0})

    def _record_timing(self, label: str, duration: float, context: Optional[str] = None) -> None:
        with self._timing_lock:
            stats = self._timing_stats[label]
            stats['count'] += 1
            stats['total'] += duration
            if duration > stats['max']:
                stats['max'] = duration
        if duration >= self._timing_alert_threshold:
            return

    def _log_timing_summary(self) -> None:
        with self._timing_lock:
            if not self._timing_stats:
                return
        
    def get_data_info(self) -> dict:
        """Get information about the worker's data for central coordination."""
        info = {'n_samples': 0, 'num_chunks': 0}
        
        # Get sample count based on loading mode
        if hasattr(self, 'async_loader') and self.async_loader is not None:
            if hasattr(self, 'n_samples') and self.n_samples is not None and self.n_samples > 0:
                info['n_samples'] = int(self.n_samples)
        elif self.data_loader:
            loader_info = self.data_loader.get_info()
            info['n_samples'] = int(loader_info['n_samples'])
        
        # Calculate chunks if we have samples
        if info['n_samples'] > 0 and self.chunk_size:
            info['num_chunks'] = (info['n_samples'] + self.chunk_size - 1) // self.chunk_size
        
        return info
    
    def _initialize_local_metadata(self):
        """Prepare local sample counts and chunk boundaries for colour processing."""
        # Determine data source
        if hasattr(self, 'async_loader') and self.async_loader is not None:
            if hasattr(self, 'n_samples') and self.n_samples is not None and self.n_samples > 0:
                self.total_local_samples = int(self.n_samples)
                self.num_chunks = int(self.async_loader.total_chunks)
            else:
                raise RuntimeError(
                    f"Worker {self.worker_id}: Async loader active but n_samples not initialized"
                )
        elif self.data_loader:
            info = self.data_loader.get_info()
            self.total_local_samples = int(info['n_samples'])
            self.num_chunks = int(info['total_chunks'])
        else:
            raise RuntimeError(
                f"Worker {self.worker_id}: No data loader or async loader initialized"
            )

        if self.total_local_samples == 0 or self.num_chunks == 0:
            self.no_local_samples = True
            logger.info(
                f"Worker {self.worker_id}: no samples assigned; participating in sync-only mode"
            )
            return

        self.no_local_samples = False

    @staticmethod
    def _split_chunk_order(order: np.ndarray, n_partitions: int) -> List[np.ndarray]:
        """Split an order array into partition chunks with stable int32 dtype."""
        return [
            arr.astype(np.int32, copy=False)
            for arr in np.array_split(np.asarray(order, dtype=np.int32), n_partitions)
        ]

    def _maybe_shuffle_chunk_order(self, order: np.ndarray, seed: Optional[int]) -> np.ndarray:
        """Return shuffled order when random sampling is active and a seed is provided."""
        shuffled = np.asarray(order, dtype=np.int32).copy()
        if self.sample_order == "random" and seed is not None and shuffled.size > 1:
            rng = np.random.default_rng(seed)
            rng.shuffle(shuffled)
        return shuffled

    @staticmethod
    def _build_loader_chunk_order(base_order: np.ndarray, num_chunks: int) -> np.ndarray:
        """
        Build sync-loader traversal order.

        The sync loader requires a full permutation of all chunk ids. For selected-index
        processing, keep selected chunks first and append the remaining chunks.
        """
        normalized = np.asarray(base_order, dtype=np.int32).copy()
        if normalized.size == 0 or normalized.size == int(num_chunks):
            return normalized

        all_chunks = np.arange(int(num_chunks), dtype=np.int32)
        selected_mask = np.ones(int(num_chunks), dtype=bool)
        selected_mask[normalized] = False
        return np.concatenate((normalized, all_chunks[selected_mask]))

    def _generate_partitions(self, n_partitions: int, seed: Optional[int]) -> List[np.ndarray]:
        """Create per-partition chunk lists using the loader's ordering."""
        if self.no_local_samples or n_partitions <= 0:
            return [np.array([], dtype=np.int32)] * max(n_partitions, 1)

        selected_chunk_ids = getattr(self, "_selected_chunk_ids", None)
        if selected_chunk_ids is not None:
            order = np.asarray(selected_chunk_ids, dtype=np.int32).copy()
            if order.size == 0:
                return [np.array([], dtype=np.int32)] * max(n_partitions, 1)
            order = self._maybe_shuffle_chunk_order(order, seed)
            return self._split_chunk_order(order, n_partitions)

        if hasattr(self, '_custom_chunk_order') and self._custom_chunk_order is not None:
            order = self._custom_chunk_order.copy()
            return self._split_chunk_order(order, n_partitions)

        if self.data_loader:
            return self.data_loader.partition_chunks(n_partitions, seed)

        order = np.arange(self.num_chunks, dtype=np.int32)
        if seed is not None:
            order = self._maybe_shuffle_chunk_order(order, seed)
        return self._split_chunk_order(order, n_partitions)

    def _build_color_order(self, num_color_sets: int, partition_idx: int) -> np.ndarray:
        """
        Build a deterministic per-partition colour-set processing order.

        The seed is shared across workers via setup plan so collective calls stay aligned.
        """
        if num_color_sets <= 0:
            return np.empty(0, dtype=np.int32)

        order = np.arange(num_color_sets, dtype=np.int32)
        if num_color_sets <= 1:
            return order

        base_seed = int(getattr(self, "color_order_seed", 0) or 0)
        seed_mod = (2**32) - 1
        partition_seed = (base_seed + int(partition_idx)) % seed_mod
        rng = np.random.default_rng(partition_seed)
        rng.shuffle(order)
        return order

    def _build_neuron_to_color_map(self) -> np.ndarray:
        """Map neuron index to colour set id for fast CPU lookup."""
        num_neurons = int(self.gpu_weights.shape[0])
        mapping = np.full((num_neurons,), -1, dtype=np.int32)
        for color_idx, color_set in enumerate(self.color_sets):
            if color_set.size > 0:
                mapping[cp.asnumpy(color_set)] = color_idx
        return mapping

    @staticmethod
    def _estimate_slice_prefetch_depth(
        active_color_counts: List[int],
        *,
        chunks_prequeued: int,
    ) -> int:
        if chunks_prequeued <= 0:
            chunks_prequeued = 1
        if not active_color_counts:
            return chunks_prequeued
        avg_active_colors = sum(active_color_counts) / len(active_color_counts)
        avg_active_colors = max(1.0, float(avg_active_colors))
        return max(
            1,
            int(round(float(chunks_prequeued) * avg_active_colors)),
        )

    def _apply_empty_update(self, feature_dim: int) -> None:
        """Participate in collectives when no samples are available."""
        accumulated_updates = cp.zeros_like(self.gpu_weights, dtype=self.gpu_weights.dtype)
        accumulated_influence = cp.zeros(self.gpu_weights.shape[0], dtype=self.gpu_weights.dtype)
        self._apply_weight_updates(accumulated_updates, accumulated_influence, 0)

    def _get_color_chunk_data(self, chunk_idx: int, include_offsets: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.data_loader:
            chunk_view = self.data_loader.get_chunk_view_for_color(chunk_idx)
            offsets = self.data_loader.get_chunk_selection_offsets(chunk_idx) if include_offsets else None
            return chunk_view, offsets
        if getattr(self, 'async_loader', None) is not None:
            chunk_view = self.async_loader.get_chunk_view_for_color(chunk_idx)
            offsets = self.async_loader.get_chunk_selection_offsets(chunk_idx) if include_offsets else None
            return chunk_view, offsets
        raise RuntimeError("No data source available for colour batch gathering")

    def _gather_color_slice(
        self,
        chunk_id: int,
        local_indices: np.ndarray,
        bmus_np: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gather_start = time.perf_counter()
        needs_offsets = getattr(self, "sampling_fraction", 1.0) < 1.0
        chunk_data, selection_offsets = self._get_color_chunk_data(chunk_id, include_offsets=needs_offsets)
        fetch_elapsed = time.perf_counter() - gather_start
        samples_start = time.perf_counter()
        local_indices = np.asarray(local_indices, dtype=np.int32)
        bmus_np = np.asarray(bmus_np, dtype=np.int32)
        if local_indices.size > 1 and isinstance(chunk_data, np.memmap):
            if not np.all(local_indices[:-1] <= local_indices[1:]):
                order = np.argsort(local_indices, kind='stable')
                local_indices = local_indices[order]
                bmus_np = bmus_np[order]
        samples_np = np.asarray(chunk_data[local_indices], dtype=np.float32)
        samples_elapsed = time.perf_counter() - samples_start
        chunk_offset = int(getattr(self, 'data_start_idx', 0) or 0)
        chunk_global_start = chunk_offset + (chunk_id * self.loader_chunk_size)
        if needs_offsets:
            if selection_offsets is None:
                selection_offsets = np.arange(len(chunk_data), dtype=np.int32)
            selected_offsets = selection_offsets[local_indices]
            global_indices_np = (chunk_global_start + selected_offsets).astype(np.int32, copy=False)
        else:
            global_indices_np = (chunk_global_start + local_indices).astype(np.int32, copy=False)
        total_elapsed = time.perf_counter() - gather_start
        self._record_timing(
            'gather_chunk',
            total_elapsed,
            context=f"chunk={chunk_id} count={len(samples_np)} fetch={fetch_elapsed:.6f}s slice={samples_elapsed:.6f}s"
        )
        return samples_np, bmus_np, global_indices_np

    def _enqueue_color_batch(self, batch: _PinnedBatch) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.cuda.Event, _PinnedBatch]:
        if self.color_transfer_stream is None:
            self.color_transfer_stream = cp.cuda.Stream(non_blocking=True)

        transfer_event = cp.cuda.Event()

        enqueue_start = time.perf_counter()
        with self.color_transfer_stream:
            samples_gpu = cp.asarray(batch.samples_view)
            bmus_gpu = cp.asarray(batch.bmus_view)
            indices_gpu = cp.asarray(batch.indices_view)
            transfer_event.record()
        transfer_elapsed = time.perf_counter() - enqueue_start
        self._record_timing(
            'transfer_batch',
            transfer_elapsed,
            context=f"count={batch.count} dim={batch.samples_view.shape[1]}"
        )

        # Keep the pinned batch alive until the transfer_event has completed to avoid
        # undefined behaviour (pinned memory freed while an async H->D copy is in flight).
        return samples_gpu, bmus_gpu, indices_gpu, transfer_event, batch

    def _maybe_report_progress(self, stage: Optional[str] = None) -> None:
        tracker = getattr(self, "_progress_tracker", None)
        if tracker is None:
            return

        interval_s = float(getattr(self, "_progress_interval_s", 5.0) or 5.0)
        now = time.time()
        last = float(getattr(self, "_progress_last_report_t", 0.0) or 0.0)
        if (now - last) < interval_s:
            return
        self._progress_last_report_t = now

        try:
            ref = tracker.report.remote(self.worker_id, stage)
            fire_and_forget = getattr(ray, "fire_and_forget", None)
            if callable(fire_and_forget):
                fire_and_forget(ref)
        except Exception:
            # Watchdog heartbeats are best-effort; ignore failures.
            return

    def _resolve_color_stage_timeout_s(self) -> float:
        """Resolve per-stage stall timeout for local colour gather/stream waits."""
        if self._cached_color_stage_timeout_s is not None:
            return float(self._cached_color_stage_timeout_s)

        def _coerce_timeout(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        timeout_s = 0.0
        cfg = getattr(self, "ray_config", None)
        if isinstance(cfg, dict):
            timeout_s = _coerce_timeout(cfg.get("color_stage_timeout_s", 0.0) or 0.0)
            if timeout_s <= 0:
                timeout_s = _coerce_timeout(cfg.get("iteration_timeout_s", 0.0) or 0.0)
        elif cfg is not None:
            timeout_s = _coerce_timeout(getattr(cfg, "color_stage_timeout_s", 0.0) or 0.0)
            if timeout_s <= 0:
                timeout_s = _coerce_timeout(getattr(cfg, "iteration_timeout_s", 0.0) or 0.0)

        if timeout_s <= 0:
            timeout_s = 300.0

        self._cached_color_stage_timeout_s = float(timeout_s)
        return float(timeout_s)

    def _wait_cuda_event_with_timeout(
        self,
        event: cp.cuda.Event,
        timeout_s: float,
        *,
        context: str,
    ) -> None:
        def _event_done(evt: cp.cuda.Event) -> bool:
            # CuPy exposes completion via `done`; some backends/wrappers may expose `query()`.
            done_attr = getattr(evt, "done", None)
            if done_attr is not None:
                return bool(done_attr() if callable(done_attr) else done_attr)
            query_fn = getattr(evt, "query", None)
            if callable(query_fn):
                return bool(query_fn())
            raise AttributeError(
                f"CUDA event object of type {type(evt)!r} does not expose `done` or `query()`"
            )

        start = time.monotonic()
        while True:
            if _event_done(event):
                return
            elapsed = time.monotonic() - start
            if elapsed > timeout_s:
                raise TimeoutError(
                    f"Worker {self.worker_id}: Timed out after {timeout_s:.1f}s waiting for CUDA event "
                    f"({context})"
                )
            time.sleep(0.01)

    def _sync_stream_with_timeout(
        self,
        stream: cp.cuda.Stream,
        timeout_s: float,
        *,
        context: str,
    ) -> None:
        sync_event = cp.cuda.Event()
        with stream:
            sync_event.record()
        self._wait_cuda_event_with_timeout(sync_event, timeout_s, context=context)

    def _iter_gathered_color_slices(
        self,
        entries: List[Dict[str, Any]],
        *,
        partition_idx: int,
        color_idx: int,
        timeout_s: float,
        prefetch_depth: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Gather colour slices with bounded waits to avoid silent threadpool stalls."""
        if not entries:
            return

        cpu_pool = self.get_cpu_pool(min_workers=2)
        max_workers = int(getattr(self, "_cpu_pool_workers", 0) or 2)
        if prefetch_depth is None:
            max_inflight = max(2, min(16, max_workers * 2))
        else:
            max_inflight = int(prefetch_depth)
        if max_inflight <= 0:
            max_inflight = 1
        pending = deque()
        ready: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        warned = set()
        next_submit = 0
        next_consume = 0
        total_entries = len(entries)

        def _submit_entry(entry_idx: int) -> None:
            entry = entries[entry_idx]
            chunk_id = int(entry['chunk'])
            future = cpu_pool.submit(
                self._gather_color_slice,
                chunk_id,
                entry['local_indices'],
                entry['bmus'],
            )
            pending.append((entry_idx, chunk_id, future))

        def _fill_prefetch() -> None:
            nonlocal next_submit
            while next_submit < total_entries and (len(pending) + len(ready)) < max_inflight:
                _submit_entry(next_submit)
                next_submit += 1

        _fill_prefetch()

        try:
            while next_consume < total_entries:
                if next_consume in ready:
                    yield ready.pop(next_consume)
                    next_consume += 1
                    _fill_prefetch()
                    continue

                if next_consume not in warned:
                    entry = entries[next_consume]
                    print(
                        f"[RayColorWorker {self.worker_id}] colour-slice on-demand fallback "
                        f"partition={partition_idx} colour={color_idx} chunk={int(entry['chunk'])} "
                        f"entry={next_consume + 1}/{total_entries}",
                        flush=True,
                    )
                    warned.add(next_consume)

                if not pending:
                    _fill_prefetch()
                    if not pending:
                        break

                futures = [future for _, _, future in pending]
                done, _ = concurrent.futures.wait(
                    futures,
                    timeout=timeout_s,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    entry = entries[next_consume]
                    chunk_id = int(entry['chunk'])
                    raise TimeoutError(
                        f"Worker {self.worker_id}: Timed out after {timeout_s:.1f}s gathering colour slice "
                        f"(partition={partition_idx} colour={color_idx} chunk={chunk_id} "
                        f"entry={next_consume + 1}/{total_entries})"
                    )

                next_pending = deque()
                for entry_idx, chunk_id, future in pending:
                    if not future.done():
                        next_pending.append((entry_idx, chunk_id, future))
                        continue
                    if future.cancelled():
                        raise RuntimeError(
                            f"Worker {self.worker_id}: Prefetch cancelled gathering colour slice "
                            f"(partition={partition_idx} colour={color_idx} chunk={chunk_id} "
                            f"entry={entry_idx + 1}/{total_entries})"
                        )
                    try:
                        result = future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Worker {self.worker_id}: Failed gathering colour slice "
                            f"(partition={partition_idx} colour={color_idx} chunk={chunk_id} "
                            f"entry={entry_idx + 1}/{total_entries}): {exc}"
                        ) from exc
                    ready[int(entry_idx)] = result
                pending = next_pending
        except Exception:
            for _, _, future in pending:
                future.cancel()
            raise

    def _wait_for_color_jobs(self) -> None:
        if not self._pending_color_jobs:
            return

        wait_start = time.perf_counter()
        job_count = len(self._pending_color_jobs)
        timeout_s = self._resolve_color_stage_timeout_s()
        for job in self._pending_color_jobs:
            self._wait_cuda_event_with_timeout(
                job['event'],
                timeout_s,
                context=f"pending_color_job worker={self.worker_id}",
            )
            del job['samples'], job['bmus'], job['indices']
        self._pending_color_jobs.clear()
        wait_elapsed = time.perf_counter() - wait_start
        self._record_timing('wait_jobs', wait_elapsed, context=f"count={job_count}")

    def _stream_color_batches_enabled(self) -> bool:
        """Enable streaming batch updates for dynamic graph topologies."""
        params = getattr(self, "params", None)
        topology_cfg = getattr(params, "topology_config", None)
        topology_type = getattr(topology_cfg, "topology_type", None)
        return (topology_type or "").lower() in {"mst", "rng"}

    def _iter_color_batches(
        self,
        entries: List[Dict[str, Any]],
        target_batch: int,
        feature_dim: int,
        color_idx: int,
        partition_idx: int,
        timeout_s: float,
        prefetch_depth: Optional[int] = None,
    ) -> Generator['_PinnedBatch', None, None]:
        if not entries:
            return

        buffer = _PinnedColorBatch(target_batch, feature_dim)
        for samples_np, bmus_np, global_indices_np in self._iter_gathered_color_slices(
            entries,
            partition_idx=partition_idx,
            color_idx=color_idx,
            timeout_s=timeout_s,
            prefetch_depth=prefetch_depth,
        ):
            append_start = time.perf_counter()
            flushed_batches = list(buffer.append(samples_np, bmus_np, global_indices_np))
            append_elapsed = time.perf_counter() - append_start
            self._record_timing(
                'colour_append',
                append_elapsed,
                context=f"colour={color_idx} samples={len(samples_np)}"
            )
            for flushed in flushed_batches:
                yield flushed

        flushed = buffer.flush()
        if flushed is not None:
            yield flushed

    def _collect_descriptors_for_chunk(
        self,
        chunk_idx: int,
        color_descriptors: List[List[Dict[str, Any]]],
        rng: Optional[np.random.Generator],
        metric: Optional[str] = None,
        metric_kwargs: Optional[Dict[str, Any]] = None,
        prefetch_indices: Optional[List[int]] = None,
    ) -> int:
        """Load chunk, compute BMUs, and populate colour descriptors.

        Returns the number of colour sets that received samples from this chunk.
        """
        if self.neuron_to_color_cpu is None:
            self.neuron_to_color_cpu = self._build_neuron_to_color_map()

        chunk_idx_int = int(chunk_idx)
        selected_positions = None
        selected_map = getattr(self, "_selected_chunk_local_positions", None)
        if selected_map is not None:
            selected_positions = selected_map.get(chunk_idx_int)
            if selected_positions is None or selected_positions.size == 0:
                self._record_timing('selected_chunk_skip', 0.0, context=f"chunk={chunk_idx_int}")
                return 0

        load_start = time.perf_counter()
        if self.multi_buffering_enabled:
            chunk_data_gpu, chunk_info = self.load_chunk_multi_buffered(
                chunk_idx_int,
                prefetch_indices=prefetch_indices,
            )
            buffer_id = chunk_info.get('buffer_id') if chunk_info else None
            if chunk_info.get('padded', False):
                chunk_data_gpu = chunk_data_gpu[:chunk_info['local_size']]
        else:
            chunk_data_gpu = self.load_chunk(chunk_idx_int)
            buffer_id = None
        load_elapsed = time.perf_counter() - load_start
        self._record_timing(
            'load_chunk',
            load_elapsed,
            context=f"chunk={chunk_idx_int} shape={tuple(chunk_data_gpu.shape)}"
        )

        if chunk_data_gpu.size == 0:
            self._record_timing('empty_chunk', 0.0, context=f"chunk={chunk_idx_int}")
            return 0

        if metric is None:
            params_obj = getattr(self, "params", None)
            processing_config = getattr(params_obj, "processing_config", None)
            metric = getattr(processing_config, "distance_metric", "euclidean") if processing_config else "euclidean"
            metric_kwargs = getattr(processing_config, "distance_metric_params", {}) if processing_config else {}
            metric_kwargs = metric_kwargs or {}
        elif metric_kwargs is None:
            metric_kwargs = {}

        active_local_indices = None
        if selected_positions is not None:
            chunk_row_count = int(chunk_data_gpu.shape[0])
            active_local_indices = selected_positions[
                (selected_positions >= 0) & (selected_positions < chunk_row_count)
            ].astype(np.int32, copy=False)
            if active_local_indices.size == 0:
                cp.get_default_memory_pool().free_all_blocks()
                return 0
        else:
            sampling_fraction = float(getattr(self, "sampling_fraction", 1.0) or 1.0)
            if sampling_fraction < 1.0:
                chunk_row_count = int(chunk_data_gpu.shape[0])
                desired_rows = max(1, int(round(chunk_row_count * sampling_fraction)))
                if desired_rows < chunk_row_count:
                    if rng is not None:
                        subset_rng = rng
                    else:
                        seed = (
                            (int(getattr(self, "worker_id", 0)) << 32)
                            ^ (int(getattr(self, "iteration_count", 0) or 0) << 12)
                            ^ int(chunk_idx_int)
                        ) % ((2**32) - 1)
                        subset_rng = np.random.default_rng(seed)
                    sampled = subset_rng.choice(chunk_row_count, size=desired_rows, replace=False)
                    sampled.sort()
                    active_local_indices = sampled.astype(np.int32, copy=False)

        bmu_start = time.perf_counter()
        if self.multi_buffering_enabled:
            if buffer_id is not None:
                self._wait_for_buffer_ready(buffer_id)
            with self.compute_stream:
                bmu_input = chunk_data_gpu
                if active_local_indices is not None:
                    bmu_input = chunk_data_gpu[cp.asarray(active_local_indices, dtype=cp.int32)]
                bmus = find_bmus(
                    bmu_input,
                    self.gpu_weights,
                    verbose=False,
                    chunk_size=self.chunk_size,
                    metric=metric,
                    **metric_kwargs,
                )
                if buffer_id is not None:
                    self._mark_buffer_done(buffer_id)
        else:
            bmu_input = chunk_data_gpu
            if active_local_indices is not None:
                bmu_input = chunk_data_gpu[cp.asarray(active_local_indices, dtype=cp.int32)]
            bmus = find_bmus(
                bmu_input,
                self.gpu_weights,
                verbose=False,
                chunk_size=self.chunk_size,
                metric=metric,
                **metric_kwargs,
            )
        bmu_elapsed = time.perf_counter() - bmu_start
        self._record_timing(
            'find_bmus',
            bmu_elapsed,
            context=f"chunk={chunk_idx_int}"
        )

        map_start = time.perf_counter()
        if self.multi_buffering_enabled and self.compute_stream is not None:
            # Ensure BMU kernels queued on the compute stream have completed before host transfer.
            bmus_np = bmus.get(stream=self.compute_stream).astype(np.int32, copy=False)
        else:
            bmus_np = cp.asnumpy(bmus).astype(np.int32, copy=False)
        color_assign_np = self.neuron_to_color_cpu[bmus_np]
        if active_local_indices is not None:
            local_indices_np = active_local_indices.astype(np.int32, copy=False)
        else:
            local_indices_np = np.arange(len(color_assign_np), dtype=np.int32)
        map_elapsed = time.perf_counter() - map_start
        self._record_timing(
            'map_colours',
            map_elapsed,
            context=f"chunk={chunk_idx_int}"
        )

        order_start = time.perf_counter()
        if rng is not None and self.sample_order == 'random' and local_indices_np.size > 1:
            shuffle_order = np.arange(local_indices_np.size, dtype=np.int32)
            rng.shuffle(shuffle_order)
            local_indices_np = local_indices_np[shuffle_order]
            color_assign_np = color_assign_np[shuffle_order]
            bmus_np = bmus_np[shuffle_order]
        order_elapsed = time.perf_counter() - order_start
        self._record_timing(
            'prepare_order',
            order_elapsed,
            context=f"chunk={chunk_idx_int}"
        )

        valid_mask = color_assign_np >= 0
        if not valid_mask.any():
            del bmus
            cp.get_default_memory_pool().free_all_blocks()
            return 0

        valid_colors = color_assign_np[valid_mask]
        valid_local_indices = local_indices_np[valid_mask]
        valid_bmus = bmus_np[valid_mask]

        num_color_sets = int(len(color_descriptors))
        max_color_id = int(valid_colors.max(initial=-1))
        if max_color_id >= num_color_sets:
            raise RuntimeError(
                f"Worker {self.worker_id}: colour id {max_color_id} out of bounds for {num_color_sets} colour sets"
            )

        color_counts = np.bincount(valid_colors, minlength=num_color_sets)
        active_colors = np.flatnonzero(color_counts).astype(np.int32, copy=False)
        self._record_timing(
            'chunk_colour_sets',
            0.0,
            context=f"chunk={chunk_idx_int} count={len(active_colors)}"
        )

        for color_idx in active_colors:
            color_mask = valid_colors == int(color_idx)
            local_positions = valid_local_indices[color_mask]
            color_bmus = valid_bmus[color_mask]
            desc_start = time.perf_counter()
            color_descriptors[color_idx].append({
                'chunk': chunk_idx_int,
                'local_indices': local_positions.astype(np.int32, copy=False),
                'bmus': color_bmus.astype(np.int32, copy=False)
            })
            desc_elapsed = time.perf_counter() - desc_start
            self._record_timing(
                'descriptor_append',
                desc_elapsed,
                context=f"colour={color_idx} samples={local_positions.size}"
            )

        # Release GPU memory associated with BMUs/chunk
        del bmus
        cp.get_default_memory_pool().free_all_blocks()

        return int(len(active_colors))

    def _process_partition(self, partition_idx: int, chunk_ids: np.ndarray, rng: Optional[np.random.Generator]) -> None:
        """Run BMU sweep and colour updates for a partition."""
        num_color_sets = len(self.color_sets)
        descriptors: List[List[Dict[str, Any]]] = [list() for _ in range(num_color_sets)]
        use_streaming = self._stream_color_batches_enabled()

        params_obj = getattr(self, "params", None)
        processing_config = getattr(params_obj, "processing_config", None)
        metric = getattr(processing_config, "distance_metric", "euclidean") if processing_config else "euclidean"
        metric_kwargs = dict(getattr(processing_config, "distance_metric_params", {}) or {}) if processing_config else {}
        if metric == "cosine":
            metric_kwargs["weight_norms"] = cp.linalg.norm(self.gpu_weights, axis=1)

        partition_rng = rng if rng is not None else None
        slice_prefetch_depth = 1
        if not self.no_local_samples and chunk_ids.size > 0:
            descriptor_start = time.perf_counter()
            prefetch_depth = (
                max(0, int(getattr(self, "multi_buffering_num_buffers", 0)) - 1)
                if self.multi_buffering_enabled
                else 0
            )
            active_color_counts = []
            for idx, chunk_idx in enumerate(chunk_ids):
                prefetch = None
                if prefetch_depth > 0:
                    upcoming = chunk_ids[idx + 1: idx + 1 + prefetch_depth]
                    if upcoming.size:
                        prefetch = [int(entry) for entry in upcoming]
                active_color_count = self._collect_descriptors_for_chunk(
                    chunk_idx,
                    descriptors,
                    partition_rng,
                    metric=metric,
                    metric_kwargs=metric_kwargs,
                    prefetch_indices=prefetch,
                )
                active_color_counts.append(int(active_color_count))
            chunks_prequeued = prefetch_depth if prefetch_depth > 0 else 1
            slice_prefetch_depth = self._estimate_slice_prefetch_depth(
                active_color_counts,
                chunks_prequeued=chunks_prequeued,
            )
            self.get_cpu_pool(min_workers=2)
            cpu_workers = int(getattr(self, "_cpu_pool_workers", 0) or 0)
            if cpu_workers <= 0:
                cpu_workers = 1
            slice_prefetch_depth = max(slice_prefetch_depth, cpu_workers * 2)
            descriptor_elapsed = time.perf_counter() - descriptor_start
            self._record_timing(
                'partition_descriptor_sweep',
                descriptor_elapsed,
                context=f"partition={partition_idx} chunks={len(chunk_ids)}"
            )
        else:
            self._record_timing(
                'partition_descriptor_sweep',
                0.0,
                context=f"partition={partition_idx} chunks={len(chunk_ids)}"
            )

        if not any(descriptors):
            self._record_timing('partition_empty', 0.0, context=f"partition={partition_idx}")

        target_batch = max(1, int(self.target_color_batch_size))
        feature_dim = int(self.gpu_weights.shape[1]) if self.gpu_weights.ndim > 1 else 1
        timeout_s = self._resolve_color_stage_timeout_s()
        compute_dtype = self.gpu_weights.dtype

        color_order = self._build_color_order(num_color_sets, partition_idx)
        for color_idx_raw in color_order:
            color_idx = int(color_idx_raw)
            self._maybe_report_progress(stage=f"partition_{partition_idx}_colour_{color_idx}")
            entries = descriptors[color_idx]
            color_start = time.perf_counter()
            color_batches: List[Optional[Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.cuda.Event, _PinnedBatch]]] = []
            batch_iter = None
            current_batch = None
            accumulated_updates = cp.zeros_like(self.gpu_weights, dtype=compute_dtype)
            accumulated_influence = cp.zeros(self.gpu_weights.shape[0], dtype=compute_dtype)
            local_color_samples = 0

            batch_iter = self._iter_color_batches(
                entries,
                target_batch,
                feature_dim,
                color_idx,
                partition_idx,
                timeout_s,
                prefetch_depth=slice_prefetch_depth,
            )
            if use_streaming:
                current_batch = next(batch_iter, None)
            else:
                color_batches = [
                    self._enqueue_color_batch(batch)
                    for batch in batch_iter
                ]

            # Optional extra sync point (debug/stability; hurts multi-node performance)
            if self.collective_barriers_enabled():
                collective.barrier(group_name=self.collective_group)

            if self.compute_stream is None:
                self.compute_stream = cp.cuda.Stream(non_blocking=True)

            flush_idx = 0
            while True:
                self._maybe_report_progress(stage=f"partition_{partition_idx}_colour_{color_idx}_flush_{flush_idx}")
                if use_streaming:
                    has_batch = 1 if current_batch is not None else 0
                else:
                    has_batch = 1 if flush_idx < len(color_batches) else 0
                has_batch_arr = cp.array([has_batch], dtype=cp.int32)
                collective.allreduce(
                    has_batch_arr,
                    group_name=self.collective_group,
                    op=ReduceOp.SUM
                )
                if int(has_batch_arr[0]) == 0:
                    break

                if use_streaming:
                    if current_batch is not None:
                        samples_gpu, bmus_gpu, indices_gpu, transfer_event, pinned_batch = self._enqueue_color_batch(current_batch)
                        self.compute_stream.wait_event(transfer_event)
                        with self.compute_stream:
                            self._accumulate_weight_updates_local(
                                samples_gpu,
                                bmus_gpu,
                                indices_gpu,
                                int(samples_gpu.shape[0]),
                                accumulated_updates,
                                accumulated_influence,
                            )
                        local_color_samples += int(samples_gpu.shape[0])
                        current_batch = next(batch_iter, None)
                elif flush_idx < len(color_batches):
                    batch_entry = color_batches[flush_idx]
                    if batch_entry is not None:
                        samples_gpu, bmus_gpu, indices_gpu, transfer_event, pinned_batch = batch_entry
                        # Drop the list reference early so the pinned batch can be released once the
                        # transfer has completed.
                        color_batches[flush_idx] = None
                        self.compute_stream.wait_event(transfer_event)
                        with self.compute_stream:
                            self._accumulate_weight_updates_local(
                                samples_gpu,
                                bmus_gpu,
                                indices_gpu,
                                int(samples_gpu.shape[0]),
                                accumulated_updates,
                                accumulated_influence,
                            )
                        local_color_samples += int(samples_gpu.shape[0])

                self._sync_stream_with_timeout(
                    self.compute_stream,
                    timeout_s,
                    context=(
                        f"flush_sync partition={partition_idx} colour={color_idx} "
                        f"flush={flush_idx}"
                    ),
                )
                if use_streaming and has_batch:
                    # Now safe to release the pinned batch + event after the wait has completed.
                    del samples_gpu, bmus_gpu, indices_gpu, transfer_event, pinned_batch
                elif (not use_streaming) and flush_idx < len(color_batches) and batch_entry is not None:
                    del samples_gpu, bmus_gpu, indices_gpu, transfer_event, pinned_batch
                flush_idx += 1

            if not use_streaming:
                # Best-effort cleanup in case any batch entries remain.
                for batch_entry in color_batches:
                    if batch_entry is None:
                        continue
                    samples_gpu, bmus_gpu, indices_gpu, transfer_event, pinned_batch = batch_entry
                    self._wait_cuda_event_with_timeout(
                        transfer_event,
                        timeout_s,
                        context=(
                            f"cleanup_transfer partition={partition_idx} colour={color_idx}"
                        ),
                    )
                    del samples_gpu, bmus_gpu, indices_gpu, transfer_event, pinned_batch
            self._apply_weight_updates(
                accumulated_updates,
                accumulated_influence,
                local_color_samples,
            )
            cp.get_default_memory_pool().free_all_blocks()
            color_elapsed = time.perf_counter() - color_start
            self._record_timing(
                'colour_total',
                color_elapsed,
                context=f"colour={color_idx} entries={len(entries)}"
            )

        self._wait_for_color_jobs()


    
    
    # Removed _recalculate_bmus_for_round - BMUs are now always calculated fresh in get_round_data_from_chunk
    # Removed _finalize_color_set_updates - no longer needed with single chunk per round
    
    def _accumulate_weight_updates_local(
        self,
        color_samples: cp.ndarray,
        color_bmus: cp.ndarray,
        color_indices: cp.ndarray,
        n_color_samples: int,
        accumulated_updates: cp.ndarray,
        accumulated_influence: cp.ndarray,
    ) -> None:
        """Accumulate local update totals for one flush of a color set."""
        from ...processing_params import TrainingStepParams

        if n_color_samples <= 0:
            return

        config = self.params.processing_config
        compute_dtype = self.gpu_weights.dtype
        training_params = TrainingStepParams(
            radius=self.params.current_radius,
            learning_rate=self.params.current_learning_rate,
            momentum=self.params.current_momentum,
            delta_weights=self.delta_weights,
            normalization=config.normalization,
            total_samples=n_color_samples,
            norm_alpha=config.norm_alpha,
            norm_clamp_factor=config.norm_clamp_factor,
            norm_percentile=config.norm_percentile,
            norm_max_update_threshold=config.norm_max_update_threshold,
            training_progress=config.training_progress,
            current_epoch=config.current_epoch,
            total_epochs=config.total_epochs,
            virtual_ratio=getattr(config, 'virtual_ratio', 0.5)
        )
        compute_start = time.perf_counter()
        local_updates, local_influence = compute_weight_updates(
            batch=color_samples,
            bmus=color_bmus,
            weights=self.gpu_weights,
            influence_matrix=self.influence_matrix,
            learning_rate=training_params.learning_rate,
            apply_lr_in_update=(config.normalization != "xpysom"),
            chunk_size=min(len(color_samples), self.chunk_size),
            verbose=False
        )
        compute_elapsed = time.perf_counter() - compute_start
        self._record_timing(
            'compute_updates',
            compute_elapsed,
            context=f"samples={n_color_samples} dim={color_samples.shape[1]}"
        )

        if local_updates.dtype != compute_dtype:
            local_updates = local_updates.astype(compute_dtype, copy=False)
        if local_influence.dtype != compute_dtype:
            local_influence = local_influence.astype(compute_dtype, copy=False)
        accumulated_updates += local_updates
        accumulated_influence += local_influence

        # Mark LOCAL samples as processed (convert global -> local offsets for RAM/async mode).
        if len(color_indices) > 0 and self.processed_mask is not None:
            local_offset = int(getattr(self, 'data_start_idx', 0) or 0)
            local_indices = color_indices
            if local_offset:
                local_indices = (color_indices - local_offset).astype(cp.int32, copy=False)

            self.processed_mask[local_indices] = True

            # Update BMU scheduler with local samples processed
            if self.bmu_scheduler:
                self.bmu_scheduler.mark_samples_processed(local_indices)

    def _apply_weight_updates(
        self,
        accumulated_updates: cp.ndarray,
        accumulated_influence: cp.ndarray,
        n_color_samples: int,
    ) -> None:
        """Allreduce and apply a single aggregated update for a full color set."""
        debug_nonfinite = str(os.environ.get("FLOATSOM_DEBUG_NONFINITE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

        config = self.params.processing_config
        compute_dtype = self.gpu_weights.dtype

        if debug_nonfinite:
            self._print_if_nonfinite(
                "gpu_weights",
                self.gpu_weights,
                stage="colors_pre_update",
                params=self.params,
            )
            self._print_if_nonfinite(
                "influence_matrix",
                self.influence_matrix,
                stage="colors_pre_update",
                params=self.params,
                include_diag=True,
            )
            if n_color_samples > 0:
                self._print_if_nonfinite(
                    "accumulated_updates_local",
                    accumulated_updates,
                    stage="colors_post_compute_weight_updates",
                    params=self.params,
                    extra={"n_color_samples": int(n_color_samples)},
                )
                self._print_if_nonfinite(
                    "accumulated_influence_local",
                    accumulated_influence,
                    stage="colors_post_compute_weight_updates",
                    params=self.params,
                    extra={"n_color_samples": int(n_color_samples)},
                )

        if n_color_samples <= 0:
            self._record_timing('colour_empty', 0.0, context="n_samples=0")

        # Keep NCCL collective signatures identical across ranks (dtype + contiguity).
        if accumulated_updates.dtype != compute_dtype:
            accumulated_updates = accumulated_updates.astype(compute_dtype, copy=False)
        if accumulated_influence.dtype != compute_dtype:
            accumulated_influence = accumulated_influence.astype(compute_dtype, copy=False)
        if not accumulated_updates.flags.c_contiguous:
            accumulated_updates = cp.ascontiguousarray(accumulated_updates)
        if not accumulated_influence.flags.c_contiguous:
            accumulated_influence = cp.ascontiguousarray(accumulated_influence)

        if self.collective_barriers_enabled():
            barrier_start = time.perf_counter()
            collective.barrier(group_name=self.collective_group)
            barrier_elapsed = time.perf_counter() - barrier_start
            self._record_timing('barrier', barrier_elapsed, context=f"samples={n_color_samples}")
        # NCCL AllReduce both raw values (stays on GPU!)
        updates_start = time.perf_counter()
        collective.allreduce(
            accumulated_updates,
            group_name=self.collective_group,
            op=ReduceOp.SUM
        )
        updates_elapsed = time.perf_counter() - updates_start
        self._record_timing(
            'allreduce_updates',
            updates_elapsed,
            context=f"shape={tuple(accumulated_updates.shape)}"
        )
        influence_start = time.perf_counter()
        collective.allreduce(
            accumulated_influence,
            group_name=self.collective_group,
            op=ReduceOp.SUM
        )
        influence_elapsed = time.perf_counter() - influence_start
        self._record_timing(
            'allreduce_influence',
            influence_elapsed,
            context=f"len={accumulated_influence.size}"
        )

        if debug_nonfinite:
            self._print_if_nonfinite(
                "accumulated_updates_allreduced",
                accumulated_updates,
                stage="colors_post_allreduce",
                params=self.params,
                extra={"n_color_samples": int(n_color_samples)},
            )
            self._print_if_nonfinite(
                "accumulated_influence_allreduced",
                accumulated_influence,
                stage="colors_post_allreduce",
                params=self.params,
                extra={"n_color_samples": int(n_color_samples)},
            )

        # Get the actual total samples across all GPUs by allreducing the count
        # This ensures all GPUs use the same normalization factor
        samples_start = time.perf_counter()
        sample_count = cp.array([n_color_samples], dtype=cp.int32)
        collective.allreduce(
            sample_count,
            group_name=self.collective_group,
            op=ReduceOp.SUM
        )
        samples_elapsed = time.perf_counter() - samples_start
        self._record_timing(
            'allreduce_samples',
            samples_elapsed,
            context=f"local={n_color_samples}"
        )
        # Ensure never 0 to avoid division issues in normalization.
        # Keep this as a host scalar: apply_normalization() assumes Python scalars
        # for computations such as max(..., 1.0). CuPy 0-d arrays can bypass that
        # guard, leading to 0/0 -> NaN (seen most often in colors mode).
        raw_total_samples_all_gpus = int(sample_count[0].get())
        total_samples_all_gpus = int(max(raw_total_samples_all_gpus, 1))
        
        # Apply normalization and momentum using shared function
        norm_start = time.perf_counter()
        normalized_updates = apply_normalization(
            accumulated_summed_updates=accumulated_updates,
            accumulated_influence_sum=accumulated_influence,
            original_batch_size=total_samples_all_gpus,
            normalization=config.normalization,
            learning_rate=self.params.current_learning_rate,
            current_weights=self.gpu_weights,
            full_bmu_counts=None,
            norm_alpha=config.norm_alpha,
            norm_clamp_factor=config.norm_clamp_factor,
            norm_percentile=config.norm_percentile,
            norm_max_update_threshold=config.norm_max_update_threshold,
            training_progress=config.training_progress,
            current_epoch=config.current_epoch,
            total_epochs=config.total_epochs,
            virtual_ratio=config.virtual_ratio if hasattr(config, 'virtual_ratio') else 0.5,
            verbose=False
        )
        if bool(getattr(config, "scale_color_updates_by_fraction", True)):
            iteration_total_samples = int(
                max(int(getattr(self, "_iteration_total_samples", total_samples_all_gpus) or 0), 1)
            )
            color_set_scale = float(raw_total_samples_all_gpus) / float(iteration_total_samples)
            if color_set_scale != 1.0:
                normalized_updates *= color_set_scale
        norm_elapsed = time.perf_counter() - norm_start
        self._record_timing('normalize', norm_elapsed)

        if debug_nonfinite:
            self._print_if_nonfinite(
                "normalized_updates",
                normalized_updates,
                stage="colors_post_normalize",
                params=self.params,
                extra={"total_samples_all_gpus": int(total_samples_all_gpus)},
            )
        # Apply momentum using shared utility function
        momentum_start = time.perf_counter()
        weight_changes, new_delta = apply_momentum(
            normalized_updates=normalized_updates,
            momentum_coefficient=self.params.current_momentum,
            delta_weights=self.delta_weights
        )
        momentum_elapsed = time.perf_counter() - momentum_start
        self._record_timing('momentum', momentum_elapsed)

        if debug_nonfinite:
            self._print_if_nonfinite(
                "weight_changes",
                weight_changes,
                stage="colors_post_momentum",
                params=self.params,
                extra={"total_samples_all_gpus": int(total_samples_all_gpus)},
            )
        # Update delta_weights for next iteration
        if new_delta is not None:
            self.delta_weights = new_delta

        # Apply weight changes
        apply_start = time.perf_counter()
        self.gpu_weights = self.gpu_weights + weight_changes
        apply_elapsed = time.perf_counter() - apply_start
        self._record_timing('apply_weights', apply_elapsed)

        if debug_nonfinite:
            self._print_if_nonfinite(
                "gpu_weights_after_apply",
                self.gpu_weights,
                stage="colors_post_apply",
                params=self.params,
                extra={"total_samples_all_gpus": int(total_samples_all_gpus)},
            )

    def cleanup(self):
        self._wait_for_color_jobs()
        self._pending_color_jobs.clear()
        super().cleanup()

    def setup_processing(
        self,
        som_weights,
        influence_matrix,
        color_sets,
        collective_group_name,
        params,
        color_round_plan=None,
        selected_indices: Optional[np.ndarray] = None,
        *,
        progress_tracker=None,
        progress_interval_s: float = 5.0,
    ):
        """Configure worker state for a new iteration of colour processing."""
        with self.device:
            # Initialize GPU weights only if provided (first iteration)
            if som_weights is not None or self.gpu_weights is None:
                self.initialize_gpu_weights(som_weights, collective_group_name)

            self.influence_matrix = cp.asarray(influence_matrix)
            self.color_sets = [cp.asarray(cs) for cs in color_sets]
            self.collective_group = collective_group_name
            self.params = params
            self._progress_tracker = progress_tracker
            self._progress_interval_s = float(progress_interval_s) if progress_interval_s else 5.0
            self._progress_last_report_t = 0.0

            plan = color_round_plan or {}
            self.sample_order = params.processing_config.sample_order
            requested_rounds = plan.get('num_partitions', params.processing_config.max_rounds)
            self.max_rounds = max(1, int(requested_rounds))
            self.partition_seed = plan.get('shuffle_seed')
            color_order_seed = plan.get('color_order_seed')
            if color_order_seed is None:
                color_order_seed = self.partition_seed
            if color_order_seed is None:
                color_order_seed = 0
            self.color_order_seed = int(color_order_seed)
            self.target_color_batch_size = int(plan.get('target_batch_size', self.chunk_size))

        effective_seed = self.partition_seed
        async_context = self._async_loader_context if getattr(self, 'async_mode', False) else None

        if async_context is None:
            self._initialize_local_metadata()
            num_chunks = self.num_chunks
        else:
            num_chunks = int(async_context['num_chunks'])

        self._configure_selected_indices(selected_indices)
        self._iteration_total_samples = self._resolve_iteration_total_samples(
            params,
            selected_indices,
        )
        if self._selected_chunk_ids is not None:
            self._async_loader_sampling_fraction_override = 1.0
            # Sentinel: 0 disables target_rows so async chunks stay unsampled.
            self._async_loader_target_rows_override = 0

        selected_chunk_ids = getattr(self, "_selected_chunk_ids", None)
        if selected_chunk_ids is not None:
            base_order = np.asarray(selected_chunk_ids, dtype=np.int32).copy()
        else:
            base_order = np.arange(num_chunks, dtype=np.int32)

        shuffle_seed = (effective_seed + self.worker_id) if effective_seed is not None else None
        base_order = self._maybe_shuffle_chunk_order(base_order, shuffle_seed)
        loader_order = self._build_loader_chunk_order(base_order, num_chunks)

        if self.data_loader:
            self.data_loader.set_chunk_order(loader_order)

        self._custom_chunk_order = base_order.copy()

        if async_context is not None:
            self._ensure_async_loader_order(base_order, force_restart=True)
            self._initialize_local_metadata()
        elif getattr(self, 'async_loader', None) is not None:
            self.async_loader.set_chunk_order(base_order.tolist())
        self.partitioned_chunks = self._generate_partitions(
            self.max_rounds,
            (effective_seed + self.worker_id) if (effective_seed is not None and self.sample_order == 'random') else effective_seed
        )
        if not self.partitioned_chunks:
            self.partitioned_chunks = [np.array([], dtype=np.int32)] * self.max_rounds

        self.neuron_to_color_cpu = self._build_neuron_to_color_map()

        self.get_cpu_pool()

        num_buffers = getattr(params.processing_config, 'enable_multi_buffering', 0)
        if num_buffers and int(num_buffers) > 1 and not self.multi_buffering_enabled:
            n_features = som_weights.shape[1]
            self.enable_multi_buffering(n_features, num_buffers=int(num_buffers))
            logger.info(
                "Worker %d: Multi-buffering enabled (%d buffers) for color processing",
                self.worker_id,
                int(num_buffers),
            )

            # Store BMU scheduler info
            self.bmu_scheduler_enabled = params.processing_config.enable_adaptive_bmu

            # Initialize BMU scheduler if enabled
            if self.bmu_scheduler_enabled:
                from ...colour_operations.bmu_scheduler import BMUScheduler
                self.bmu_scheduler = BMUScheduler(
                    enabled=True,
                    initial_frequency=params.processing_config.bmu_recalc_initial,
                    decay_type=params.processing_config.bmu_recalc_decay_type,
                    min_samples=params.processing_config.bmu_recalc_min_samples,
                    verbose=params.verbose
                )

            if self.data_loader:
                self.data_loader.clear_color_cache()
            if getattr(self, 'async_loader', None) is not None:
                self.async_loader.clear_color_cache()
        
    def process_full_iteration(self):
        """
        Process all rounds and color sets with synchronization after each color set
        Uses chunk-aligned round processing for efficient memory usage
        """
        with self.device:
            self._start_worker_profile(self.params, label="colors")
            self._reset_timing_stats()
            try:
                # Snapshot starting weights to compute ||dW|| without host transfer
                weights_before = self.gpu_weights.copy() if self.gpu_weights is not None else None
                # Synchronize all workers at iteration start (multi-node support)
                if self.collective_barriers_enabled():
                    try:
                        collective.barrier(group_name=self.collective_group)
                        logger.debug(f"Worker {self.worker_id}: Synchronized at iteration start")
                    except Exception as e:
                        logger.error(f"Worker {self.worker_id}: Start barrier synchronization failed: {e}")
                        raise
                self._maybe_report_progress(stage="iteration_start")
                current_iter = int(getattr(self, 'iteration_count', 0) or 0)
                # Reset pipeline for new iteration if not first
                if current_iter > 0:
                    self.reset_for_iteration()

                # Ensure metadata reflects current loader state
                self._initialize_local_metadata()
                if self.sample_order == 'random':
                    partition_seed = (self.partition_seed + self.worker_id) if self.partition_seed is not None else None
                else:
                    partition_seed = None
                self.partitioned_chunks = self._generate_partitions(self.max_rounds, partition_seed)

                self.processed_mask = cp.zeros(self.total_local_samples, dtype=bool) if not self.no_local_samples else None

                # Initialize local BMU scheduler
                if self.bmu_scheduler:
                    total_local_color_sets = self.max_rounds * len(self.color_sets)
                    self.bmu_scheduler.initialize_iteration(self.total_local_samples, total_local_color_sets)

                # Initialize delta weights for momentum tracking
                if self.params.current_momentum > 0 and self.params.delta_weights is not None:
                    self.delta_weights = cp.asarray(self.params.delta_weights)
                else:
                    self.delta_weights = cp.zeros_like(self.gpu_weights) if self.params.current_momentum > 0 else None

                rng_base = None
                if self.sample_order == 'random':
                    if self.partition_seed is not None:
                        rng_base = int(self.partition_seed)
                    else:
                        rng_base = int(np.random.default_rng().integers(0, 2**32 - 1))

                for partition_idx, chunk_ids in enumerate(self.partitioned_chunks):
                    self._maybe_report_progress(stage=f"partition_{partition_idx}_start")
                    if self.sample_order == 'random':
                        partition_seed = (rng_base + partition_idx) % (2**32 - 1) if rng_base is not None else None
                        partition_rng = np.random.default_rng(partition_seed) if partition_seed is not None else np.random.default_rng()
                    else:
                        partition_rng = None

                    try:
                        self._process_partition(partition_idx, chunk_ids, partition_rng)
                    except Exception as e:
                        logger.error(
                            f"Worker {self.worker_id}: Failed to process partition {partition_idx}: {e}"
                        )
                        raise
                    self._maybe_report_progress(stage=f"partition_{partition_idx}_done")

                # Final weights are already in self.gpu_weights with momentum applied throughout

                # Compute lightweight metrics before clearing temp data
                if weights_before is not None and self.gpu_weights is not None:
                    weight_change_norm = float(cp.linalg.norm(self.gpu_weights - weights_before))
                else:
                    weight_change_norm = 0.0

                # Count local samples processed if mask exists
                samples_processed = int(cp.count_nonzero(self.processed_mask).get()) if self.processed_mask is not None else 0

                # Clear GPU memory for temporary data only
                cp.get_default_memory_pool().free_all_blocks()

                # Synchronize all workers at iteration end (multi-node support)
                if self.collective_barriers_enabled():
                    try:
                        collective.barrier(group_name=self.collective_group)
                        logger.debug(f"Worker {self.worker_id}: Synchronized at iteration end")
                    except Exception as e:
                        logger.error(f"Worker {self.worker_id}: End barrier synchronization failed: {e}")
                        raise
                self._maybe_report_progress(stage="iteration_end")

                # Log aggregated timing information for this iteration
                self._log_timing_summary()

                # Advance iteration counter after successful completion.
                self.iteration_count = current_iter + 1

                # Return status only - weights stay on GPU
                return {
                    'status': 'success',
                    'worker_id': self.worker_id,
                    'iteration': current_iter,
                    'rounds_processed': self.max_rounds,
                    'samples_processed': samples_processed,
                    'weight_change_norm': weight_change_norm
                }
            except Exception:
                self._maybe_report_progress(stage="iteration_failed")
                try:
                    self.destroy_collective_group(self.collective_group)
                except Exception:
                    pass
                raise
