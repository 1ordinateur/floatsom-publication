"""
Base Ray worker with CPU-GPU pipeline for common functionality
Implements YAGNI: Core worker functionality shared by batch and color processors
"""

import cProfile
import io
import math
import os
import pstats

from .numexpr_config import configure_numexpr_threads

# Configure numexpr before numpy import to avoid thread limit errors
configure_numexpr_threads()

import socket
import shutil
import ray
import cupy as cp
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import logging
import concurrent.futures
import copy

from ....data.cpugpu_fast_loader import CPUGPUFastLoader
from ....data.zarr_utils import open_array_read
from ..async_chunk_loader import AsyncChunkLoader
from ...processing_params import AsyncLoadingConfig
from ..local_storage import LocalStorageWipeError, wipe_worker_local_storage
# chunk_size must now be explicitly provided

logger = logging.getLogger(__name__)


class RayPipelineBaseWorker:
    """
    Base Ray worker that provides common CPU-GPU pipeline and GPU functionality
    Single Responsibility: Core GPU and data management for distributed processing
    """
    
    def __init__(self, worker_id: int, num_gpus: int, worker_config: Dict[str, Any], 
                 gpu_id: Optional[int] = None, chunk_size: Optional[int] = None):
        """
        Initialize base worker with CPU-GPU pipeline support
        
        Args:
            worker_id: Worker ID for identification
            num_gpus: Total number of GPUs in cluster
            worker_config: Worker configuration dictionary
            gpu_id: Optional specific GPU ID to use (defaults to Ray-assigned GPU 0)
            chunk_size: Optimal chunk size in number of samples (from processor)
        """
        self.worker_id = worker_id
        self.num_gpus = num_gpus
        self.ray_config = worker_config  # Configuration for the worker
        
        # Use specified GPU or Ray-assigned GPU 0 in isolated environment
        self.gpu_id = gpu_id if gpu_id is not None else 0
        gpu_device_id = self.gpu_id
        cp.cuda.Device(gpu_device_id).use()
        self.device = cp.cuda.Device(gpu_device_id)
        self.assigned_gpu = gpu_device_id
        
        # Data management
        self.data_path = None
        self.local_samples = None
        self.local_bmus = None
        self.data_start_idx = 0
        self.data_end_idx = 0
        self.ram_data = None
        self._ram_data_is_worker_shard = False
        
        # Processing state
        self.current_weights = None
        self.influence_matrix = None
        self.collective_group = None
        
        # Persistent GPU state for weight management
        self.gpu_weights = None  # Persistent weights on GPU across iterations
        self.iteration_count = 0  # Track iterations for broadcasting
        
        # Chunk size must be explicitly provided
        if chunk_size is None:
            raise ValueError("chunk_size must be provided to RayPipelineBaseWorker")
        self.chunk_size = chunk_size  # Effective samples per GPU batch
        self.loader_chunk_size = chunk_size  # Size used when reading from storage
        self.sampling_fraction = 1.0  # Default: process full chunk
        self.sampling_method = "full"
        self.whole_chunk_random = False
        self.randomize_chunk_order = True
        
        # CPUGPUFastLoader for data loading
        self.data_loader = None
        self._loader_initialized = False
        self._current_chunk_iteration = 0
        self._chunk_sampling_counter = 0
        self.async_loader_config: Optional[AsyncLoadingConfig] = None
        
        # Async loading state
        self.async_loader = None
        self.async_mode = False
        self._async_loader_context: Optional[Dict[str, Any]] = None
        self._current_async_order: Optional[Tuple[int, ...]] = None
        self._async_loader_sampling_fraction_override: Optional[float] = None
        self._async_loader_target_rows_override: Optional[int] = None
        
        # Multi-buffering state (disabled by default)
        self.multi_buffering_enabled = False
        self.multi_buffering_num_buffers = 0
        self.transfer_stream = None
        self.compute_stream = None
        self.gpu_buffers = None  # List[cp.ndarray]
        self._multi_buffer_step = 0
        self._buffer_chunk_indices = None
        self._buffer_chunk_info = None
        self._buffer_ready_events = None
        self._buffer_ready_valid = None
        self._buffer_done_events = None
        self._buffer_done_valid = None

        # Shared CPU pool for host-side preprocessing
        self._cpu_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._cpu_pool_workers = 0
        self._node_staged_worker_shards: Dict[int, np.ndarray] = {}

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)
        except Exception:
            return None

    def _finite_stats(self, array: Any) -> Dict[str, Any]:
        """Return lightweight finite/shape stats for a CuPy array (device-side)."""
        arr = cp.asarray(array)
        stats: Dict[str, Any] = {
            "shape": tuple(getattr(arr, "shape", ())),
            "dtype": str(getattr(arr, "dtype", "unknown")),
            "size": int(getattr(arr, "size", 0) or 0),
        }

        try:
            is_finite = cp.isfinite(arr)
            stats["all_finite"] = bool(is_finite.all())
        except Exception as exc:
            stats["all_finite"] = None
            stats["finite_check_error"] = str(exc)
            return stats

        if stats["all_finite"]:
            return stats

        try:
            stats["nan"] = int(cp.isnan(arr).sum().get())
            stats["+inf"] = int(cp.isposinf(arr).sum().get())
            stats["-inf"] = int(cp.isneginf(arr).sum().get())
        except Exception as exc:
            stats["count_error"] = str(exc)

        try:
            stats["min"] = float(cp.nanmin(arr).get())
            stats["max"] = float(cp.nanmax(arr).get())
        except Exception:
            # nanmin/nanmax can fail if all entries are non-finite.
            stats["min"] = None
            stats["max"] = None

        return stats

    def _print_if_nonfinite(
        self,
        name: str,
        array: Any,
        *,
        stage: str,
        params: Any = None,
        extra: Optional[Dict[str, Any]] = None,
        include_diag: bool = False,
    ) -> bool:
        """Print a detailed message when an array becomes non-finite (NaN/Inf)."""
        stats = self._finite_stats(array)
        if stats.get("all_finite"):
            return False

        context: Dict[str, Any] = {
            "worker_id": int(getattr(self, "worker_id", -1)),
            "assigned_gpu": int(getattr(self, "assigned_gpu", -1)),
            "iteration_count": int(getattr(self, "iteration_count", -1)),
            "stage": stage,
        }

        if params is not None:
            context.update(
                {
                    "current_iteration": self._as_float(getattr(params, "current_iteration", None)),
                    "radius": self._as_float(getattr(params, "current_radius", None)),
                    "learning_rate": self._as_float(getattr(params, "current_learning_rate", None)),
                    "momentum": self._as_float(getattr(params, "current_momentum", None)),
                }
            )
            processing_config = getattr(params, "processing_config", None)
            if processing_config is not None:
                context["normalization"] = getattr(processing_config, "normalization", None)
                context["distance_metric"] = getattr(processing_config, "distance_metric", None)

        if extra:
            context.update(extra)

        if include_diag:
            arr = cp.asarray(array)
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                diag = cp.diag(arr)
                context["diag_nonfinite"] = bool((~cp.isfinite(diag)).any())
                context["diag_nan"] = int(cp.isnan(diag).sum().get())

        print(
            f"[FloatSOM][NonFinite] name={name} context={context} stats={stats}",
            flush=True,
        )
        return True

        # Worker profiling state (optional)
        self._worker_profile = None
        self._worker_profile_active = False
        self._worker_profile_written = False
        self._worker_profile_label = None
        self._worker_profile_config = None

        # Debug info
        actual_gpu = cp.cuda.runtime.getDeviceProperties(gpu_device_id)
        logger.info(f"Worker {worker_id} initialized on GPU {self.assigned_gpu}: {actual_gpu['name'].decode()}")
        logger.info(f"Worker {worker_id} using effective chunk size: {self.chunk_size} samples")

        node_id = None
        try:
            runtime_context = ray.get_runtime_context()
            node_id = getattr(runtime_context, "get_node_id", lambda: None)()
        except Exception:
            node_id = None

        ray_gpu_ids = None
        try:
            ray_gpu_ids = ray.get_gpu_ids()
        except Exception:
            ray_gpu_ids = None

        cpu_affinity = None
        try:
            cpu_affinity = sorted(os.sched_getaffinity(0))
        except Exception:
            cpu_affinity = None

        cpu_affinity_summary = "unknown"
        if cpu_affinity:
            cpu_affinity_summary = f"{len(cpu_affinity)} cpus ({cpu_affinity[0]}..{cpu_affinity[-1]})"

        logger.info(
            "Worker %d placement: host=%s node_id=%s pid=%d assigned_cuda_device=%d cuda_visible=%s ray_gpu_ids=%s cpu_affinity=%s",
            worker_id,
            socket.gethostname(),
            node_id,
            os.getpid(),
            int(self.assigned_gpu),
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            ray_gpu_ids,
            cpu_affinity_summary,
        )

    def destroy_collective_group(self, group_name: Optional[str] = None) -> None:
        """Best-effort teardown of a Ray collective group inside this actor."""
        target_name = group_name or self.collective_group
        if not target_name:
            return

        try:
            from ray.util import collective  # type: ignore

            collective.destroy_collective_group(group_name=target_name)
        except Exception as exc:
            logger.debug(
                "Worker %d: destroy_collective_group(%s) failed: %s",
                self.worker_id,
                target_name,
                exc,
            )

        if self.collective_group == target_name:
            self.collective_group = None

    def collective_barriers_enabled(self) -> bool:
        """Return whether optional collective.barrier sync points are enabled."""
        cfg = getattr(self, "ray_config", None)
        if cfg is None:
            return True
        if isinstance(cfg, dict):
            return bool(cfg.get("enable_collective_barriers", True))
        return bool(getattr(cfg, "enable_collective_barriers", True))

    def warmup_collectives(
        self,
        group_name: str,
        iters: int = 1,
        tensor_elements: int = 1024,
    ) -> Dict[str, Any]:
        """
        One-time NCCL warmup.

        Ray's NCCL communicator initialization is typically lazy and may happen
        on the first collective operation. This method forces that cost to be
        paid during setup, not during the first real training iteration.
        """
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("group_name must be a non-empty string")
        if not isinstance(iters, int):
            raise ValueError("iters must be an integer")
        if iters < 0:
            raise ValueError("iters must be >= 0")
        if not isinstance(tensor_elements, int):
            raise ValueError("tensor_elements must be an integer")
        if tensor_elements <= 0:
            raise ValueError("tensor_elements must be > 0")

        if iters == 0:
            return {
                "worker_id": int(self.worker_id),
                "hostname": socket.gethostname(),
                "group_name": group_name,
                "iters": 0,
                "tensor_elements": int(tensor_elements),
                "elapsed_s": 0.0,
            }

        import time
        from ray.util import collective
        from ray.util.collective.types import ReduceOp

        with self.device:
            self.collective_group = group_name

            # Warm up CUDA context + allocator (no RNG; deterministic).
            x = cp.ones((tensor_elements,), dtype=cp.float32)
            y = cp.ones((tensor_elements,), dtype=cp.float32)
            x = x + y
            cp.cuda.get_current_stream().synchronize()

            start = time.perf_counter()
            for _ in range(iters):
                collective.allreduce(x, group_name=group_name, op=ReduceOp.SUM)
                collective.allreduce(y, group_name=group_name, op=ReduceOp.SUM)
            cp.cuda.get_current_stream().synchronize()
            elapsed_s = float(time.perf_counter() - start)

        return {
            "worker_id": int(self.worker_id),
            "hostname": socket.gethostname(),
            "group_name": group_name,
            "iters": int(iters),
            "tensor_elements": int(tensor_elements),
            "elapsed_s": elapsed_s,
        }
    
    def configure_chunk_sampling(
        self,
        loader_chunk_size: int,
        sampling_fraction: float,
        sampling_method: str = "full",
        whole_chunk_random: bool = False,
    ) -> None:
        """
        Configure how many samples are read from storage versus processed on GPU.
        """
        if loader_chunk_size <= 0:
            raise ValueError("loader_chunk_size must be positive")
        if not (0 < sampling_fraction <= 1.0):
            raise ValueError("sampling_fraction must be in (0, 1]")

        self.loader_chunk_size = loader_chunk_size
        self.sampling_fraction = sampling_fraction
        self.sampling_method = str(sampling_method or "full").lower()
        self.whole_chunk_random = bool(whole_chunk_random)
        logger.info(
            "Worker %d sampling configured: method=%s loader chunk size %d, effective chunk size %d, fraction %.4f, whole_chunk_random=%s",
            self.worker_id,
            self.sampling_method,
            self.loader_chunk_size,
            self.chunk_size,
            self.sampling_fraction,
            self.whole_chunk_random,
        )

    def uses_shard_local_random_sampling(self) -> bool:
        """
        Whether this worker should subsample directly from its local shard.
        """
        if str(getattr(self, "sampling_method", "full")).lower() != "random":
            return False
        try:
            fraction = float(getattr(self, "sampling_fraction", 1.0) or 1.0)
        except (TypeError, ValueError):
            fraction = 1.0
        return 0.0 < fraction < 1.0 and not bool(getattr(self, "whole_chunk_random", False))

    def _subsample_loaded_chunk(
        self,
        chunk_data: cp.ndarray,
        chunk_info: Dict[str, Any],
    ) -> Tuple[cp.ndarray, Dict[str, Any]]:
        """
        Apply shard-local random subsampling to a loaded chunk.
        """
        if not self.uses_shard_local_random_sampling():
            return chunk_data, chunk_info

        total_rows = int(chunk_data.shape[0]) if chunk_data.ndim > 0 else 0
        if total_rows <= 0:
            return chunk_data, chunk_info

        desired = max(1, int(round(total_rows * float(self.sampling_fraction))))
        if desired >= total_rows:
            return chunk_data, chunk_info

        chunk_index = int(chunk_info.get("chunk_index", 0) or 0)
        epoch_index = int(getattr(self, "_current_chunk_iteration", 0) or 0)
        seed = (
            (int(self.worker_id) << 48)
            ^ (chunk_index << 24)
            ^ (epoch_index << 8)
        )
        rng = np.random.default_rng(seed)
        indices = rng.choice(total_rows, size=desired, replace=False)
        indices.sort()

        sampled = chunk_data[cp.asarray(indices.astype(np.int32, copy=False), dtype=cp.int32)]
        normalized_info = dict(chunk_info)
        normalized_info["local_size_before_sampling"] = total_rows
        normalized_info["local_size"] = int(sampled.shape[0]) if sampled.ndim > 0 else 0
        normalized_info["sampling_fraction"] = float(self.sampling_fraction)
        normalized_info["source"] = f"{normalized_info.get('source', 'loader')}_shard_random"
        return sampled, normalized_info
    
    
    def _initialize_data_loader(self):
        """
        Initialize the CPUGPUFastLoader.
        Called after data path is set in any data loading method (disk mode only).
        """
        if self.data_path and not self._loader_initialized:
            # Clean up any existing loader first to prevent memory leak
            if self.data_loader is not None:
                logger.warning(f"Worker {self.worker_id}: Cleaning up existing loader before reinitializing")
                del self.data_loader
                self.data_loader = None
            
            assigned_cpus = self._detect_assigned_cpu_count()
            prefetch_threads = max(1, assigned_cpus - 1)

            # Create CPU-GPU loader with fixed chunk size
            self.data_loader = CPUGPUFastLoader(
                array_path=self.data_path,
                chunk_size=self.loader_chunk_size,
                randomize_chunks=self.randomize_chunk_order,
                cpu_workers=prefetch_threads,
            )

            if self.multi_buffering_enabled and self.multi_buffering_num_buffers > 1:
                try:
                    self.data_loader.ensure_pinned_buffers(self.multi_buffering_num_buffers)
                    self.data_loader.set_prefetch_buffer_size(
                        max(6, int(self.multi_buffering_num_buffers))
                    )
                except Exception as exc:
                    logger.warning(
                        "Worker %d: Failed to ensure %d pinned buffers for multi-buffering (%s)",
                        self.worker_id,
                        self.multi_buffering_num_buffers,
                        exc,
                    )
            
            self._loader_initialized = True
            logger.info(f"Worker {self.worker_id}: Initialized FastArrayStore loader with pinned memory")
    
    
    def enable_multi_buffering(self, n_features: int, num_buffers: int = 3) -> None:
        """
        Enable multi-buffering with dual CUDA streams.
        
        Args:
            n_features: Number of features in the data
            num_buffers: Number of GPU transfer buffers to allocate.
        """
        if self.multi_buffering_enabled:
            logger.warning(f"Worker {self.worker_id}: Multi-buffering already enabled")
            return

        if not isinstance(num_buffers, int):
            raise ValueError("num_buffers must be an integer")
        if num_buffers <= 1:
            logger.info(
                "Worker %d: Multi-buffering disabled (num_buffers=%d)",
                self.worker_id,
                num_buffers,
            )
            return
        
        with self.device:
            try:
                # Create streams
                self.transfer_stream = cp.cuda.Stream(non_blocking=True)
                self.compute_stream = cp.cuda.Stream(non_blocking=True)
                
                # Pre-allocate GPU buffers
                effective_size = self.loader_chunk_size if hasattr(self, 'loader_chunk_size') else self.chunk_size
                buffer_shape = (effective_size, n_features) if n_features > 1 else (effective_size,)
                self.gpu_buffers = [
                    cp.zeros(buffer_shape, dtype=cp.float32) for _ in range(num_buffers)
                ]
                self._buffer_chunk_indices = [-1 for _ in range(num_buffers)]
                self._buffer_chunk_info = [None for _ in range(num_buffers)]
                self._buffer_ready_events = [cp.cuda.Event() for _ in range(num_buffers)]
                self._buffer_ready_valid = [False for _ in range(num_buffers)]
                self._buffer_done_events = [cp.cuda.Event() for _ in range(num_buffers)]
                self._buffer_done_valid = [False for _ in range(num_buffers)]
                self._multi_buffer_step = 0

                if self.data_loader is not None:
                    self.data_loader.ensure_pinned_buffers(num_buffers)
                    try:
                        # Keep RAM prefetch window aligned with GPU transfer lookahead.
                        self.data_loader.set_prefetch_buffer_size(max(6, int(num_buffers)))
                    except Exception as exc:
                        logger.warning(
                            "Worker %d: Failed to set loader prefetch depth to %d (%s)",
                            self.worker_id,
                            num_buffers,
                            exc,
                        )

                self.multi_buffering_num_buffers = num_buffers
                self.multi_buffering_enabled = True
                logger.info(
                    "Worker %d: Multi-buffering enabled (%d buffers) with buffer shape %s",
                    self.worker_id,
                    num_buffers,
                    buffer_shape,
                )
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Failed to enable multi-buffering: {e}")
                logger.error("Suggestion: Reduce batch size or disable multi-buffering")
                raise RuntimeError(f"Multi-buffering initialization failed: {e}")

    def _wait_for_buffer_ready(self, buffer_id: int) -> None:
        if not self.multi_buffering_enabled or self.compute_stream is None:
            return
        if (
            self._buffer_ready_valid
            and 0 <= int(buffer_id) < len(self._buffer_ready_valid)
            and self._buffer_ready_valid[int(buffer_id)]
        ):
            self.compute_stream.wait_event(self._buffer_ready_events[int(buffer_id)])

    def _mark_buffer_ready(self, buffer_id: int) -> None:
        if not self.multi_buffering_enabled or self.transfer_stream is None:
            return
        if self._buffer_ready_events is not None:
            self._buffer_ready_events[int(buffer_id)].record(self.transfer_stream)
            self._buffer_ready_valid[int(buffer_id)] = True

    def _wait_for_buffer_available(self, buffer_id: int) -> None:
        if not self.multi_buffering_enabled or self.transfer_stream is None:
            return
        if (
            self._buffer_done_valid
            and 0 <= int(buffer_id) < len(self._buffer_done_valid)
            and self._buffer_done_valid[int(buffer_id)]
        ):
            self.transfer_stream.wait_event(self._buffer_done_events[int(buffer_id)])

    def _mark_buffer_done(self, buffer_id: int) -> None:
        if not self.multi_buffering_enabled or self.compute_stream is None:
            return
        if self._buffer_done_events is not None:
            self._buffer_done_events[int(buffer_id)].record(self.compute_stream)
            self._buffer_done_valid[int(buffer_id)] = True
    
    def _schedule_chunk_to_buffer(
        self,
        chunk_index: int,
        buffer_id: int,
        *,
        advance_cursor: bool = True,
    ) -> Dict[str, Any]:
        if self.data_loader is None:
            raise RuntimeError(f"Worker {self.worker_id}: Data loader not initialized for multi-buffering")
        if self.transfer_stream is None:
            raise RuntimeError(f"Worker {self.worker_id}: Transfer stream not initialized for multi-buffering")
        if self.gpu_buffers is None:
            raise RuntimeError(f"Worker {self.worker_id}: GPU buffers not initialized for multi-buffering")

        buffer_id_int = int(buffer_id)
        if buffer_id_int < 0 or buffer_id_int >= self.multi_buffering_num_buffers:
            raise ValueError("buffer_id out of range")

        # If already scheduled/loaded, reuse metadata.
        if (
            self._buffer_chunk_indices is not None
            and self._buffer_chunk_info is not None
            and self._buffer_chunk_indices[buffer_id_int] == int(chunk_index)
            and self._buffer_chunk_info[buffer_id_int] is not None
        ):
            reused_info = dict(self._buffer_chunk_info[buffer_id_int])
            if bool(advance_cursor) and not bool(reused_info.get("cursor_advanced", False)):
                self.data_loader.mark_chunk_consumed(int(chunk_index))
                reused_info["cursor_advanced"] = True
                self._buffer_chunk_info[buffer_id_int] = reused_info
            return reused_info

        # Pinned staging safety: when reusing a pinned buffer for a new transfer,
        # ensure the previous transfer that used this pinned buffer has completed.
        if (
            self._buffer_ready_valid is not None
            and self._buffer_ready_valid[buffer_id_int]
            and self._buffer_ready_events is not None
            and self._buffer_chunk_indices is not None
            and self._buffer_chunk_indices[buffer_id_int] != -1
            and self._buffer_chunk_indices[buffer_id_int] != int(chunk_index)
        ):
            self._buffer_ready_events[buffer_id_int].synchronize()

        # Ensure previous compute using this buffer has completed before overwriting it.
        self._wait_for_buffer_available(buffer_id_int)

        # Defensive: if we are discarding a prefetched-but-not-consumed buffer,
        # ensure the previous H2D transfer is complete before reusing the pinned
        # staging buffer for this buffer_id.
        if (
            self._buffer_chunk_indices is not None
            and self._buffer_chunk_indices[buffer_id_int] not in (-1, int(chunk_index))
            and self._buffer_done_valid is not None
            and not self._buffer_done_valid[buffer_id_int]
            and self._buffer_ready_valid is not None
            and self._buffer_ready_valid[buffer_id_int]
            and self._buffer_ready_events is not None
        ):
            self._buffer_ready_events[buffer_id_int].synchronize()

        if self._buffer_ready_valid is not None:
            self._buffer_ready_valid[buffer_id_int] = False

        target_buffer = self.gpu_buffers[buffer_id_int]
        with self.transfer_stream:
            chunk_info = self.data_loader.transfer_to_gpu_buffer(
                int(chunk_index),
                target_buffer,
                stream=self.transfer_stream,
                pinned_buffer_index=buffer_id_int,
                advance_cursor=bool(advance_cursor),
            )
            self._mark_buffer_ready(buffer_id_int)

        chunk_info = dict(chunk_info)
        chunk_info['buffer_id'] = buffer_id_int
        chunk_info['cursor_advanced'] = bool(advance_cursor)
        if self._buffer_chunk_indices is not None:
            self._buffer_chunk_indices[buffer_id_int] = int(chunk_index)
        if self._buffer_chunk_info is not None:
            self._buffer_chunk_info[buffer_id_int] = chunk_info
        return chunk_info

    def load_chunk_multi_buffered(
        self,
        chunk_index: int,
        prefetch_indices: Optional[List[int]] = None,
    ) -> Tuple[cp.ndarray, Dict[str, Any]]:
        """
        Load chunk with overlapped transfer using multi-buffering.
        
        Args:
            chunk_index: Index of chunk to load
            prefetch_indices: Optional upcoming chunk indices to prefetch (in call order).
            
        Returns:
            Tuple of (chunk_data, chunk_info)
        """
        if not self.multi_buffering_enabled:
            raise RuntimeError("Multi-buffering not enabled. Call enable_multi_buffering first.")

        if self._using_async_loader():
            return self._load_chunk_async_to_gpu(
                chunk_index,
                prefetch_indices=prefetch_indices,
            )
        
        if self.data_loader is None:
            raise RuntimeError(f"Worker {self.worker_id}: Data loader not initialized for multi-buffering")

        with self.device:
            num_chunks = self.get_num_chunks()
            num_buffers = int(self.multi_buffering_num_buffers)
            step = int(self._multi_buffer_step)
            buffer_id = step % num_buffers

            # Default prefetch: assume sequential chunk indices.
            if prefetch_indices is None:
                prefetch_indices = [
                    int(chunk_index) + offset
                    for offset in range(1, num_buffers)
                    if int(chunk_index) + offset < num_chunks
                ]
            else:
                prefetch_indices = [int(idx) for idx in prefetch_indices]

            # Ensure we only prefetch at most N-1 upcoming chunks.
            prefetch_indices = prefetch_indices[: max(0, num_buffers - 1)]

            # For sync loading, avoid scheduling farther ahead than the loader's
            # RAM prefetch window can keep resident.
            loader_prefetch_size = int(getattr(self.data_loader, "prefetch_buffer_size", 0) or 0)
            if loader_prefetch_size > 0:
                prefetch_indices = prefetch_indices[: max(0, loader_prefetch_size - 1)]

            # Schedule current chunk and prefetches in call order.
            current_info = self._schedule_chunk_to_buffer(
                int(chunk_index),
                buffer_id,
                advance_cursor=True,
            )
            for offset, next_chunk in enumerate(prefetch_indices, start=1):
                if next_chunk < 0 or next_chunk >= num_chunks:
                    continue
                next_buffer_id = (step + offset) % num_buffers
                self._schedule_chunk_to_buffer(
                    int(next_chunk),
                    next_buffer_id,
                    advance_cursor=False,
                )

            self._multi_buffer_step = step + 1

            current_data = self.gpu_buffers[buffer_id]
            current_info = dict(current_info)
            current_info['buffer_id'] = int(buffer_id)
            return current_data, current_info

    def _using_async_loader(self) -> bool:
        return bool(getattr(self, "async_mode", False) and getattr(self, "async_loader", None) is not None)

    def _load_chunk_async_to_gpu(
        self,
        chunk_index: int,
        prefetch_indices: Optional[List[int]] = None,
    ) -> Tuple[cp.ndarray, Dict[str, Any]]:
        """Load one chunk through AsyncChunkLoader and return canonical chunk metadata."""
        if not self._using_async_loader():
            raise RuntimeError(f"Worker {self.worker_id}: Async loader is not active")

        chunk_idx_int = int(chunk_index)
        priority_chunks: List[int] = [chunk_idx_int]
        if prefetch_indices:
            for next_idx in prefetch_indices:
                next_idx_int = int(next_idx)
                if next_idx_int not in priority_chunks:
                    priority_chunks.append(next_idx_int)
        self.async_loader.request_priority_chunks(priority_chunks)

        chunk_data = self.async_loader.get_chunk_when_ready(chunk_idx_int)
        if chunk_data is None:
            raise RuntimeError(
                f"Worker {self.worker_id}: Async loader failed to provide chunk {chunk_idx_int}"
            )

        gpu_chunk = cp.asarray(chunk_data, dtype=cp.float32)
        cp.cuda.get_current_stream().synchronize()
        self.async_loader.mark_chunk_done(chunk_idx_int)

        chunk_info = {
            'chunk_index': chunk_idx_int,
            'local_size': int(gpu_chunk.shape[0]) if gpu_chunk.ndim > 0 else 0,
            'source': 'async_loader',
            'padded': False,
        }
        self.last_chunk_info = chunk_info
        return gpu_chunk, chunk_info

    def _load_chunk_sync_to_gpu(self, chunk_index: int) -> Tuple[cp.ndarray, Dict[str, Any]]:
        """Load one chunk through the synchronous CPUGPUFastLoader path."""
        if self.data_loader is None:
            raise RuntimeError(
                f"Worker {self.worker_id}: Data loader not initialized before load_chunk({chunk_index})"
            )

        gpu_chunk, chunk_info = self.data_loader.get_chunk(
            int(chunk_index),
            advance_cursor=True,
        )
        if gpu_chunk is None:
            raise RuntimeError(
                f"Worker {self.worker_id}: Data loader returned no chunk data for index {chunk_index}"
            )

        normalized_info = dict(chunk_info or {})
        normalized_info.setdefault('chunk_index', int(chunk_index))
        normalized_info.setdefault(
            'local_size',
            int(gpu_chunk.shape[0]) if gpu_chunk.ndim > 0 else 0,
        )
        gpu_chunk, normalized_info = self._subsample_loaded_chunk(gpu_chunk, normalized_info)
        self.last_chunk_info = normalized_info
        logger.debug(f"Worker {self.worker_id}: Loaded chunk {chunk_index}")
        return gpu_chunk, normalized_info

    def load_chunk(self, chunk_index: int) -> cp.ndarray:
        """
        Load a chunk from the loader.
        Modified to support async loading - will wait for chunk if needed.
        
        Args:
            chunk_index: Index of chunk to load
            
        Returns:
            Loaded data as CuPy array
        """
        with self.device:
            if self._using_async_loader():
                gpu_chunk, _ = self._load_chunk_async_to_gpu(int(chunk_index))
                return gpu_chunk

            gpu_chunk, _ = self._load_chunk_sync_to_gpu(int(chunk_index))
            return gpu_chunk
    
    
    
    def write_fast_array_shard(self, data: np.ndarray, worker_id: int) -> Dict[str, Any]:
        """
        Write data shard as local FastArrayStore.
        
        Args:
            data: Data shard for this worker
            worker_id: Worker ID for path naming
            
        Returns:
            Metadata about the FastArrayStore shard
        """
        import os
        from pathlib import Path
        from ....data.fast_array_store import FastArrayStore
        
        # Get local storage path
        local_path = self._get_local_storage_path()
        
        # Create FastArrayStore
        store_path = os.path.join(local_path, f'shard_{worker_id}.fast')
        
        store = FastArrayStore(store_path, mode='w')
        
        # Write data
        arr = store.create(
            shape=data.shape,
            dtype=np.float32,
            chunks=(min(self.loader_chunk_size, data.shape[0]), data.shape[1] if len(data.shape) > 1 else 1)
        )
        arr[:] = data.astype(np.float32)
        store.close()
        
        # Store path for loader
        self.data_path = store_path
        n_samples = data.shape[0]
        
        # Initialize CPUGPUFastLoader for this file
        try:
            self._initialize_data_loader()
            logger.info(f"Worker {self.worker_id}: Wrote {n_samples} samples to FastArrayStore at {store_path}")
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Failed to initialize loader: {e}")
            raise
        
        return {
            'worker_id': self.worker_id,
            'path': store_path,
            'shape': data.shape,
            'n_samples': n_samples
        }
    
    
    def reset_for_iteration(self):
        """
        Reset for a new iteration.
        Called at the start of each training iteration.
        """
        if (self.data_loader and self._loader_initialized) or (
            hasattr(self, 'async_loader') and self.async_loader is not None
        ):
            self._current_chunk_iteration += 1
            logger.debug(f"Worker {self.worker_id}: Reset for iteration {self._current_chunk_iteration}")
            self._chunk_sampling_counter = 0
        if hasattr(self, 'async_loader') and self.async_loader is not None:
            self.async_loader.reset_sampling_epoch()
        
        # Reset multi-buffering state for new iteration
        if self.multi_buffering_enabled:
            self._multi_buffer_step = 0
            if self._buffer_ready_valid is not None:
                for idx in range(len(self._buffer_ready_valid)):
                    self._buffer_ready_valid[idx] = False
            if self._buffer_done_valid is not None:
                for idx in range(len(self._buffer_done_valid)):
                    self._buffer_done_valid[idx] = False
            if self._buffer_chunk_indices is not None:
                for idx in range(len(self._buffer_chunk_indices)):
                    self._buffer_chunk_indices[idx] = -1
            if self._buffer_chunk_info is not None:
                for idx in range(len(self._buffer_chunk_info)):
                    self._buffer_chunk_info[idx] = None
    
    def get_num_chunks(self) -> int:
        """
        Get the total number of chunks based on fixed chunk size.
        
        Returns:
            Number of chunks that can be loaded
        """
        # Handle async loading mode
        if hasattr(self, 'async_loader') and self.async_loader:
            return self.async_loader.total_chunks
        
        # Handle standard data loader mode
        if not self.data_loader:
            if not self.data_path:
                raise RuntimeError("Data not initialized. Call write_fast_array_shard first.")
            # Initialize loader if needed
            self._initialize_data_loader()
        
        # Get from loader
        info = self.data_loader.get_info()
        num_chunks = info['total_chunks']
        
        logger.debug(f"Worker {self.worker_id}: {num_chunks} chunks available")
        
        return num_chunks
    
    def load_selective(self, start_idx: int, end_idx: int) -> cp.ndarray:
        """
        Load specific index range from FastArrayStore.
        Aligns to chunk boundaries for efficiency.
        
        Args:
            start_idx: Start index of the range to load
            end_idx: End index (exclusive) of the range to load
            
        Returns:
            Data loaded to GPU
        """
        if self.async_mode and getattr(self, 'async_loader', None) is not None:
            return self._load_selective_async(start_idx, end_idx)

        if not self.data_loader:
            if self.data_path:
                self._initialize_data_loader()
            elif self.ram_data is not None:
                with self.device:
                    relative_start = start_idx - self.data_start_idx
                    relative_end = relative_start + (end_idx - start_idx)
                    if relative_start < 0 or relative_end > len(self.ram_data):
                        raise RuntimeError(
                            f"Worker {self.worker_id}: Requested RAM slice {start_idx}:{end_idx} outside allocated range"
                        )
                    return cp.asarray(self.ram_data[relative_start:relative_end], dtype=cp.float32)
            else:
                raise RuntimeError("Loader not initialized. Call write_fast_array_shard first.")

        if not self.data_loader:
            raise RuntimeError("Loader not initialized. Call write_fast_array_shard first.")

        global_start = int(getattr(self, 'data_start_idx', 0) or 0)
        global_end = getattr(self, 'data_end_idx', None)
        if global_end is None or global_end <= global_start:
            info = self.data_loader.get_info() if self.data_loader else {}
            local_n_samples = int(info.get('n_samples', 0) or 0)
            if local_n_samples <= 0 and hasattr(self, 'n_samples'):
                local_n_samples = int(getattr(self, 'n_samples', 0) or 0)
            global_end = global_start + max(0, local_n_samples)

        if start_idx < global_start or end_idx > global_end or end_idx <= start_idx:
            raise RuntimeError(
                f"Worker {self.worker_id}: Requested range {start_idx}:{end_idx} "
                f"outside worker span {global_start}:{global_end}"
            )

        rel_start = start_idx - global_start
        rel_end = end_idx - global_start

        with self.device:
            # Calculate which chunks we need
            start_chunk = rel_start // self.loader_chunk_size
            end_chunk = (rel_end + self.loader_chunk_size - 1) // self.loader_chunk_size
            
            # Load chunks from loader
            all_data = []
            for chunk_idx in range(start_chunk, end_chunk):
                chunk_data, _ = self.data_loader.get_chunk(
                    chunk_idx,
                    advance_cursor=True,
                )
                if chunk_data is not None:
                    all_data.append(chunk_data)
            
            if not all_data:
                raise RuntimeError(
                    f"Worker {self.worker_id}: No data returned for selective load {start_idx}:{end_idx}"
                )
            
            # Concatenate if multiple chunks
            if len(all_data) > 1:
                data = cp.concatenate(all_data, axis=0)
            else:
                data = all_data[0]
            
            # Return only the selected portion
            chunk_start = start_chunk * self.loader_chunk_size
            relative_start = rel_start - chunk_start
            relative_end = relative_start + (rel_end - rel_start)
            
            logger.debug(f"Worker {self.worker_id}: Loaded indices {start_idx}-{end_idx}")
            
            return data[relative_start:relative_end]

    def _load_selective_async(self, start_idx: int, end_idx: int) -> cp.ndarray:
        """
        Load a selective range when using the async chunk loader in RAM mode.
        """
        if not hasattr(self, 'async_loader') or self.async_loader is None:
            raise RuntimeError("Async loader not initialized.")

        if self.data_start_idx is None or self.data_end_idx is None:
            raise RuntimeError("Worker RAM range not initialized for async loading.")

        relative_start = start_idx - self.data_start_idx
        relative_end = end_idx - self.data_start_idx

        total_span = self.data_end_idx - self.data_start_idx
        if relative_start < 0 or relative_end > total_span:
            raise RuntimeError(
                f"Worker {self.worker_id}: Requested async slice {start_idx}:{end_idx} outside allocated range"
            )

        if relative_start >= relative_end:
            raise ValueError(
                f"Worker {self.worker_id}: Invalid selective range {start_idx}:{end_idx}"
            )

        with self.device:
            start_chunk = relative_start // self.loader_chunk_size
            end_chunk = (relative_end + self.loader_chunk_size - 1) // self.loader_chunk_size

            gpu_chunks: List[cp.ndarray] = []
            total_rows = 0

            for chunk_idx in range(start_chunk, end_chunk):
                gpu_chunk, _ = self._load_chunk_async_to_gpu(chunk_idx)
                total_rows += gpu_chunk.shape[0]
                gpu_chunks.append(gpu_chunk)

            if not gpu_chunks:
                raise RuntimeError(
                    f"Worker {self.worker_id}: Async loader returned no chunks for range {start_idx}:{end_idx}"
                )

            if len(gpu_chunks) == 1:
                data = gpu_chunks[0]
            else:
                data = cp.concatenate(gpu_chunks, axis=0)

            chunk_span_start = start_chunk * self.loader_chunk_size
            slice_start = relative_start - chunk_span_start
            slice_end = slice_start + (relative_end - relative_start)
            if slice_start < 0 or slice_end > data.shape[0]:
                raise RuntimeError(
                    f"Worker {self.worker_id}: Async slice mapping overflow for range "
                    f"{start_idx}:{end_idx} within chunk span start={chunk_span_start}, rows={data.shape[0]}"
                )
            selected = data[slice_start:slice_end]

            logger.debug(
                f"Worker {self.worker_id}: Async-loaded indices {start_idx}-{end_idx} "
                f"into GPU buffer of shape {data.shape} and returned slice {selected.shape} "
                f"(rows {total_rows})"
            )

            return selected
    
    def align_indices_to_chunks(self, indices: Optional[np.ndarray]) -> Optional[List[Tuple[int, int]]]:
        """
        Convert random indices to contiguous chunk-aligned ranges for efficient pipeline loading.
        
        Args:
            indices: Array of selected indices (can be None for all data)
            
        Returns:
            List of (start, end) tuples representing chunk-aligned ranges,
            or None if indices is None
        """
        if indices is None:
            return None
        
        if len(indices) == 0:
            return []

        global_start = int(getattr(self, 'data_start_idx', 0) or 0)
        global_end = getattr(self, 'data_end_idx', None)
        if global_end is None or global_end <= global_start:
            local_n_samples = None
            if self.data_loader is not None:
                info = self.data_loader.get_info()
                local_n_samples = int(info.get('n_samples', 0) or 0)
            if (local_n_samples is None or local_n_samples <= 0) and hasattr(self, 'n_samples'):
                local_n_samples = int(getattr(self, 'n_samples', 0) or 0)
            global_end = global_start + max(0, int(local_n_samples or 0))

        indices = np.asarray(indices)
        in_span = indices[(indices >= global_start) & (indices < global_end)]
        if in_span.size == 0:
            return []

        local_offsets = in_span - global_start
        chunk_ids = np.unique(local_offsets // self.loader_chunk_size)
        if chunk_ids.size == 0:
            return []

        chunk_ids = np.sort(chunk_ids)
        ranges = []
        for chunk_id in chunk_ids:
            start = global_start + int(chunk_id) * self.loader_chunk_size
            end = start + self.loader_chunk_size
            if end <= global_start or start >= global_end:
                continue
            start = max(start, global_start)
            end = min(end, global_end)
            if end > start:
                ranges.append((start, end))

        logger.debug(
            f"Worker {self.worker_id}: Deterministic chunk mapping produced {len(ranges)} ranges"
        )
        self._chunk_sampling_counter += 1
        return ranges
    
    
    def use_existing_fast_array(self, fast_array_path: str) -> Dict[str, Any]:
        """
        Use an existing FastArrayStore file.
        
        Args:
            fast_array_path: Path to existing FastArrayStore
            
        Returns:
            Metadata about the FastArrayStore file
        """
        from ....data.fast_array_store import FastArrayStore
        
        logger.info(f"Worker {self.worker_id}: Using existing FastArrayStore: {fast_array_path}")
        
        # Store path for loader
        self.data_path = fast_array_path
        
        # Open FastArrayStore to get metadata
        store = FastArrayStore(fast_array_path, mode='r')
        n_samples = store.n_samples
        n_features = store.n_features
        store.close()
        
        # Initialize CPUGPUFastLoader
        try:
            self._initialize_data_loader()
            logger.info(f"Worker {self.worker_id}: Initialized loader for existing FastArrayStore with {n_samples} samples")
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Failed to initialize loader: {e}")
            raise
        
        return {
            'worker_id': self.worker_id,
            'path': fast_array_path,
            'shape': (n_samples, n_features),
            'n_samples': n_samples,
            'n_features': n_features,
            'existing_file': True
        }
    
    def use_ram_data(
        self,
        data_ref,
        start_idx: int,
        end_idx: int,
        worker_shard: bool = False,
    ) -> Dict[str, Any]:
        """
        Use data directly from Ray object store (RAM mode).
        
        Args:
            data_ref: Full data array or this worker's pre-sharded RAM object
            start_idx: Global start index for this worker's slice
            end_idx: Global end index for this worker's slice
            worker_shard: Whether `data_ref` already contains only this worker's shard
            
        Returns:
            Metadata about the RAM data
        """
        logger.info(f"Worker {self.worker_id}: Using RAM mode data (samples {start_idx}-{end_idx})")

        if worker_shard:
            worker_data = np.asarray(data_ref, dtype=np.float32)
        else:
            full_data = data_ref
            worker_data = np.asarray(full_data[start_idx:end_idx], dtype=np.float32)

        # Store for loader
        self.ram_data = worker_data
        self._ram_data_is_worker_shard = bool(worker_shard)
        self.data_path = None  # No file path in RAM mode
        self.data_start_idx = start_idx
        self.data_end_idx = end_idx
        
        # Get dimensions
        n_samples = worker_data.shape[0]
        n_features = worker_data.shape[1] if len(worker_data.shape) > 1 else 1
        
        # Initialize CPUGPUFastLoader in RAM mode
        if self.data_loader is not None:
            del self.data_loader
            self.data_loader = None
        
        self.data_loader = CPUGPUFastLoader(
            array_path=None,
            chunk_size=self.loader_chunk_size,
            randomize_chunks=self.randomize_chunk_order,
            worker_id=self.worker_id,
            n_workers_per_node=1,
            ram_data=worker_data,
            ram_mode=True
        )
        self._loader_initialized = True
        
        logger.info(f"Worker {self.worker_id}: RAM mode initialized with {n_samples} samples "
                   f"({n_samples * n_features * 4 / (1024**3):.3f} GB)")
        
        return {
            'worker_id': self.worker_id,
            'mode': 'RAM',
            'shape': (n_samples, n_features),
            'n_samples': n_samples,
            'n_features': n_features,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'data_size_gb': n_samples * n_features * 4 / (1024**3)
        }

    def stage_zarr_node_span(
        self,
        zarr_path: str,
        node_start: int,
        node_end: int,
        worker_specs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Stage one contiguous node span and prepare local worker shards on this worker.
        """
        if node_start < 0 or node_end < node_start:
            raise ValueError(f"Invalid node span {node_start}:{node_end}")
        if not worker_specs:
            raise ValueError("worker_specs must be non-empty for node-local RAM staging")

        z = open_array_read(zarr_path).array
        shape = getattr(z, "shape", None) or ()
        if not shape:
            raise ValueError(f"Zarr array has no shape: {zarr_path}")

        total_samples = int(shape[0])
        n_features = int(shape[1]) if len(shape) > 1 else 1
        if int(node_end) > total_samples:
            raise ValueError(
                f"Invalid node span {node_start}:{node_end} for total samples {total_samples}"
            )

        node_rows = int(node_end - node_start)
        bytes_per_row = int(n_features) * 4 if len(shape) > 1 else 4
        node_span_bytes = int(node_rows) * int(bytes_per_row)

        mem_limit_bytes = self._detect_job_memory_limit_bytes()
        mem_used_bytes = self._detect_cgroup_memory_usage_bytes()
        workers_on_node = max(1, int(self._detect_gpu_workers_on_node() or 1))
        if mem_limit_bytes is not None and mem_used_bytes is not None:
            headroom_bytes = max(0, int(mem_limit_bytes) - int(mem_used_bytes))
        elif mem_limit_bytes is not None:
            reserve_bytes = max(2 * 1024**3, int(mem_limit_bytes) // 10)
            headroom_bytes = max(0, int(mem_limit_bytes) - reserve_bytes)
        else:
            try:
                import psutil  # type: ignore

                headroom_bytes = int(psutil.virtual_memory().available)
            except Exception:
                headroom_bytes = None

        if headroom_bytes is not None:
            per_node_headroom = max(0, int(headroom_bytes) * 9 // 10)
            estimated_peak = int(node_span_bytes * 5 // 2) + 512 * 1024**2
            if estimated_peak > per_node_headroom:
                raise RuntimeError(
                    "Insufficient memory headroom for node-local RAM staging: "
                    f"peak_est={estimated_peak / (1024**3):.2f} GB > "
                    f"per_node_headroom={per_node_headroom / (1024**3):.2f} GB "
                    f"(workers_on_node={workers_on_node})"
                )

        print(
            f"[node-stage leader={int(self.worker_id)} host={socket.gethostname()}] start "
            f"rows={node_rows:,} range={int(node_start):,}:{int(node_end):,} "
            f"workers={len(worker_specs)}",
            flush=True,
        )

        node_data = np.asarray(z[int(node_start):int(node_end)], dtype=np.float32)
        if not node_data.flags["C_CONTIGUOUS"]:
            node_data = np.ascontiguousarray(node_data)

        max_workers = min(len(worker_specs), max(1, (self._detect_assigned_cpu_count() or 1) - 1))
        shard_results: Dict[int, np.ndarray] = {}

        def _build_shard(spec: Dict[str, Any]) -> tuple[int, np.ndarray]:
            worker_idx = int(spec["worker_idx"])
            local_start = int(spec["local_start"])
            local_end = int(spec["local_end"])
            shard = np.array(node_data[local_start:local_end], dtype=np.float32, copy=True)
            if not shard.flags["C_CONTIGUOUS"]:
                shard = np.ascontiguousarray(shard)
            return worker_idx, shard

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"node-stage-{self.worker_id}",
        ) as executor:
            futures = [executor.submit(_build_shard, spec) for spec in worker_specs]
            for future in concurrent.futures.as_completed(futures):
                worker_idx, shard = future.result()
                shard_results[worker_idx] = shard

        self._node_staged_worker_shards = shard_results
        del node_data

        print(
            f"[node-stage leader={int(self.worker_id)} host={socket.gethostname()}] completed "
            f"workers={len(worker_specs)} copy_workers={max_workers}",
            flush=True,
        )
        return {
            "hostname": socket.gethostname(),
            "node_start": int(node_start),
            "node_end": int(node_end),
            "workers_staged": int(len(worker_specs)),
            "copy_workers_used": int(max_workers),
        }

    def get_node_staged_worker_shard(self, worker_idx: int) -> np.ndarray:
        worker_idx_int = int(worker_idx)
        if worker_idx_int not in self._node_staged_worker_shards:
            raise RuntimeError(
                f"Worker {self.worker_id}: missing node-staged shard for worker {worker_idx_int}"
            )
        return self._node_staged_worker_shards[worker_idx_int]

    def clear_node_staged_shards(self) -> bool:
        self._node_staged_worker_shards.clear()
        return True

    def use_node_staged_ram_shard(
        self,
        leader_worker,
        worker_idx: int,
        start_idx: int,
        end_idx: int,
    ) -> Dict[str, Any]:
        """
        Fetch this worker's prepared RAM shard from a node-local staging actor.
        """
        import ray

        shard = ray.get(leader_worker.get_node_staged_worker_shard.remote(int(worker_idx)))
        return self.use_ram_data(shard, start_idx, end_idx, worker_shard=True)

    def use_local_node_staged_ram_shard(
        self,
        worker_idx: int,
        start_idx: int,
        end_idx: int,
    ) -> Dict[str, Any]:
        """Use a shard already staged on this worker without a remote self-call."""
        shard = self.get_node_staged_worker_shard(int(worker_idx))
        return self.use_ram_data(shard, start_idx, end_idx, worker_shard=True)

    def use_zarr_ram_slice(self, zarr_path: str, start_idx: int, end_idx: int) -> Dict[str, Any]:
        """
        Load a worker-local slice of a Zarr array directly into RAM and initialize
        the RAM-mode CPUGPUFastLoader.

        This avoids multi-node broadcast of a full `ray.put(z[:])` object by letting
        each worker read only its assigned slice from shared storage.
        """
        import time

        requested_copy_workers = 1

        try:
            from numcodecs import blosc  # type: ignore

            blosc.set_nthreads(1)
        except Exception:
            pass

        if start_idx < 0 or end_idx < start_idx:
            raise ValueError(f"Invalid zarr slice range {start_idx}:{end_idx}")

        z = open_array_read(zarr_path).array
        shape = getattr(z, "shape", None) or ()
        if not shape:
            raise ValueError(f"Zarr array has no shape: {zarr_path}")

        total_samples = int(shape[0])
        n_features = int(shape[1]) if len(shape) > 1 else 1
        if end_idx > total_samples:
            raise ValueError(
                f"Invalid zarr slice range {start_idx}:{end_idx} for total samples {total_samples}"
            )

        shard_rows = int(end_idx - start_idx)
        bytes_per_row = int(n_features) * 4 if len(shape) > 1 else 4
        bytes_total = int(shard_rows) * int(bytes_per_row)

        logger.info(
            "Worker %d: Loading Zarr slice to RAM: path=%s rows=%d features=%d range=%d:%d",
            self.worker_id,
            zarr_path,
            shard_rows,
            n_features,
            start_idx,
            end_idx,
        )
        print(
            f"[worker {int(self.worker_id)}][zarr-ram-load] start "
            f"host={socket.gethostname()} rows={shard_rows:,} features={n_features:,} "
            f"range={int(start_idx):,}:{int(end_idx):,} requested_copy_workers={requested_copy_workers}",
            flush=True,
        )

        # Clear any prior data loaders/async state (benchmarks may reuse actors).
        if self.async_loader is not None:
            self.async_loader.stop()
            self.async_loader = None
        self._async_loader_context = None
        self._current_async_order = None
        self.async_mode = False

        if self.data_loader is not None:
            self.data_loader.close()
            self.data_loader = None

        if self.ram_data is not None:
            del self.ram_data
            self.ram_data = None

        # Allocate worker-local RAM buffer and fill in blocks.
        load_start = time.perf_counter()
        if len(shape) > 1:
            worker_data = np.empty((shard_rows, n_features), dtype=np.float32)
        else:
            worker_data = np.empty((shard_rows,), dtype=np.float32)

        copy_workers_used = 0
        if shard_rows > 0:
            target_bytes = self._resolve_zarr_ram_block_target_bytes()
            block_rows = max(1, min(shard_rows, target_bytes // max(1, bytes_per_row)))

            zarr_chunks = getattr(z, "chunks", None)
            if zarr_chunks:
                try:
                    zarr_chunk_rows = int(zarr_chunks[0] or 0)
                except (TypeError, ValueError):
                    zarr_chunk_rows = 0
                if zarr_chunk_rows > 0 and zarr_chunk_rows <= block_rows:
                    aligned = (block_rows // zarr_chunk_rows) * zarr_chunk_rows
                    block_rows = max(1, aligned or zarr_chunk_rows)

            total_blocks = max(1, math.ceil(shard_rows / block_rows))
            progress_interval = max(1, total_blocks // 10)
            z_dtype = np.dtype(getattr(z, "dtype", np.float32))
            mem_limit_bytes = self._detect_job_memory_limit_bytes()
            mem_used_bytes = self._detect_cgroup_memory_usage_bytes()
            workers_on_node = max(1, int(self._detect_gpu_workers_on_node() or 1))
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
                per_worker_headroom = max(0, int(headroom_bytes) * 9 // 10 // workers_on_node)
                if per_worker_headroom <= 0:
                    raise RuntimeError(
                        "Insufficient memory headroom for worker-local Zarr RAM loading: "
                        "per_worker_headroom=0 bytes "
                        f"(workers_on_node={workers_on_node} requested_copy_workers={requested_copy_workers})"
                    )

            src_itemsize = int(z_dtype.itemsize)
            zarr_chunk_cols = 0
            if zarr_chunks:
                try:
                    if len(zarr_chunks) > 1:
                        zarr_chunk_cols = int(zarr_chunks[1] or 0)
                except (TypeError, ValueError):
                    zarr_chunk_cols = 0
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
                if z_dtype != np.float32:
                    peak += int(dst_bytes)
                if chunk_src_bytes is not None:
                    aligned_to_row_chunks = (int(start_idx) % int(zarr_chunk_rows) == 0) and (
                        int(rows) % int(zarr_chunk_rows) == 0
                    )
                    if not aligned_to_row_chunks or int(rows) < int(zarr_chunk_rows):
                        peak += int(chunk_src_bytes)
                peak = int(peak * 13 // 10) + 512 * 1024**2
                return int(peak)

            peak_bytes = _estimate_peak_bytes(block_rows)
            if per_worker_headroom is not None and peak_bytes > per_worker_headroom:
                max_iters = 12
                while max_iters > 0 and block_rows > 1 and peak_bytes > per_worker_headroom:
                    block_rows = max(1, block_rows // 2)
                    peak_bytes = _estimate_peak_bytes(block_rows)
                    max_iters -= 1
                if peak_bytes > per_worker_headroom:
                    raise RuntimeError(
                        "Insufficient memory headroom for worker-local Zarr RAM loading: "
                        f"peak_est={peak_bytes / (1024**3):.2f} GB > "
                        f"per_worker_headroom={per_worker_headroom / (1024**3):.2f} GB "
                        f"(workers_on_node={workers_on_node} requested_copy_workers={requested_copy_workers})"
                    )

            total_blocks = max(1, math.ceil(shard_rows / block_rows))
            progress_interval = max(1, total_blocks // 10)
            block_tasks = [
                (
                    int(block_idx),
                    int(offset),
                    int(min(offset + block_rows, shard_rows)),
                )
                for block_idx, offset in enumerate(range(0, shard_rows, block_rows), start=1)
            ]
            copy_workers_used = 1

            print(
                f"[worker {int(self.worker_id)}][zarr-ram-load] config "
                f"block_rows={block_rows:,} total_blocks={total_blocks} "
                f"copy_workers_used={copy_workers_used}"
                + (
                    f" per_worker_headroom_gb={per_worker_headroom / (1024**3):.2f}"
                    if per_worker_headroom is not None
                    else ""
                ),
                flush=True,
            )

            def _should_report_progress(*, rows_done: int, blocks_done: int) -> bool:
                return (
                    blocks_done == 1
                    or blocks_done == total_blocks
                    or blocks_done % progress_interval == 0
                    or rows_done >= shard_rows
                )

            def _print_progress(*, rows_done: int, blocks_done: int) -> None:
                progress_pct = (100.0 * rows_done / shard_rows) if shard_rows > 0 else 100.0
                print(
                    f"[worker {int(self.worker_id)}][zarr-ram-load] progress "
                    f"{rows_done:,}/{shard_rows:,} rows ({progress_pct:.1f}%) "
                    f"blocks={blocks_done}/{total_blocks}",
                    flush=True,
                )

            def _read_block(z_local: Any, selection: Any, destination: np.ndarray) -> None:
                if z_dtype == np.float32:
                    wrote_direct = False
                    try:
                        if hasattr(z_local, "get_basic_selection"):
                            z_local.get_basic_selection(selection, out=destination)
                            wrote_direct = True
                        elif hasattr(z_local, "get_orthogonal_selection"):
                            z_local.get_orthogonal_selection(selection, out=destination)
                            wrote_direct = True
                        elif hasattr(z_local, "read_direct"):
                            try:
                                z_local.read_direct(destination, source_sel=selection)
                            except TypeError:
                                z_local.read_direct(destination, selection)
                            wrote_direct = True
                    except Exception:
                        wrote_direct = False

                    if wrote_direct:
                        return

                block = z_local[selection]
                if block.dtype != np.float32:
                    block = block.astype(np.float32, copy=False)
                flags = getattr(block, "flags", None)
                if flags is not None and not bool(getattr(flags, "c_contiguous", True)):
                    block = np.ascontiguousarray(block)
                destination[...] = block

            rows_done = 0
            for blocks_done, (_, dst_start, dst_end) in enumerate(block_tasks, start=1):
                src_start = int(start_idx + dst_start)
                src_end = int(start_idx + dst_end)
                if len(shape) > 1:
                    selection = (slice(src_start, src_end), slice(None))
                else:
                    selection = slice(src_start, src_end)
                _read_block(z, selection, worker_data[dst_start:dst_end])
                rows_done += int(dst_end - dst_start)
                if _should_report_progress(rows_done=rows_done, blocks_done=blocks_done):
                    _print_progress(rows_done=rows_done, blocks_done=blocks_done)

        elapsed_s = float(time.perf_counter() - load_start)
        throughput_gb_s = (bytes_total / (1024**3)) / elapsed_s if elapsed_s > 0 else 0.0

        # Store for loader and selective indexing.
        self.ram_data = worker_data
        self.data_path = None
        self.data_start_idx = int(start_idx)
        self.data_end_idx = int(end_idx)
        self.ram_mode = True

        # Initialize CPUGPUFastLoader in RAM mode.
        chunk_size = int(getattr(self, "loader_chunk_size", None) or self.chunk_size)
        self.data_loader = CPUGPUFastLoader(
            array_path=None,
            chunk_size=chunk_size,
            randomize_chunks=self.randomize_chunk_order,
            worker_id=self.worker_id,
            n_workers_per_node=1,
            ram_data=worker_data,
            ram_mode=True,
        )
        self._loader_initialized = True

        logger.info(
            "Worker %d: Zarr slice loaded to RAM: %.3f GB in %.3fs (%.2f GB/s)",
            self.worker_id,
            bytes_total / (1024**3),
            elapsed_s,
            throughput_gb_s,
        )
        print(
            f"[worker {int(self.worker_id)}][zarr-ram-load] completed "
            f"size_gb={bytes_total / (1024**3):.3f} elapsed_s={elapsed_s:.3f} "
            f"throughput_gb_s={throughput_gb_s:.2f}",
            flush=True,
        )

        return {
            "worker_id": int(self.worker_id),
            "hostname": socket.gethostname(),
            "mode": "RAM",
            "strategy": "zarr_to_worker_ram",
            "shape": (shard_rows, n_features) if len(shape) > 1 else (shard_rows,),
            "n_samples": int(shard_rows),
            "n_features": int(n_features),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "data_size_gb": float(bytes_total / (1024**3)),
            "copy_workers_used": int(copy_workers_used),
            "elapsed_s": float(elapsed_s),
            "throughput_gb_s": float(throughput_gb_s),
        }

    def _resolve_zarr_ram_block_target_bytes(self) -> int:
        """Return the target per-block read size for worker-local RAM loading."""
        return 256 * 1024**2

    def create_local_fast_array_from_zarr(
        self,
        zarr_path: str,
        start_idx: int,
        end_idx: int,
        shard_id: int,
        run_id: str,
    ) -> Dict[str, Any]:
        """
        Create a local FastArrayStore shard directly from a Zarr array.

        Args:
            zarr_path: Path to source Zarr array
            start_idx: Global start index for this shard
            end_idx: Global end index (exclusive) for this shard
            shard_id: Shard ID for naming/debugging
            run_id: Unique run identifier to prevent path collisions

        Returns:
            Metadata about the local shard
        """
        from ....data.shard_staging import stage_zarr_slice

        local_path = self._get_local_storage_path()
        local_store_path = os.path.join(local_path, "runs", run_id, f"shard_{shard_id}")

        bytes_per_row = 4
        try:
            z = open_array_read(zarr_path).array
            shape = getattr(z, "shape", None) or ()
            n_features_hint = int(shape[1]) if len(shape) > 1 else 1
            bytes_per_row = int(n_features_hint) * 4
        except Exception:
            bytes_per_row = 4

        last_report_rows = 0

        def _progress_cb(rows_done: int, total_rows: int, elapsed_s: float) -> None:
            nonlocal last_report_rows
            if total_rows <= 0:
                return
            report_every_rows = max(1, total_rows // 10)
            if rows_done - last_report_rows >= report_every_rows or rows_done >= total_rows:
                last_report_rows = rows_done
                pct = (rows_done / total_rows * 100.0) if total_rows else 100.0
                throughput_gb_s = (
                    (rows_done * bytes_per_row / (1024**3)) / elapsed_s
                    if elapsed_s > 0
                    else 0.0
                )
                logger.info(
                    "Worker %d: Zarr shard %d progress %d/%d (%.1f%%, %.2f GB/s)",
                    self.worker_id,
                    shard_id,
                    rows_done,
                    total_rows,
                    pct,
                    throughput_gb_s,
                )
                print(
                    f"Worker {self.worker_id}: Zarr shard {shard_id} progress "
                    f"{rows_done}/{total_rows} ({pct:.1f}%, {throughput_gb_s:.2f} GB/s)",
                    flush=True,
                )

        assigned_cpus = self._detect_assigned_cpu_count()
        mem_limit_bytes = self._detect_job_memory_limit_bytes()
        mem_used_bytes = self._detect_cgroup_memory_usage_bytes()
        workers_on_node = self._detect_gpu_workers_on_node()

        meta = stage_zarr_slice(
            zarr_path=zarr_path,
            local_store_path=local_store_path,
            start_idx=start_idx,
            end_idx=end_idx,
            shard_id=shard_id,
            loader_chunk_size=self.loader_chunk_size,
            assigned_cpus=assigned_cpus,
            mem_limit_bytes=mem_limit_bytes,
            mem_used_bytes=mem_used_bytes,
            workers_on_node=workers_on_node,
            progress_cb=_progress_cb,
        )

        self.data_path = meta["path"]
        self.data_start_idx = start_idx
        self.data_end_idx = end_idx
        self.n_samples = int(meta["n_samples"])
        self.n_features = int(meta["n_features"])
        self._initialize_data_loader()

        logger.info(
            "Worker %d: Zarr shard %d completed in %.2fs (%.2f GB/s)",
            self.worker_id,
            shard_id,
            float(meta.get("elapsed_s", 0.0)),
            float(meta.get("throughput_gb_s", 0.0)),
        )

        return {"worker_id": self.worker_id, **meta}
    
    def create_local_fast_array(
        self,
        source_path: str,
        start_idx: int,
        end_idx: int,
        shard_id: int,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a local FastArrayStore copy from source.
        
        Args:
            source_path: Path to source FastArrayStore
            start_idx: Start index in source array
            end_idx: End index in source array
            shard_id: Shard ID for naming
            
        Returns:
            Metadata about the local copy
        """
        from ....data.shard_staging import stage_fast_array_slice

        local_path = self._get_local_storage_path()
        if run_id:
            local_store_path = os.path.join(local_path, "runs", run_id, f"shard_{shard_id}")
        else:
            local_store_path = os.path.join(local_path, f"worker_{self.worker_id}_data")

        copy_workers = self._detect_assigned_cpu_count()
        meta = stage_fast_array_slice(
            source_path=source_path,
            local_store_path=local_store_path,
            start_idx=start_idx,
            end_idx=end_idx,
            loader_chunk_size=self.loader_chunk_size,
            copy_workers=copy_workers,
            executor_cls=concurrent.futures.ThreadPoolExecutor,
        )

        # Store path and initialize loader
        self.data_path = local_store_path
        self.data_start_idx = start_idx
        self.data_end_idx = end_idx
        self.n_samples = int(meta["n_samples"])
        self.n_features = int(meta["n_features"])
        self._initialize_data_loader()

        logger.info(f"Worker {self.worker_id}: Created local FastArrayStore at {local_store_path}")

        return {
            'worker_id': self.worker_id,
            'n_samples': int(meta["n_samples"]),
            'n_features': int(meta["n_features"]),
            'path': local_store_path
        }
    
    def _get_local_storage_path(self) -> str:
        """
        Get local storage path for this worker.
        """
        base = self._get_local_storage_base()
        host_dir = os.path.join(base, socket.gethostname())
        worker_dir = os.path.join(host_dir, f'ray_worker_{self.worker_id}')
        os.makedirs(worker_dir, exist_ok=True)
        return worker_dir

    def _get_local_storage_base(self) -> str:
        if not self.ray_config:
            raise ValueError("Ray worker config missing local_storage_path")

        base = None
        if isinstance(self.ray_config, dict):
            base = self.ray_config.get("local_storage_path")
        else:
            base = getattr(self.ray_config, "local_storage_path", None)

        if not base:
            raise ValueError("local_storage_path must be specified in RayConfig for worker-local storage")

        return str(base)

    def wipe_local_storage(self) -> Dict[str, Any]:
        """Delete and recreate this worker's local storage directory."""
        base = self._get_local_storage_base()
        hostname = socket.gethostname()
        try:
            path = wipe_worker_local_storage(
                base=base,
                hostname=hostname,
                worker_id=self.worker_id,
            )
        except LocalStorageWipeError as exc:
            logger.error(
                "Worker %d: Failed to wipe local storage under %s (%s)",
                self.worker_id,
                base,
                exc,
            )
            raise

        logger.info("Worker %d: Wiped local storage at %s", self.worker_id, path)
        return {"worker_id": int(self.worker_id), "hostname": hostname, "path": path}
    
    
    
    def initialize_gpu_weights(self, weights: np.ndarray, collective_group: str) -> None:
        """
        Initialize GPU weights for first iteration or reset.
        
        Args:
            weights: Initial weights as numpy array
            collective_group: NCCL collective group name
        """
        with self.device:
            self.gpu_weights = cp.asarray(weights)
            self.collective_group = collective_group
            self.iteration_count = 0
            logger.info(f"Worker {self.worker_id}: Initialized GPU weights with shape {self.gpu_weights.shape}")
    
    
    def get_weights_from_gpu(self) -> np.ndarray:
        """
        Explicitly fetch weights from GPU to CPU.
        Only use when weights are needed on CPU (e.g., checkpointing).
        
        Returns:
            Weights as numpy array on CPU
        """
        with self.device:
            if self.gpu_weights is None:
                raise RuntimeError(f"Worker {self.worker_id}: No GPU weights to fetch")
            
            logger.info(f"Worker {self.worker_id}: Transferring weights from GPU to CPU")
            return self.gpu_weights.get()
    
    
    def get_final_weights(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get final weights and delta weights from GPU.
        Only call when weights are needed on CPU (e.g., end of training).
        
        Returns:
            Tuple of (weights, delta_weights) as numpy arrays on CPU
        """
        from typing import Tuple
        weights_cpu = self.get_weights_from_gpu()
        delta_cpu = self.delta_weights.get() if hasattr(self, 'delta_weights') and self.delta_weights is not None else None
        return weights_cpu, delta_cpu
    
    def get_gpu_weights_only(self) -> Tuple[cp.ndarray, Optional[cp.ndarray]]:
        """
        Get weights that remain on GPU (no CPU transfer).
        Use this during training iterations to avoid unnecessary transfers.
        
        Returns:
            Tuple of (weights, delta_weights) as CuPy arrays on GPU
        """
        with self.device:
            if self.gpu_weights is None:
                raise RuntimeError(f"Worker {self.worker_id}: No GPU weights available")
            
            delta_gpu = self.delta_weights if hasattr(self, 'delta_weights') and self.delta_weights is not None else None
            return self.gpu_weights, delta_gpu
    
    def get_info(self) -> Dict[str, Any]:
        """Get worker information"""
        info = {
            'worker_id': self.worker_id,
            'assigned_gpu': self.assigned_gpu,
            'num_gpus': self.num_gpus,
            'has_data': self.data_loader is not None,
            'data_range': (self.data_start_idx, self.data_end_idx),
            'loader_initialized': self._loader_initialized,
            'has_gpu_weights': self.gpu_weights is not None,
            'iteration_count': self.iteration_count
        }
        
        # Add loader info if initialized
        if self.data_loader:
            info['loader_info'] = self.data_loader.get_info()
        
        return info
    
    def get_memory_status(self) -> Optional[Dict[str, Any]]:
        """Get current memory status"""
        # Simple memory status without complex pipeline
        import psutil
        mem = psutil.virtual_memory()
        return {
            'system_memory': {
                'total_gb': mem.total / (1024**3),
                'available_gb': mem.available / (1024**3),
                'used_gb': mem.used / (1024**3),
                'percent': mem.percent
            }
        }
    
    def use_ram_data_async(
        self,
        data_ref,
        start_idx: int,
        end_idx: int,
        async_config: Optional[Any] = None,
        worker_shard: bool = False,
    ):
        """
        Async version of use_ram_data with progressive chunk loading.
        Processing can begin before all data is loaded.
        
        Args:
            data_ref: Full data array or this worker's pre-sharded RAM object
            start_idx: Global start index for this worker's data
            end_idx: Global end index for this worker's data
            async_config: Async loading configuration
            worker_shard: Whether `data_ref` already contains only this worker's shard
        """
        base_config = async_config or AsyncLoadingConfig()
        config = copy.deepcopy(base_config)
        self.async_loader_config = config
        
        # Calculate worker's data info
        self.data_start_idx = start_idx
        self.data_end_idx = end_idx
        n_samples = end_idx - start_idx
        self._ram_data_is_worker_shard = bool(worker_shard)

        # Get data shape from reference if possible
        import ray
        if hasattr(data_ref, '__getitem__'):
            # Direct array access
            n_features = data_ref.shape[1] if len(data_ref.shape) > 1 else 1
        else:
            # Ray object reference - peek at shape
            sample_data = ray.get(data_ref)
            n_features = sample_data.shape[1] if len(sample_data.shape) > 1 else 1

        load_start_idx = 0 if worker_shard else start_idx
        load_end_idx = n_samples if worker_shard else end_idx
        
        # Store for later use
        self.n_samples = n_samples
        self.n_features = n_features
        
        # Calculate chunks for this worker
        num_chunks = (n_samples + self.loader_chunk_size - 1) // self.loader_chunk_size
        
        logger.info(f"Worker {self.worker_id}: Starting async loading of {num_chunks} chunks "
                   f"({n_samples} samples)")
        
        # Adapt loader configuration based on chunk count so small chunk sizes stay fed
        self._tune_async_loader_config(config, num_chunks)
        self.async_loader_config = config
        
        # Store data reference for loader and defer actual loading until order is known
        self._async_loader_context = {
            'config': config,
            'data_ref': data_ref,
            'start_idx': load_start_idx,
            'end_idx': load_end_idx,
            'num_chunks': num_chunks,
            'initial_chunks': config.get_initial_chunks(num_chunks)
        }
        self.async_loader = None
        self._current_async_order = None

        self.data_ref = data_ref
        self.ram_mode = True  # Flag for RAM mode
        self.async_mode = True  # Flag for async loading
        self._loader_initialized = True  # Async loader ready for selective fetches
        self.data_loader = None  # Ensure sync loader path remains disabled
        
        # Loading will start once order is configured
        initial_chunks = self._async_loader_context['initial_chunks']

        return {
            'worker_id': self.worker_id,
            'status': 'async_loader_configured',
            'total_chunks': num_chunks,
            'initial_chunks': initial_chunks,
            'n_samples': n_samples,
            'n_features': n_features
        }

    def use_node_staged_ram_shard_async(
        self,
        leader_worker,
        worker_idx: int,
        start_idx: int,
        end_idx: int,
        async_config: Optional[Any] = None,
    ):
        """
        Fetch this worker's prepared RAM shard from a node-local staging actor and
        initialize async RAM loading over that shard.
        """
        import ray

        shard = ray.get(leader_worker.get_node_staged_worker_shard.remote(int(worker_idx)))
        return self.use_ram_data_async(
            shard,
            start_idx,
            end_idx,
            async_config=async_config,
            worker_shard=True,
        )

    def use_local_node_staged_ram_shard_async(
        self,
        worker_idx: int,
        start_idx: int,
        end_idx: int,
        async_config: Optional[Any] = None,
    ):
        """Use a shard already staged on this worker without a remote self-call."""
        shard = self.get_node_staged_worker_shard(int(worker_idx))
        return self.use_ram_data_async(
            shard,
            start_idx,
            end_idx,
            async_config=async_config,
            worker_shard=True,
        )
    
    def wait_for_initial_chunks(self, num_chunks: int) -> bool:
        """
        Wait for specified number of chunks to be ready.
        Called by manager before processing begins.
        """
        if hasattr(self, 'async_loader') and self.async_loader is not None:
            success = self.async_loader.wait_for_initial_chunks(num_chunks)
            if success:
                logger.info(f"Worker {self.worker_id}: {num_chunks} initial chunks ready")
            return success
        return True  # Not using async loading

    def _start_async_loader_with_order(self, order: Optional[np.ndarray]) -> None:
        """Start or restart async loader using the provided chunk order."""
        if not self.async_mode or self._async_loader_context is None:
            return

        context = self._async_loader_context
        if self.async_loader is not None:
            self.async_loader.stop()
            self.async_loader = None

        config = context['config']
        total_chunks = context['num_chunks']
        loader_sampling_fraction = self._async_loader_sampling_fraction_override
        if loader_sampling_fraction is None:
            loader_sampling_fraction = self.sampling_fraction
        loader_target_rows = self._async_loader_target_rows_override
        if loader_target_rows == 0:
            loader_target_rows = None
        elif loader_target_rows is None:
            if self.uses_shard_local_random_sampling():
                loader_target_rows = None
            else:
                loader_target_rows = self.chunk_size

        self.async_loader = AsyncChunkLoader(
            worker_id=self.worker_id,
            total_chunks=total_chunks,
            chunk_size=self.loader_chunk_size,
            config=config,
            ram_mode=True,
            sampling_fraction=float(loader_sampling_fraction),
            target_rows=loader_target_rows
        )

        order_list = order.tolist() if order is not None else None
        requested_initial_chunks = int(context['initial_chunks'])
        if order_list is None:
            initial_chunks = min(requested_initial_chunks, int(total_chunks))
        else:
            initial_chunks = min(requested_initial_chunks, int(len(order_list)))
        self.async_loader.start_loading(
            data_ref=context['data_ref'],
            start_idx=context['start_idx'],
            end_idx=context['end_idx'],
            initial_chunks=initial_chunks,
            chunk_order=order_list
        )

        if initial_chunks > 0 and not self.wait_for_initial_chunks(initial_chunks):
            raise RuntimeError(f"Worker {self.worker_id}: Failed to prepare initial async chunks")

        self._current_async_order = tuple(order_list) if order_list is not None else None

    def _ensure_async_loader_order(self, order: np.ndarray, force_restart: bool = False) -> None:
        """Ensure async loader is aligned to the requested chunk order."""
        if not self.async_mode or self._async_loader_context is None:
            return

        order_list = order.tolist()
        order_signature = tuple(int(x) for x in order_list)
        if not force_restart and self.async_loader is not None and self._current_async_order == order_signature:
            return

        self._start_async_loader_with_order(order)
    
    def get_loading_progress(self) -> Dict[str, Any]:
        """Get async loading progress for monitoring"""
        if hasattr(self, 'async_loader') and self.async_loader is not None:
            progress = self.async_loader.get_loading_progress()
            if isinstance(progress, dict):
                progress.setdefault('status', 'loading')
                progress.setdefault('started', True)
            return progress
        elif getattr(self, 'async_mode', False) and getattr(self, '_async_loader_context', None) is not None:
            context = self._async_loader_context
            return {
                'worker_id': self.worker_id,
                'status': 'async_configured_not_started',
                'ready_count': 0,
                'total_chunks': int(context.get('num_chunks', 0) or 0),
                'started': False,
            }
        else:
            return {
                'worker_id': self.worker_id,
                'status': 'not_using_async',
                'ready_count': 0,
                'total_chunks': 0,
                'started': False,
            }
    
    def _tune_async_loader_config(self, config, num_chunks: int) -> None:
        """Adapt async loader settings for workloads with many small chunks."""
        if num_chunks >= 60:
            config.initial_chunks_per_worker = max(config.initial_chunks_per_worker, 3)
            config.max_chunks_in_memory = max(config.max_chunks_in_memory, config.initial_chunks_per_worker + 1)
        if num_chunks >= 120:
            config.initial_chunks_per_worker = max(config.initial_chunks_per_worker, 5)
            config.max_chunks_in_memory = max(config.max_chunks_in_memory, config.initial_chunks_per_worker * 2)
            config.parallel_chunk_loads = max(config.parallel_chunk_loads, 3)
        if num_chunks >= 200:
            config.initial_chunks_per_worker = max(config.initial_chunks_per_worker, 6)
            config.max_chunks_in_memory = max(config.max_chunks_in_memory, config.initial_chunks_per_worker * 2)
            config.parallel_chunk_loads = max(config.parallel_chunk_loads, 4)
            config.chunk_ready_timeout = max(config.chunk_ready_timeout, 60.0)

    def _resolve_worker_profile_config(self, params: Optional[Any] = None):
        processing_config = None
        if params is not None:
            processing_config = getattr(params, "processing_config", None)
        elif hasattr(self, "params"):
            processing_config = getattr(self.params, "processing_config", None)

        profile_config = getattr(processing_config, "worker_profile_config", None)
        if profile_config and getattr(profile_config, "enabled", False):
            return profile_config
        return None

    def _start_worker_profile(self, params: Optional[Any] = None, label: str = "worker") -> None:
        profile_config = self._resolve_worker_profile_config(params)
        if profile_config is None or self._worker_profile_active:
            return

        self._worker_profile_config = profile_config
        self._worker_profile_label = label
        self._worker_profile = cProfile.Profile()
        self._worker_profile.enable()
        self._worker_profile_active = True

    def _write_worker_profile(self) -> None:
        profile_config = self._worker_profile_config
        if profile_config is None or self._worker_profile is None:
            return

        output_dir = profile_config.output_dir or os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        label = self._worker_profile_label or "worker"
        filename = f"{label}_worker{self.worker_id}_profile.txt"
        output_path = os.path.join(output_dir, filename)

        try:
            with open(output_path, "w") as f:
                stream = io.StringIO()
                stats = pstats.Stats(self._worker_profile, stream=stream).sort_stats("cumulative")
                limit = profile_config.max_stats if profile_config.max_stats and profile_config.max_stats > 0 else 50
                stats.print_stats(limit)
                f.write(stream.getvalue())
            logger.info("Worker %d: Wrote profile to %s", self.worker_id, output_path)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Worker %d: Failed to write profile: %s", self.worker_id, exc)

    def _flush_worker_profile(self) -> None:
        if self._worker_profile is None or self._worker_profile_written:
            return

        if self._worker_profile_active:
            self._worker_profile.disable()
            self._worker_profile_active = False

        self._write_worker_profile()
        self._worker_profile_written = True

    def finalize_profile(self) -> None:
        """Flush profiling output without full worker cleanup."""
        self._flush_worker_profile()
    
    def cleanup(self) -> None:
        """Clean up worker resources"""
        self._flush_worker_profile()
        # Stop async loader if active
        if hasattr(self, 'async_loader') and self.async_loader is not None:
            self.async_loader.stop()
            self.async_loader = None
            self._current_async_order = None
            self._async_loader_context = None
        self._node_staged_worker_shards.clear()
        
        logger.info(f"Worker {self.worker_id} cleaning up...")

        self.destroy_collective_group()

        # Shutdown shared CPU pool resources
        self._shutdown_cpu_pool()

        # Clean up loader and explicitly free pinned memory
        if self.data_loader is not None:
            if hasattr(self.data_loader, 'close'):
                try:
                    self.data_loader.close()
                except Exception as exc:
                    logger.warning(f"Worker {self.worker_id}: loader close raised {exc}")
            # Explicitly delete to trigger cleanup
            del self.data_loader
            self.data_loader = None
            self._loader_initialized = False
        
        # Clear GPU memory
        self.local_bmus = None
        self.current_weights = None
        self.influence_matrix = None
        self.gpu_weights = None  # Clear persistent GPU weights
        
        # Force GPU memory cleanup
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        
        # Also free pinned memory pool
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()

        # Remove any local FastArrayStore copies to keep temp storage tidy
        if self.data_path:
            try:
                shutil.rmtree(self.data_path, ignore_errors=True)
                parent_dir = os.path.dirname(self.data_path)
                # Remove parent directory if we created it and it's now empty
                if parent_dir and os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                    shutil.rmtree(parent_dir, ignore_errors=True)
            except Exception as exc:
                logger.warning(f"Worker {self.worker_id}: Failed to remove data path {self.data_path}: {exc}")
            finally:
                self.data_path = None

        logger.info(f"Worker {self.worker_id} cleanup complete")

    # ------------------------------------------------------------------
    # Shared CPU worker utilities
    # ------------------------------------------------------------------

    def get_cpu_pool(self, min_workers: int = 1) -> concurrent.futures.ThreadPoolExecutor:
        if self._cpu_pool is None:
            detected = int(self._detect_assigned_cpu_count() or 1)
            cpu_workers = max(1, detected - 1)
            cpu_workers = max(min_workers, cpu_workers)
            self._cpu_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=cpu_workers,
                thread_name_prefix=f"ray-worker-{self.worker_id}-cpu"
            )
            self._cpu_pool_workers = cpu_workers
        return self._cpu_pool

    def _detect_assigned_cpu_count(self) -> int:
        """
        Best-effort detection of how many CPUs this Ray actor was scheduled with.

        Falls back to OS-level CPU count if Ray context is unavailable.
        """
        from ..resource_limits import detect_assigned_cpu_count

        return detect_assigned_cpu_count()

    def _detect_job_memory_limit_bytes(self) -> Optional[int]:
        """
        Best-effort detection of job-level memory limits (scheduler or cgroup).

        Returns the limit in bytes, or None if no limit could be determined.
        """
        from ..resource_limits import get_job_memory_limit_bytes

        return get_job_memory_limit_bytes()

    def _detect_cgroup_memory_usage_bytes(self) -> Optional[int]:
        """Return current cgroup memory usage in bytes if available."""
        from ..resource_limits import get_cgroup_memory_usage_bytes

        return get_cgroup_memory_usage_bytes()

    def _detect_gpu_workers_on_node(self) -> int:
        """
        Best-effort estimate of how many GPU workers share this node.

        In this pipeline we schedule 1 GPU per worker actor, so we treat the node's
        GPU count as the worker-sharing factor.
        """
        from ..resource_limits import detect_gpu_workers_on_node

        return detect_gpu_workers_on_node()

    def _shutdown_cpu_pool(self) -> None:
        if self._cpu_pool is not None:
            self._cpu_pool.shutdown(wait=True)
            self._cpu_pool = None
            self._cpu_pool_workers = 0
