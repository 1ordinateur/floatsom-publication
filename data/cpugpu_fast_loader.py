#!/usr/bin/env python3
"""
CPU-GPU Fast Array Loading Utilities with Predictive Prefetching
"""

import numpy as np
import cupy as cp
import json
import os
import logging
import mmap
import psutil
import concurrent.futures
import threading
from typing import Optional, Tuple, Dict, Any, List, Set
from .fast_array_store import FastArrayStore
# chunk_size must now be explicitly provided

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class CPUGPUFastLoader:
    """High-performance loader with predictive prefetching for known chunk order"""
    
    def __init__(self, array_path: str = None, chunk_size: int = None,
                 randomize_chunks: bool = True, random_seed: Optional[int] = None,
                 prefetch_buffer_size: int = 6, worker_id: int = 0,
                 n_workers_per_node: int = 1, ram_data: np.ndarray = None,
                 ram_mode: bool = False, cpu_workers: Optional[int] = None,
                 pinned_buffer_count: int = 1):
        """
        Initialize fast loader with predictive prefetching
        
        Args:
            array_path: Path to FastArrayStore directory (for disk mode)
            chunk_size: Fixed chunk size in samples (default 1M)
            randomize_chunks: If True, randomize the order of chunk indices
            random_seed: Random seed for reproducibility
            prefetch_buffer_size: Number of upcoming chunks to keep in RAM (disk mode only)
            worker_id: Worker ID for memory management
            n_workers_per_node: Total workers on this node for RAM division
            ram_data: Direct numpy array for RAM mode
            ram_mode: If True, use direct RAM access instead of disk
            cpu_workers: Optional CPU worker count for disk prefetch thread pool
            pinned_buffer_count: Number of pinned host buffers to allocate for safe
                concurrent CPU->GPU transfers.
        """
        if chunk_size is None:
            raise ValueError("chunk_size must be explicitly provided")
        self.chunk_size = chunk_size
        self.randomize_chunks = randomize_chunks
        self.worker_id = worker_id
        self.n_workers_per_node = n_workers_per_node
        self.ram_mode = ram_mode
        
        if ram_mode:
            # RAM mode - direct array access
            if ram_data is None:
                raise ValueError("ram_data must be provided in RAM mode")
            
            self.ram_data = ram_data
            self.n_samples = ram_data.shape[0]
            self.n_features = ram_data.shape[1] if len(ram_data.shape) > 1 else 1
            self.dtype = np.float32
            
            # No store or mmap in RAM mode
            self.store = None
            self.mmap_array = None
            
            # No prefetch buffer needed in RAM mode
            self.prefetch_buffer_size = 0
            self.ram_cache = {}  # Not used in RAM mode
        else:
            # Disk mode - use FastArrayStore
            if array_path is None:
                raise ValueError("array_path must be provided in disk mode")
            
            self.array_path = array_path
            
            # Open FastArrayStore
            self.store = FastArrayStore(array_path, mode='r')
            self.mmap_array = self.store.mmap_array
            
            # Store array info
            self.n_samples = self.store.n_samples
            self.n_features = self.store.n_features
            self.dtype = np.float32  # Always use float32
            
            # Setup prefetch buffer based on available RAM
            self.prefetch_buffer_size = self._calculate_prefetch_buffer_size(prefetch_buffer_size)
            self.ram_cache = {}  # chunk_index -> numpy array
        
        # Calculate total chunks
        self.total_chunks = int(np.ceil(self.n_samples / self.chunk_size))
        
        # Create randomized chunk order if requested
        if self.randomize_chunks:
            rng = np.random.RandomState(random_seed)
            self.chunk_order = rng.permutation(self.total_chunks)
        else:
            self.chunk_order = np.arange(self.total_chunks)

        self._update_chunk_position_map()

        # Track current position in chunk order
        self.current_chunk_position = 0

        # Allocate pinned memory buffer for GPU transfers
        self._allocate_pinned_buffers(pinned_buffer_count)

        self._prefetch_executor = None
        self._prefetch_inflight_futures: Dict[int, concurrent.futures.Future] = {}
        self._prefetch_lock = threading.RLock()
        self._prefetch_worker_count = 0
        timeout_env = os.environ.get("FLOATSOM_PREFETCH_FUTURE_TIMEOUT_S")
        try:
            timeout_s = float(timeout_env) if timeout_env is not None else 30.0
        except (TypeError, ValueError):
            timeout_s = 30.0
        self.prefetch_future_timeout_s = max(1.0, timeout_s)
        wait_env = os.environ.get("FLOATSOM_PREFETCH_ON_DEMAND_WAIT_S")
        try:
            wait_s = float(wait_env) if wait_env is not None else 0.01
        except (TypeError, ValueError):
            wait_s = 0.01
        self.prefetch_on_demand_wait_s = max(0.0, wait_s)
        self.log_pending_prefetch_miss = _env_flag(
            "FLOATSOM_LOG_PENDING_PREFETCH_MISS",
            default=False,
        )
        if not self.ram_mode:
            self._prefetch_worker_count = (
                int(cpu_workers)
                if cpu_workers is not None
                else max(1, (os.cpu_count() or 2) - 1)
            )
        if not self.ram_mode and self.prefetch_buffer_size > 0:
            self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._prefetch_worker_count,
                thread_name_prefix=f"cpugpu-prefetch-{worker_id}"
            )

        # Prefetch initial chunks (disk mode only)
        if not self.ram_mode:
            self._prefetch_upcoming_chunks()

        logger.info(f"CPUGPUFastLoader initialized in {'RAM' if ram_mode else 'DISK'} mode:")
        if ram_mode:
            logger.info(f"  Data in RAM: {self.n_samples * self.n_features * 4 / (1024**3):.2f} GB")
        else:
            logger.info(f"  Array path: {array_path}")
            logger.info(f"  Prefetch buffer: {self.prefetch_buffer_size} chunks")
        logger.info(f"  Data shape: ({self.n_samples}, {self.n_features})")
        logger.info(f"  Chunk size: {chunk_size} samples")
        logger.info(f"  Total chunks: {self.total_chunks}")
        logger.info(f"  Pinned memory: {self.pinned_buffer_bytes / (1024**2):.2f} MB")

    def _prefetch_state_lock(self) -> threading.RLock:
        """Return lock guarding prefetch/cache state, creating one for legacy test stubs."""
        lock = getattr(self, "_prefetch_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._prefetch_lock = lock
        return lock
    
    def _calculate_prefetch_buffer_size(self, desired_size: int) -> int:
        """
        Resolve prefetch buffer size using a hard requested depth.

        The requested size is applied directly (unless explicitly set to zero),
        and RAM-fit metrics are logged for observability only.
        """
        requested = int(max(0, desired_size))
        if requested == 0:
            logger.info(
                "Worker %d: Prefetch buffer explicitly disabled (requested=0)",
                self.worker_id,
            )
            return 0

        # Keep RAM-fit computation for diagnostics, but do not clamp.
        available_ram = psutil.virtual_memory().available
        usable_ram = available_ram * 0.8
        ram_per_worker = usable_ram / self.n_workers_per_node
        chunk_bytes = self.chunk_size * self.n_features * 4  # float32 = 4 bytes
        max_chunks = int(ram_per_worker / chunk_bytes) if chunk_bytes > 0 else 0

        if requested > max_chunks:
            logger.warning(
                "Worker %d: Prefetch buffer hard-set to %d chunks (estimated RAM-fit %d)",
                self.worker_id,
                requested,
                max_chunks,
            )
        else:
            logger.info(
                "Worker %d: Prefetch buffer hard-set to %d chunks",
                self.worker_id,
                requested,
            )

        return requested

    def set_prefetch_buffer_size(self, desired_size: int) -> int:
        """Resize RAM prefetch depth using hard requested size semantics."""
        if not isinstance(desired_size, int):
            raise ValueError("desired_size must be an integer")
        if desired_size < 0:
            raise ValueError("desired_size must be >= 0")

        if self.ram_mode:
            self.prefetch_buffer_size = 0
            return 0

        lock = self._prefetch_state_lock()
        with lock:
            if not hasattr(self, "ram_cache") or self.ram_cache is None:
                self.ram_cache = {}
            if (
                not hasattr(self, "_prefetch_inflight_futures")
                or self._prefetch_inflight_futures is None
            ):
                self._prefetch_inflight_futures = {}

        actual_size = self._calculate_prefetch_buffer_size(desired_size)
        self.prefetch_buffer_size = int(actual_size)
        if self.prefetch_buffer_size <= 0:
            with lock:
                futures_to_cancel = list(self._prefetch_inflight_futures.values())
                self._prefetch_inflight_futures.clear()
                self.ram_cache.clear()
                executor = self._prefetch_executor
                self._prefetch_executor = None
            for future in futures_to_cancel:
                future.cancel()
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)
            return self.prefetch_buffer_size

        with lock:
            if self._prefetch_executor is None:
                worker_count = int(getattr(self, "_prefetch_worker_count", 0) or 0)
                if worker_count <= 0:
                    worker_count = max(1, (os.cpu_count() or 2) - 1)
                    self._prefetch_worker_count = worker_count
                self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix=f"cpugpu-prefetch-{self.worker_id}"
                )

        self._prefetch_upcoming_chunks()
        return self.prefetch_buffer_size
    
    def _allocate_pinned_buffers(self, count: int) -> None:
        """Allocate pinned host buffers for efficient CPU-GPU transfers."""
        if not isinstance(count, int):
            raise ValueError("pinned_buffer_count must be an integer")
        if count <= 0:
            raise ValueError("pinned_buffer_count must be positive")

        buffer_elements = self.chunk_size * self.n_features
        self.pinned_buffer_bytes = buffer_elements * 4  # float32 = 4 bytes

        self.pinned_memories = []
        self.pinned_buffer_flats = []
        for _ in range(count):
            pinned_memory = cp.cuda.alloc_pinned_memory(self.pinned_buffer_bytes)
            pinned_buffer_flat = np.frombuffer(
                pinned_memory,
                dtype=np.float32,
                count=buffer_elements
            )
            self.pinned_memories.append(pinned_memory)
            self.pinned_buffer_flats.append(pinned_buffer_flat)

    def ensure_pinned_buffers(self, count: int) -> None:
        """Ensure at least `count` pinned host buffers are available."""
        if not isinstance(count, int):
            raise ValueError("count must be an integer")
        if count <= 0:
            raise ValueError("count must be positive")

        current = len(getattr(self, "pinned_buffer_flats", []) or [])
        if current >= count:
            return

        buffer_elements = self.chunk_size * self.n_features
        for _ in range(count - current):
            pinned_memory = cp.cuda.alloc_pinned_memory(self.pinned_buffer_bytes)
            pinned_buffer_flat = np.frombuffer(
                pinned_memory,
                dtype=np.float32,
                count=buffer_elements
            )
            self.pinned_memories.append(pinned_memory)
            self.pinned_buffer_flats.append(pinned_buffer_flat)
    
    def _prefetch_upcoming_chunks(self):
        """Preload next N chunks based on known access order (disk mode only)"""
        if self.ram_mode:
            return  # No prefetching needed in RAM mode
        if self.prefetch_buffer_size <= 0 or self.total_chunks == 0:
            return  # Prefetching disabled by memory guard or no data
        if self._prefetch_executor is None:
            return  # Executor not available

        lock = self._prefetch_state_lock()
        with lock:
            current_pos = self.current_chunk_position

            # Calculate which chunks to keep in RAM
            indices_to_keep = set()
            for i in range(self.prefetch_buffer_size):
                next_pos = (current_pos + i) % self.total_chunks
                next_chunk_idx = int(self.chunk_order[next_pos])
                indices_to_keep.add(next_chunk_idx)

            # Evict old chunks not in upcoming set
            self.ram_cache = {
                k: v for k, v in self.ram_cache.items()
                if k in indices_to_keep
            }
        self._collect_completed_prefetch_futures(indices_to_keep)

        with lock:
            executor = self._prefetch_executor
            if executor is None:
                return

            # Prefetch missing chunks
            for i in range(self.prefetch_buffer_size):
                next_pos = (current_pos + i) % self.total_chunks
                next_chunk_idx = int(self.chunk_order[next_pos])

                if (
                    next_chunk_idx in self.ram_cache
                    or next_chunk_idx in self._prefetch_inflight_futures
                ):
                    continue

                future = executor.submit(
                    self._load_chunk_to_ram, next_chunk_idx
                )
                self._prefetch_inflight_futures[next_chunk_idx] = future

    def _try_resolve_prefetched_chunk(
        self,
        chunk_index: int,
        wait_timeout_s: float = 0.0,
    ) -> bool:
        """Try to materialize one prefetched chunk from inflight futures into RAM cache."""
        if self.ram_mode:
            return False

        chunk_idx = int(chunk_index)
        lock = self._prefetch_state_lock()
        with lock:
            if chunk_idx in self.ram_cache:
                return True
            future = self._prefetch_inflight_futures.get(chunk_idx)
        if future is None:
            return False

        timeout_s = max(0.0, float(wait_timeout_s or 0.0))
        if not future.done() and timeout_s > 0.0:
            try:
                concurrent.futures.wait(
                    [future],
                    timeout=timeout_s,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
            except Exception:
                return False
        if not future.done():
            return False

        with lock:
            active = self._prefetch_inflight_futures.get(chunk_idx)
            if active is None:
                return chunk_idx in self.ram_cache
            if active is not future:
                future = active
                if not future.done():
                    return False
            self._prefetch_inflight_futures.pop(chunk_idx, None)
        if future.cancelled():
            return False
        try:
            loaded_idx, chunk_data = future.result()
        except Exception as exc:
            logger.warning(
                "Worker %d: Prefetch task failed for chunk %d: %s",
                self.worker_id,
                chunk_idx,
                exc,
            )
            return False

        loaded_idx = int(loaded_idx)
        with lock:
            self.ram_cache[loaded_idx] = chunk_data
        return loaded_idx == chunk_idx

    def _try_get_chunk_from_cache_or_inflight(
        self,
        chunk_index: int,
        wait_timeout_s: float = 0.0,
    ) -> Optional[np.ndarray]:
        """
        Return chunk data from RAM cache/prefetch futures when available.

        This helper avoids mmap fallback when the requested chunk has just been
        materialized by background prefetch.
        """
        if self.ram_mode:
            return None

        chunk_idx = int(chunk_index)
        lock = self._prefetch_state_lock()
        with lock:
            if chunk_idx in self.ram_cache:
                return self.ram_cache[chunk_idx]

        if self._try_resolve_prefetched_chunk(chunk_idx, wait_timeout_s=wait_timeout_s):
            with lock:
                return self.ram_cache.get(chunk_idx)

        # Final non-blocking harvest so we do not mmap if cache just recovered.
        self._collect_completed_prefetch_futures()
        with lock:
            if chunk_idx in self.ram_cache:
                return self.ram_cache[chunk_idx]

        return None

    def _collect_completed_prefetch_futures(
        self,
        indices_to_keep: Optional[Set[int]] = None,
    ) -> None:
        """Move completed prefetch results into RAM cache and clear finished futures."""
        keep_set = None
        if indices_to_keep is not None:
            keep_set = {int(idx) for idx in indices_to_keep}

        lock = self._prefetch_state_lock()
        with lock:
            if not self._prefetch_inflight_futures:
                return
            inflight_items = list(self._prefetch_inflight_futures.items())

        completed_items = [
            (int(chunk_idx), future)
            for chunk_idx, future in inflight_items
            if future.done()
        ]
        for chunk_idx, future in completed_items:
            with lock:
                active = self._prefetch_inflight_futures.get(chunk_idx)
                if active is None or active is not future:
                    continue
                self._prefetch_inflight_futures.pop(chunk_idx, None)
            if future.cancelled():
                continue
            try:
                loaded_idx, chunk_data = future.result()
            except Exception as exc:
                logger.warning(
                    "Worker %d: Prefetch task failed for chunk %d: %s",
                    self.worker_id,
                    chunk_idx,
                    exc,
                )
                continue

            loaded_idx = int(loaded_idx)
            if keep_set is not None and loaded_idx not in keep_set:
                continue
            with lock:
                self.ram_cache[loaded_idx] = chunk_data
            logger.debug(f"Prefetched chunk {loaded_idx} into RAM (thread pool)")

    def _print_prepositioned_cache_miss(
        self,
        caller: str,
        requested_chunk_index: int,
        actual_chunk_index: int,
    ) -> None:
        """Emit a direct stdout diagnostic when prefetched cache lookup misses."""
        if self.ram_mode:
            return

        lock = self._prefetch_state_lock()
        with lock:
            cache_size = len(self.ram_cache)
            inflight = int(actual_chunk_index in self._prefetch_inflight_futures)
            current_pos = int(self.current_chunk_position)
            expected_chunk = -1
            if self.total_chunks > 0:
                expected_chunk = int(self.chunk_order[current_pos % self.total_chunks])

            upcoming: List[int] = []
            if self.total_chunks > 0:
                depth = min(max(0, int(self.prefetch_buffer_size)), self.total_chunks)
                for offset in range(depth):
                    pos = (current_pos + offset) % self.total_chunks
                    upcoming.append(int(self.chunk_order[pos]))
        if inflight and not bool(getattr(self, "log_pending_prefetch_miss", False)):
            return

        print(
            f"[CPUGPUFastLoader {self.worker_id}] prepositioned-cache miss "
            f"caller={caller} requested_chunk={int(requested_chunk_index)} "
            f"actual_chunk={int(actual_chunk_index)} expected_chunk={expected_chunk} "
            f"inflight={inflight} cache_size={cache_size} "
            f"prefetch_window={upcoming}",
            flush=True,
        )
    
    def get_chunk(
        self,
        chunk_index: int,
        use_randomized_order: bool = False,
        *,
        advance_cursor: bool,
    ) -> Tuple[cp.ndarray, Dict[str, Any]]:
        """
        Get a chunk of data on GPU using pinned memory transfer

        Args:
            chunk_index: Requested chunk index.
            use_randomized_order: Whether to map through randomized traversal order.
            advance_cursor: Explicit consume intent for predictive prefetch cursor.
        
        Returns:
            Tuple of (gpu_chunk, chunk_info)
        """
        # Map index through randomized order if requested
        if use_randomized_order and self.randomize_chunks:
            actual_chunk_index = self.chunk_order[chunk_index]
        else:
            actual_chunk_index = chunk_index
        
        # Calculate sample indices
        start_sample = actual_chunk_index * self.chunk_size
        end_sample = min(start_sample + self.chunk_size, self.n_samples)
        actual_samples = end_sample - start_sample
        
        # Get data from appropriate source
        if self.ram_mode:
            # Direct slice from RAM array
            chunk_data = self.ram_data[start_sample:end_sample]
            source = 'ram_direct'
        else:
            chunk_data = self._try_get_chunk_from_cache_or_inflight(
                actual_chunk_index,
                wait_timeout_s=float(getattr(self, "prefetch_on_demand_wait_s", 0.0) or 0.0),
            )
            if chunk_data is not None:
                source = 'ram_cache'
            else:
            # Fall back to memory-mapped read
                chunk_data = np.asarray(self.mmap_array[start_sample:end_sample], dtype=np.float32)
                source = 'mmap'
                self._print_prepositioned_cache_miss(
                    caller="get_chunk",
                    requested_chunk_index=int(chunk_index),
                    actual_chunk_index=int(actual_chunk_index),
                )
                logger.debug(f"Chunk {actual_chunk_index} not in RAM cache, reading from mmap")
        
        # Copy to pinned memory
        pinned_buffer_flat = self.pinned_buffer_flats[0]
        if self.n_features > 1:
            actual_elements = actual_samples * self.n_features
            pinned_view = pinned_buffer_flat[:actual_elements].reshape(actual_samples, self.n_features)
        else:
            pinned_view = pinned_buffer_flat[:actual_samples]
        
        pinned_view[:] = chunk_data
        
        # Transfer to GPU
        gpu_data = cp.asarray(pinned_view, dtype=cp.float32)
        
        # Create chunk info
        chunk_info = {
            'chunk_index': actual_chunk_index,
            'global_start': start_sample,
            'global_end': end_sample,
            'local_size': actual_samples,
            'shape': gpu_data.shape,
            'source': source
        }
        
        if bool(advance_cursor):
            self._advance_position_after_fetch(actual_chunk_index)

        return gpu_data, chunk_info
    
    def get_next_chunk(self) -> Tuple[cp.ndarray, Dict[str, Any]]:
        """Get the next chunk in the randomized order with prefetching"""
        lock = self._prefetch_state_lock()
        with lock:
            if self.current_chunk_position >= self.total_chunks:
                # Reset for next epoch
                logger.info("All chunks processed, resetting to beginning")
                self.current_chunk_position = 0
                
                if self.randomize_chunks:
                    rng = np.random.RandomState()
                    self.chunk_order = rng.permutation(self.total_chunks)
                    logger.debug("Re-shuffled chunk order for new epoch")
                    self._update_chunk_position_map()
                else:
                    # Ensure position map stays aligned for sequential traversal
                    self._update_chunk_position_map()
            
            # Get current chunk position before advancing
            position = self.current_chunk_position
            actual_chunk_index = int(self.chunk_order[position])
        gpu_data, chunk_info = self.get_chunk(
            actual_chunk_index,
            advance_cursor=True,
        )

        # Ensure internal cursor reflects sequential traversal
        with lock:
            self.current_chunk_position = position + 1 if self.total_chunks else 0

        # Add position info
        chunk_info['position_in_order'] = position
        chunk_info['total_positions'] = self.total_chunks

        return gpu_data, chunk_info
    
    def reset_chunk_order(self, random_seed: Optional[int] = None):
        """Reset and optionally re-shuffle the chunk order"""
        if self.randomize_chunks:
            rng = np.random.RandomState(random_seed)
            new_order = rng.permutation(self.total_chunks)
        else:
            new_order = np.arange(self.total_chunks, dtype=np.int32)
        self.set_chunk_order(new_order)

    def set_chunk_order(self, chunk_order: np.ndarray) -> None:
        """Apply an explicit canonical chunk traversal order for prefetching."""
        order = np.asarray(chunk_order, dtype=np.int64)
        if order.ndim != 1:
            raise ValueError("chunk_order must be 1-dimensional")
        if int(order.size) != int(self.total_chunks):
            raise ValueError(
                f"chunk_order length {int(order.size)} must equal total_chunks {int(self.total_chunks)}"
            )

        if order.size > 0:
            if int(order.min()) < 0 or int(order.max()) >= int(self.total_chunks):
                raise ValueError("chunk_order entries must be within [0, total_chunks)")
            if np.unique(order).size != int(self.total_chunks):
                raise ValueError("chunk_order must be a permutation of all chunk indices")

        lock = self._prefetch_state_lock()
        with lock:
            self.chunk_order = order.astype(np.int32, copy=False)
            self.current_chunk_position = 0
            self._update_chunk_position_map()
        self._prefetch_upcoming_chunks()

    # ------------------------------------------------------------------
    # Colour processing helpers
    # ------------------------------------------------------------------

    def shuffle_chunks(self, seed: Optional[int] = None) -> np.ndarray:
        """Return a shuffled copy of chunk indices using provided seed."""
        order = np.arange(self.total_chunks)
        if seed is not None:
            rng = np.random.RandomState(seed)
            rng.shuffle(order)
        elif self.randomize_chunks:
            rng = np.random.RandomState()
            rng.shuffle(order)
        return order

    def partition_chunks(self, n_partitions: int, seed: Optional[int] = None) -> List[np.ndarray]:
        """Split chunk indices into roughly equal partitions for colour rounds."""
        if n_partitions <= 0:
            raise ValueError("n_partitions must be positive")
        order = self.shuffle_chunks(seed)
        if len(order) == 0:
            return [np.array([], dtype=np.int32) for _ in range(n_partitions)]
        partitions = np.array_split(order, n_partitions)
        return [p.astype(np.int32, copy=False) for p in partitions]

    def get_chunk_bounds(self, chunk_index: int) -> Tuple[int, int]:
        """Return (start, end) sample indices for the chunk."""
        start_sample = chunk_index * self.chunk_size
        end_sample = min(start_sample + self.chunk_size, self.n_samples)
        return start_sample, end_sample

    def get_chunk_view_for_color(self, chunk_index: int) -> np.ndarray:
        """Return a host view of the chunk for colour gather without persistent caching."""
        start, end = self.get_chunk_bounds(chunk_index)

        if self.ram_mode:
            return self.ram_data[start:end]

        chunk_idx = int(chunk_index)
        chunk_data = self._try_get_chunk_from_cache_or_inflight(
            chunk_idx,
            wait_timeout_s=float(getattr(self, "prefetch_on_demand_wait_s", 0.0) or 0.0),
        )
        if chunk_data is None:
            lock = self._prefetch_state_lock()
            with lock:
                if chunk_idx in self.ram_cache:
                    chunk_data = self.ram_cache[chunk_idx]
        if chunk_data is not None:
            self._advance_position_after_fetch(chunk_idx)
            return chunk_data

        view = self.mmap_array[start:end]
        if view.dtype != np.float32:
            view = view.astype(np.float32, copy=False)
        self._advance_position_after_fetch(chunk_idx)
        return view

    def get_chunk_selection_offsets(self, chunk_index: int) -> np.ndarray:
        """Return sequential offsets for the chunk (used by colour sampling)."""
        start, end = self.get_chunk_bounds(chunk_index)
        length = end - start
        return np.arange(length, dtype=np.int32)

    def clear_color_cache(self) -> None:
        """Placeholder to mirror async loader interface."""
        return

    def _load_chunk_to_ram(self, chunk_idx: int) -> Tuple[int, np.ndarray]:
        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, self.n_samples)
        chunk_data = np.asarray(self.mmap_array[start:end], dtype=np.float32).copy()
        return chunk_idx, chunk_data

    def close(self):
        lock = self._prefetch_state_lock()
        with lock:
            futures_to_cancel = list(getattr(self, "_prefetch_inflight_futures", {}).values())
            if hasattr(self, '_prefetch_inflight_futures'):
                self._prefetch_inflight_futures.clear()
            executor = getattr(self, "_prefetch_executor", None)
            self._prefetch_executor = None
        for future in futures_to_cancel:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=True)
    
    def transfer_to_gpu_buffer(
        self,
        chunk_index: int,
        target_buffer: cp.ndarray,
        stream: Optional[cp.cuda.Stream] = None,
        use_randomized_order: bool = False,
        pinned_buffer_index: int = 0,
        *,
        advance_cursor: bool,
    ) -> Dict[str, Any]:
        """
        Transfer chunk directly to pre-allocated GPU buffer.
        
        Args:
            chunk_index: Index of chunk to transfer
            target_buffer: Pre-allocated GPU buffer to transfer to
            stream: Optional CUDA stream for async transfer
            use_randomized_order: Whether to use randomized chunk order
            pinned_buffer_index: Which pinned host buffer to use (enables multiple
                in-flight transfers without overwriting host staging memory).
            advance_cursor: If True, advance predictive prefetch cursor for this
                chunk fetch. Set False for speculative prefetch transfers that are
                not yet consumed.
            
        Returns:
            Dict with chunk metadata including actual size and padding info
        """
        # Map index through randomized order if requested
        if use_randomized_order and self.randomize_chunks:
            actual_chunk_index = self.chunk_order[chunk_index]
        else:
            actual_chunk_index = chunk_index
        
        # Calculate sample indices
        start_sample = actual_chunk_index * self.chunk_size
        end_sample = min(start_sample + self.chunk_size, self.n_samples)
        actual_samples = end_sample - start_sample
        
        # Get data from appropriate source
        if self.ram_mode:
            # Direct slice from RAM array
            chunk_data = self.ram_data[start_sample:end_sample]
            source = 'ram_direct'
        else:
            chunk_data = self._try_get_chunk_from_cache_or_inflight(
                actual_chunk_index,
                wait_timeout_s=float(getattr(self, "prefetch_on_demand_wait_s", 0.0) or 0.0),
            )
            if chunk_data is not None:
                source = 'ram_cache'
            else:
            # Fall back to memory-mapped read
                chunk_data = np.asarray(self.mmap_array[start_sample:end_sample], dtype=np.float32)
                source = 'mmap'
                if bool(advance_cursor):
                    # Only report misses for consumed chunks; speculative prefetch
                    # loads are expected to miss occasionally and add noise.
                    self._print_prepositioned_cache_miss(
                        caller="transfer_to_gpu_buffer",
                        requested_chunk_index=int(chunk_index),
                        actual_chunk_index=int(actual_chunk_index),
                    )
                logger.debug(f"Chunk {actual_chunk_index} not in RAM cache, reading from mmap")
        
        if pinned_buffer_index is None:
            pinned_buffer_index = 0
        self.ensure_pinned_buffers(int(pinned_buffer_index) + 1)
        pinned_buffer_flat = self.pinned_buffer_flats[int(pinned_buffer_index)]

        # Copy to pinned memory
        if self.n_features > 1:
            actual_elements = actual_samples * self.n_features
            pinned_view = pinned_buffer_flat[:actual_elements].reshape(actual_samples, self.n_features)
        else:
            pinned_view = pinned_buffer_flat[:actual_samples]
        
        pinned_view[:] = chunk_data
        
        # Transfer to target GPU buffer
        # Handle potential padding in target buffer
        padded = False
        if actual_samples < self.chunk_size:
            # Last chunk might be smaller - zero out padding
            padded = True
            target_slice = target_buffer[:actual_samples]
            if stream is not None:
                target_slice.set(pinned_view, stream=stream)
                with stream:
                    target_buffer[actual_samples:].fill(0)
            else:
                target_slice.set(pinned_view)
                target_buffer[actual_samples:].fill(0)
        else:
            # Full chunk - direct transfer
            if stream is not None:
                target_buffer.set(pinned_view, stream=stream)
            else:
                target_buffer.set(pinned_view)
        
        # Create chunk info
        chunk_info = {
            'chunk_index': actual_chunk_index,
            'global_start': start_sample,
            'global_end': end_sample,
            'local_size': actual_samples,
            'shape': (actual_samples, self.n_features) if self.n_features > 1 else (actual_samples,),
            'source': source,
            'padded': padded,
            'buffer_shape': target_buffer.shape
        }
        
        if bool(advance_cursor):
            self._advance_position_after_fetch(actual_chunk_index)

        return chunk_info

    def mark_chunk_consumed(self, chunk_index: int) -> None:
        """Advance traversal cursor for a chunk that has become current/consumed."""
        self._advance_position_after_fetch(int(chunk_index))
    
    def get_info(self) -> Dict[str, Any]:
        """Get loader information"""
        pinned_buffer_count = len(getattr(self, "pinned_memories", []) or [])
        if pinned_buffer_count <= 0:
            pinned_buffer_count = 1

        lock = self._prefetch_state_lock()
        with lock:
            current_position = self.current_chunk_position
            ram_cache_chunks = len(self.ram_cache) if not self.ram_mode else 0

        info = {
            'mode': 'RAM' if self.ram_mode else 'DISK',
            'chunk_size': self.chunk_size,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'total_chunks': self.total_chunks,
            'dtype': 'float32',
            'pinned_buffer_mb': (self.pinned_buffer_bytes * pinned_buffer_count) / (1024**2),
            'pinned_buffer_count': pinned_buffer_count,
            'randomize_chunks': self.randomize_chunks,
            'current_position': current_position
        }
        
        if self.ram_mode:
            info['data_size_gb'] = self.n_samples * self.n_features * 4 / (1024**3)
        else:
            info['array_path'] = self.array_path
            info['prefetch_buffer_size'] = self.prefetch_buffer_size
            info['ram_cache_chunks'] = ram_cache_chunks

        return info

    def _update_chunk_position_map(self):
        """Track each chunk's position within the current traversal order."""
        self.chunk_position_map = {
            int(chunk_idx): pos for pos, chunk_idx in enumerate(self.chunk_order)
        }

    def _advance_position_after_fetch(self, actual_chunk_index: int) -> None:
        """Advance internal cursor and prefetch future chunks after direct access."""
        if self.ram_mode or self.prefetch_buffer_size <= 0 or self.total_chunks == 0:
            return

        lock = self._prefetch_state_lock()
        with lock:
            position = self.chunk_position_map.get(int(actual_chunk_index))
            if position is None:
                return

            # Advance to the next position in traversal order (may equal total_chunks)
            self.current_chunk_position = position + 1
        self._prefetch_upcoming_chunks()
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'store') and self.store is not None:
            self.store.close()
        if hasattr(self, 'ram_cache'):
            self.ram_cache.clear()
        if hasattr(self, 'ram_data'):
            # In RAM mode, data will be freed when ref count drops
            self.ram_data = None
