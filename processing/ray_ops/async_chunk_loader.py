"""
Asynchronous chunk loading coordinator with safety guarantees.
Implements chunk-atomic loading where every chunk is guaranteed
complete before processing begins.
"""

import threading
import queue
import time
import logging
import numpy as np
import cupy as cp
import os
from typing import Dict, Optional, List, Tuple, Any
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ChunkState(Enum):
    """States for chunk loading lifecycle"""
    NOT_STARTED = "not_started"
    LOADING = "loading"
    READY = "ready"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"

@dataclass
class ChunkInfo:
    """Metadata for a loaded chunk"""
    chunk_idx: int
    start_idx: int
    end_idx: int
    size: int
    load_time: float
    retry_count: int = 0
    checksum: Optional[str] = None

class AsyncChunkLoader:
    """
    Thread-safe asynchronous chunk loader with completeness guarantees.
    Ensures no chunk is processed until fully loaded.
    Includes memory semaphore to prevent overflow (addressing review feedback).
    """
    
    def __init__(self, worker_id: int, total_chunks: int, chunk_size: int,
                 config: 'AsyncLoadingConfig', ram_mode: bool = False,
                 sampling_fraction: float = 1.0, target_rows: Optional[int] = None):
        self.worker_id = worker_id
        self.total_chunks = total_chunks
        self.chunk_size = chunk_size
        self.config = config
        self.ram_mode = ram_mode  # Store RAM mode flag
        self.sampling_fraction = max(min(sampling_fraction, 1.0), 1e-6)
        self.target_rows = target_rows
        self.chunk_request_counter = 0
        self.epoch_counter = 0

        # Chunk state tracking
        self.chunk_states = [ChunkState.NOT_STARTED] * total_chunks
        self.chunk_data: Dict[int, np.ndarray] = {}
        self.chunk_info: Dict[int, ChunkInfo] = {}
        self.chunk_offsets: Dict[int, np.ndarray] = {}
        
        # Thread safety
        self.state_lock = threading.Lock()
        self.chunk_events = [threading.Event() for _ in range(total_chunks)]
        
        # Memory management - semaphore to limit chunks in memory
        if ram_mode:
            capacity = max(1, min(total_chunks, config.max_chunks_in_memory))
            self.memory_semaphore = threading.Semaphore(capacity)
            logger.info(
                "AsyncChunkLoader: RAM mode with chunk cap %d (max_chunks_in_memory=%d)",
                capacity,
                config.max_chunks_in_memory,
            )
        else:
            capacity = max(1, config.max_chunks_in_memory)
            self.memory_semaphore = threading.Semaphore(capacity)
            logger.info(
                "AsyncChunkLoader: Disk mode, limiting to %d chunks in memory",
                capacity,
            )
        self.memory_capacity = capacity
        
        # Queues for coordination
        self.ready_queue = queue.Queue()  # Fully loaded chunks
        # Priority queue so on-demand chunk requests jump the line
        self.loading_queue = queue.PriorityQueue()
        self._queue_counter = 0  # Preserve insertion order for same-priority items
        
        # Control flags
        self.stop_loading = threading.Event()
        self.loading_threads: List[threading.Thread] = []
        
        # Cache for dereferenced Ray object (addressing review feedback)
        self.cached_data_array = None
        self.data_array_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'chunks_loaded': 0,
            'chunks_failed': 0,
            'total_load_time': 0.0,
            'total_wait_time': 0.0
        }

        logger.info(
            "AsyncChunkLoader initialized for worker %d: %d chunks of size %d (sampling_fraction=%.4f, target_rows=%s)",
            worker_id,
            total_chunks,
            chunk_size,
            self.sampling_fraction,
            str(self.target_rows) if self.target_rows is not None else "auto",
        )
        self.initial_chunks = 0
    
    def start_loading(self, data_ref: Any, start_idx: int, end_idx: int,
                     initial_chunks: int, chunk_order: Optional[List[int]] = None):
        """
        Start async loading of chunks from data reference.
        
        Args:
            data_ref: Ray object reference or data array
            start_idx: Start index in global data
            end_idx: End index in global data  
            initial_chunks: Number of chunks to load immediately
        """
        self.data_ref = data_ref
        self.global_start = start_idx
        self.global_end = end_idx
        self.data_size = end_idx - start_idx

        if chunk_order is None:
            self.chunk_order = list(range(self.total_chunks))
        else:
            normalized_order = [int(chunk_idx) for chunk_idx in chunk_order]
            if len(normalized_order) != len(set(normalized_order)):
                raise ValueError("chunk_order contains duplicate chunk indices")
            if any(chunk_idx < 0 or chunk_idx >= self.total_chunks for chunk_idx in normalized_order):
                raise ValueError("chunk_order contains invalid chunk indices")
            self.chunk_order = normalized_order

        self.initial_chunks = max(0, min(int(initial_chunks), len(self.chunk_order)))

        # Queue initial chunks for immediate loading BEFORE starting threads
        for chunk_idx in self.chunk_order[:self.initial_chunks]:
            self._enqueue_chunk(chunk_idx, priority=0)

        # Queue remaining chunks
        for chunk_idx in self.chunk_order[self.initial_chunks:]:
            self._enqueue_chunk(chunk_idx, priority=1)
        
        thread_count = self.config.parallel_chunk_loads or 1

        # Start loading threads AFTER queuing work
        for i in range(thread_count):
            thread = threading.Thread(
                target=self._loading_worker,
                args=(initial_chunks,),
                daemon=True
            )
            thread.start()
            self.loading_threads.append(thread)
    
    def _enqueue_chunk(self, chunk_idx: int, priority: int) -> None:
        """Enqueue a chunk while preserving insertion order for same priority."""
        self._queue_counter += 1
        self.loading_queue.put((priority, self._queue_counter, chunk_idx))

    def request_priority_chunks(self, chunk_indices: List[int]) -> None:
        """
        Promote specific chunk indices to priority loading when they are not started yet.
        """
        if not chunk_indices:
            return

        to_enqueue: List[int] = []
        with self.state_lock:
            for chunk_idx in chunk_indices:
                chunk_idx_int = int(chunk_idx)
                if chunk_idx_int < 0 or chunk_idx_int >= self.total_chunks:
                    continue
                if self.chunk_states[chunk_idx_int] != ChunkState.NOT_STARTED:
                    continue
                to_enqueue.append(chunk_idx_int)

        for chunk_idx_int in to_enqueue:
            self._enqueue_chunk(chunk_idx_int, priority=0)

    def set_chunk_order(self, order: List[int]) -> None:
        """Requeue remaining chunks according to the provided order."""
        with self.state_lock:
            new_queue = queue.PriorityQueue()
            counter = 0
            for idx in order:
                state = self.chunk_states[idx]
                if state == ChunkState.NOT_STARTED:
                    counter += 1
                    new_queue.put((1, counter, idx))
            self.loading_queue = new_queue
            self.chunk_order = order
            self._queue_counter = counter

    def _get_data_array(self):
        """Get cached data array, dereferencing Ray object only once"""
        if self.cached_data_array is None:
            with self.data_array_lock:
                if self.cached_data_array is None:  # Double-check
                    import ray
                    if hasattr(self.data_ref, '__getitem__'):
                        # Direct array access
                        self.cached_data_array = self.data_ref
                    else:
                        # Ray object reference - dereference once
                        self.cached_data_array = ray.get(self.data_ref)
        return self.cached_data_array

    def reset_sampling_epoch(self) -> None:
        """Increment epoch counter so chunk sampling produces fresh subsets."""
        with self.state_lock:
            self.chunk_request_counter = 0
            self.epoch_counter += 1

    def _prepare_chunk_output(self, chunk_idx: int, chunk_array: np.ndarray) -> np.ndarray:
        """Apply sampling fraction/target rows to the loaded chunk."""
        if chunk_array is None:
            return None

        base = chunk_array.astype(np.float32, copy=False)
        total_rows = base.shape[0]

        if self.target_rows is not None:
            desired = min(self.target_rows, total_rows)
        elif self.sampling_fraction >= 1.0:
            desired = total_rows
        else:
            desired = max(1, int(round(total_rows * self.sampling_fraction)))

        if desired >= total_rows:
            selected = np.arange(total_rows, dtype=np.int32)
            result = base
        else:
            seed = (
                (self.worker_id << 48)
                ^ (chunk_idx << 24)
                ^ (self.epoch_counter << 8)
                ^ self.chunk_request_counter
            )
            rng = np.random.default_rng(seed)
            indices = rng.choice(total_rows, size=desired, replace=False)
            indices.sort()
            selected = indices.astype(np.int32, copy=False)
            result = base[selected].copy()

        self.chunk_offsets[chunk_idx] = selected
        self.chunk_request_counter += 1
        return result
    
    def _loading_worker(self, initial_chunks: int):
        """Worker thread that loads chunks"""

        while not self.stop_loading.is_set():
            try:
                priority, _, chunk_idx = self.loading_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Skip if already loaded
            with self.state_lock:
                if self.chunk_states[chunk_idx] != ChunkState.NOT_STARTED:
                    continue
                self.chunk_states[chunk_idx] = ChunkState.LOADING

            # Acquire memory semaphore before loading
            self.memory_semaphore.acquire()
            
            # Load the chunk
            success = False
            for retry in range(self.config.retry_failed_chunks):
                try:
                    start_time = time.time()
                    
                    # Calculate chunk boundaries
                    chunk_start = self.global_start + (chunk_idx * self.chunk_size)
                    chunk_end = min(chunk_start + self.chunk_size, self.global_end)
                    actual_size = chunk_end - chunk_start
                    
                    # Slice chunk from cached array without unconditional host copy.
                    data_array = self._get_data_array()
                    chunk_data = np.asarray(data_array[chunk_start:chunk_end], dtype=np.float32)
                    if self.config.require_contiguous and not chunk_data.flags['C_CONTIGUOUS']:
                        chunk_data = np.ascontiguousarray(chunk_data)
                    
                    # Validate chunk
                    if self.config.chunk_validation:
                        self._validate_chunk(chunk_idx, chunk_data, actual_size)
                    
                    load_time = time.time() - start_time
                    
                    # Store chunk and metadata
                    with self.state_lock:
                        self.chunk_data[chunk_idx] = chunk_data
                        self.chunk_info[chunk_idx] = ChunkInfo(
                            chunk_idx=chunk_idx,
                            start_idx=chunk_start - self.global_start,  # Relative to worker
                            end_idx=chunk_end - self.global_start,
                            size=actual_size,
                            load_time=load_time,
                            retry_count=retry
                        )
                        self.chunk_states[chunk_idx] = ChunkState.READY
                        self.stats['chunks_loaded'] += 1
                        self.stats['total_load_time'] += load_time
                    
                    # Signal chunk ready
                    self.chunk_events[chunk_idx].set()
                    self.ready_queue.put(chunk_idx)

                    success = True
                    logger.debug(f"Worker {self.worker_id}: Chunk {chunk_idx} "
                               f"loaded in {load_time:.3f}s")
                    break
                    
                except Exception as e:
                    logger.warning(f"Worker {self.worker_id}: Failed to load "
                                 f"chunk {chunk_idx} (attempt {retry+1}): {e}")
                    if retry == self.config.retry_failed_chunks - 1:
                        # Final failure
                        with self.state_lock:
                            self.chunk_states[chunk_idx] = ChunkState.FAILED
                            self.stats['chunks_failed'] += 1
                        # Release semaphore on failure
                        self.memory_semaphore.release()
                        logger.error(f"Worker {self.worker_id}: "
                                   f"Permanently failed to load chunk {chunk_idx}")
            
            # Small delay between chunks to prevent overwhelming
            if priority != 0 and success:
                time.sleep(0.001)
    
    def _validate_chunk(self, chunk_idx: int, chunk_data: np.ndarray, 
                       expected_size: int):
        """Validate chunk completeness and integrity"""
        # Size check
        if len(chunk_data) != expected_size:
            raise ValueError(f"Chunk {chunk_idx} size mismatch: "
                           f"got {len(chunk_data)}, expected {expected_size}")
        
        # Contiguity check
        if self.config.require_contiguous and not chunk_data.flags['C_CONTIGUOUS']:
            raise ValueError(f"Chunk {chunk_idx} is not contiguous")
        
        # Type check
        if chunk_data.dtype != np.float32:
            raise ValueError(f"Chunk {chunk_idx} has wrong dtype: {chunk_data.dtype}")
        
        # Optional checksum
        if self.config.validate_checksum:
            import hashlib
            checksum = hashlib.md5(chunk_data.tobytes()).hexdigest()
            # Store for later verification if needed
            if hasattr(self, 'expected_checksums'):
                if self.expected_checksums.get(chunk_idx) != checksum:
                    raise ValueError(f"Chunk {chunk_idx} checksum mismatch")
    
    def get_chunk_when_ready(self, chunk_idx: int, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get a chunk, waiting if necessary until it's fully loaded.
        
        Args:
            chunk_idx: Index of chunk to get
            timeout: Maximum time to wait (uses config default if None)
            
        Returns:
            Chunk data if successful, None if failed
        """
        timeout = timeout or self.config.chunk_ready_timeout
        
        # Fast path: already ready or done (in RAM mode, DONE chunks are reusable)
        schedule_priority = False
        chunk_array = None
        with self.state_lock:
            state = self.chunk_states[chunk_idx]
            if state == ChunkState.READY:
                chunk_array = self.chunk_data.get(chunk_idx)
                if chunk_array is not None:
                    self.chunk_states[chunk_idx] = ChunkState.PROCESSING
            elif self.ram_mode and state == ChunkState.DONE:
                # In RAM mode, DONE chunks can be reused
                chunk_array = self.chunk_data.get(chunk_idx)
                if chunk_array is not None:
                    self.chunk_states[chunk_idx] = ChunkState.PROCESSING
            elif state == ChunkState.NOT_STARTED:
                schedule_priority = True
        if chunk_array is not None:
            prepared = self._prepare_chunk_output(chunk_idx, chunk_array)
            with self.state_lock:
                self.chunk_data[chunk_idx] = prepared
            return prepared

        if schedule_priority:
            self._enqueue_chunk(chunk_idx, priority=0)
        
        # Wait for chunk to be ready
        start_time = time.time()
        if not self.chunk_events[chunk_idx].wait(timeout):
            logger.error(f"Worker {self.worker_id}: Timeout waiting for chunk {chunk_idx}")
            return None
        
        wait_time = time.time() - start_time
        self.stats['total_wait_time'] += wait_time
        
        # Get the ready chunk
        with self.state_lock:
            state = self.chunk_states[chunk_idx]
            if state == ChunkState.READY:
                chunk_array = self.chunk_data.get(chunk_idx)
                if chunk_array is not None:
                    self.chunk_states[chunk_idx] = ChunkState.PROCESSING
            elif self.ram_mode and state == ChunkState.DONE:
                chunk_array = self.chunk_data.get(chunk_idx)
                if chunk_array is not None:
                    self.chunk_states[chunk_idx] = ChunkState.PROCESSING

        if chunk_array is not None:
            prepared = self._prepare_chunk_output(chunk_idx, chunk_array)
            with self.state_lock:
                self.chunk_data[chunk_idx] = prepared
            return prepared
                    
        logger.error(f"Worker {self.worker_id}: Chunk {chunk_idx} marked ready but no data (state={self.chunk_states[chunk_idx]})")
        return None

    def wait_for_initial_chunks(self, num_chunks: int, timeout: float = 60.0) -> bool:
        """
        Wait for specified number of chunks to be ready.
        
        Args:
            num_chunks: Number of chunks to wait for
            timeout: Maximum time to wait
            
        Returns:
            True if chunks are ready, False if timeout
        """
        start_time = time.time()
        
        while self.get_ready_count() < num_chunks:
            if time.time() - start_time > timeout:
                logger.error(f"Worker {self.worker_id}: Timeout waiting for "
                           f"{num_chunks} initial chunks")
                return False
            time.sleep(0.1)
        
        logger.info(f"Worker {self.worker_id}: {num_chunks} initial chunks ready")
        return True
    
    def get_ready_count(self) -> int:
        """Get number of chunks currently ready"""
        with self.state_lock:
            return self._get_ready_count_unsafe()
    
    def _get_ready_count_unsafe(self) -> int:
        """Get ready count without acquiring lock (must be called with lock held)"""
        return sum(1 for state in self.chunk_states 
                  if state in [ChunkState.READY, ChunkState.PROCESSING, ChunkState.DONE])

    def mark_chunk_done(self, chunk_idx: int):
        """Mark a chunk as done processing and free memory"""
        with self.state_lock:
            if self.chunk_states[chunk_idx] == ChunkState.PROCESSING:
                self.chunk_states[chunk_idx] = ChunkState.DONE
                if not self.ram_mode and chunk_idx in self.chunk_data:
                    del self.chunk_data[chunk_idx]
                if not self.ram_mode and chunk_idx in self.chunk_offsets:
                    del self.chunk_offsets[chunk_idx]
                self.memory_semaphore.release()
    def get_chunk_view_for_color(self, chunk_idx: int) -> np.ndarray:
        """Return a NumPy view of the chunk for colour gathering."""
        with self.state_lock:
            chunk_data = self.chunk_data.get(chunk_idx)
            if chunk_data is not None:
                return chunk_data

        data_array = self._get_data_array()
        chunk_start = self.global_start + (chunk_idx * self.chunk_size)
        chunk_end = min(chunk_start + self.chunk_size, self.global_end)
        return data_array[chunk_start:chunk_end]

    def get_chunk_selection_offsets(self, chunk_idx: int) -> np.ndarray:
        """Return the original row offsets selected for the processed chunk."""
        offsets = self.chunk_offsets.get(chunk_idx)
        if offsets is not None:
            return offsets
        array = self.chunk_data.get(chunk_idx)
        if isinstance(array, np.ndarray):
            length = array.shape[0]
        else:
            chunk_start = self.global_start + (chunk_idx * self.chunk_size)
            chunk_end = min(chunk_start + self.chunk_size, self.global_end)
            length = chunk_end - chunk_start
        return np.arange(length, dtype=np.int32)

    def clear_color_cache(self) -> None:
        # No-op for compatibility with sync loader interface
        return
    
    def get_next_ready_chunk(self, timeout: float = 1.0) -> Optional[int]:
        """Get the next chunk index that's ready for processing"""
        try:
            return self.ready_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_loading_progress(self) -> Dict[str, Any]:
        """Get current loading progress statistics"""
        with self.state_lock:
            state_counts = {}
            for state in ChunkState:
                state_counts[state.value] = sum(1 for s in self.chunk_states if s == state)
            
            return {
                'worker_id': self.worker_id,
                'total_chunks': self.total_chunks,
                'state_counts': state_counts,
                'ready_count': self._get_ready_count_unsafe(),  # Use unsafe version since we already hold the lock
                'stats': self.stats.copy(),
                'progress_pct': (self.stats['chunks_loaded'] / self.total_chunks * 100)
                               if self.total_chunks > 0 else 0
            }
    
    def stop(self):
        """Stop all loading threads"""
        self.stop_loading.set()
        for thread in self.loading_threads:
            thread.join(timeout=5.0)
        logger.info(f"Worker {self.worker_id}: AsyncChunkLoader stopped")
