"""
Ray batch worker for batch processing mode
Implements batch-specific SOM training logic for distributed multi-GPU training
"""

# Configure numexpr before numpy import to avoid thread limit errors
from .numexpr_config import configure_numexpr_threads

configure_numexpr_threads()

import ray
from ray.util import collective
from ray.util.collective.types import ReduceOp
import cupy as cp
import numpy as np
import math
import socket
import time
from typing import Dict, Any, Tuple, Optional
import logging

from .ray_pipeline_base_worker import RayPipelineBaseWorker
from ...utils import (
    find_bmus,
    compute_weight_updates,
    compute_weight_updates_sparse,
    apply_normalization,
    apply_momentum,
    _ensure_sparse_influence_csr,
)

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class RayBatchWorker(RayPipelineBaseWorker):
    """
    Batch-specific worker that extends base worker
    Handles batch processing with NCCL synchronization
    """
    
    def _process_single_chunk(self, chunk_index: int, params: Dict, 
                            topology_data: Dict) -> Tuple[cp.ndarray, cp.ndarray, int]:
        """
        Process a single chunk of data.
        
        Args:
            chunk_index: Index of chunk to process
            params: Training parameters
            topology_data: Topology information
            
        Returns:
            Tuple of (accumulated_updates, accumulated_influence, samples_processed)
        """
        # Load chunk
        local_samples = self.load_chunk(chunk_index)
        processing_config = getattr(params, "processing_config", None)
        metric = getattr(processing_config, "distance_metric", "euclidean") if processing_config else "euclidean"
        metric_kwargs = getattr(processing_config, "distance_metric_params", {}) or {}
        
        # Find BMUs for local samples
        bmus = find_bmus(
            local_samples, self.gpu_weights, return_distances=False,
            metric=metric, **metric_kwargs
        )
        
        # Get raw accumulated updates and influence
        use_sparse = bool(getattr(processing_config, "use_sparse_influence", False))
        if use_sparse:
            accumulated_updates, accumulated_influence = compute_weight_updates_sparse(
                batch=local_samples,
                bmus=bmus,
                weights=self.gpu_weights,
                influence_matrix=self.influence_matrix,
                learning_rate=params.current_learning_rate,
                apply_lr_in_update=(getattr(processing_config, "normalization", None) != "xpysom"),
                verbose=False
            )
        else:
            accumulated_updates, accumulated_influence = compute_weight_updates(
                batch=local_samples,
                bmus=bmus,
                weights=self.gpu_weights,
                influence_matrix=self.influence_matrix,
                learning_rate=params.current_learning_rate,
                apply_lr_in_update=(getattr(processing_config, "normalization", None) != "xpysom"),
                chunk_size=self.chunk_size or 1000,
                verbose=False
            )
        
        return accumulated_updates, accumulated_influence, len(local_samples)
    
    def _process_chunk_range(self, start_idx: int, end_idx: int, params: Dict,
                           topology_data: Dict,
                           selected_positions: Optional[np.ndarray] = None) -> Tuple[cp.ndarray, cp.ndarray, int]:
        """
        Process a range of samples aligned to chunk boundaries.
        
        Args:
            start_idx: Start index of range
            end_idx: End index of range
            params: Training parameters
            topology_data: Topology information
            
        Returns:
            Tuple of (accumulated_updates, accumulated_influence, samples_processed)
        """
        # Load selected range from FastArrayStore
        local_samples = self.load_selective(start_idx, end_idx)
        if selected_positions is not None:
            selected_positions = np.asarray(selected_positions, dtype=np.int32)
            if selected_positions.size == 0:
                del local_samples
                return (
                    cp.zeros_like(self.gpu_weights),
                    cp.zeros(self.gpu_weights.shape[0], dtype=self.gpu_weights.dtype),
                    0,
                )
            valid_positions = selected_positions[
                (selected_positions >= 0) & (selected_positions < int(local_samples.shape[0]))
            ]
            if valid_positions.size == 0:
                del local_samples
                return (
                    cp.zeros_like(self.gpu_weights),
                    cp.zeros(self.gpu_weights.shape[0], dtype=self.gpu_weights.dtype),
                    0,
                )
            if valid_positions.size < int(local_samples.shape[0]):
                local_samples = local_samples[cp.asarray(valid_positions, dtype=cp.int32)]

        if int(local_samples.shape[0]) == 0:
            del local_samples
            return (
                cp.zeros_like(self.gpu_weights),
                cp.zeros(self.gpu_weights.shape[0], dtype=self.gpu_weights.dtype),
                0,
            )
        processing_config = getattr(params, "processing_config", None)
        metric = getattr(processing_config, "distance_metric", "euclidean") if processing_config else "euclidean"
        metric_kwargs = getattr(processing_config, "distance_metric_params", {}) or {}
        
        # Find BMUs for this chunk
        bmus = find_bmus(
            local_samples, self.gpu_weights, return_distances=False,
            metric=metric, **metric_kwargs
        )
        
        # Get raw accumulated updates and influence for this chunk
        use_sparse = bool(getattr(processing_config, "use_sparse_influence", False))
        if use_sparse:
            chunk_updates, chunk_influence = compute_weight_updates_sparse(
                batch=local_samples,
                bmus=bmus,
                weights=self.gpu_weights,
                influence_matrix=self.influence_matrix,
                learning_rate=params.current_learning_rate,
                apply_lr_in_update=(getattr(processing_config, "normalization", None) != "xpysom"),
                verbose=False
            )
        else:
            chunk_updates, chunk_influence = compute_weight_updates(
                batch=local_samples,
                bmus=bmus,
                weights=self.gpu_weights,
                influence_matrix=self.influence_matrix,
                learning_rate=params.current_learning_rate,
                apply_lr_in_update=(getattr(processing_config, "normalization", None) != "xpysom"),
                chunk_size=self.chunk_size or 1000,
                verbose=False
            )
        
        # Store length before deletion
        samples_count = len(local_samples)
        
        # Free chunk memory
        del local_samples, bmus
        cp.cuda.get_current_stream().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        
        return chunk_updates, chunk_influence, samples_count
    
    def _process_all_chunks(
        self,
        params: Dict,
        topology_data: Dict,
        *,
        chunk_indices: Optional[np.ndarray] = None,
        timings: Optional[Dict[str, Any]] = None,
    ) -> Tuple[cp.ndarray, cp.ndarray, int]:
        """
        Process all chunks with super-chunk accumulation to reduce memory overhead.
        
        Args:
            params: Training parameters
            topology_data: Topology information
            
        Returns:
            Tuple of (total_accumulated_updates, total_accumulated_influence, total_samples_processed)
        """
        # Initialize accumulated updates
        total_accumulated_updates = cp.zeros_like(self.gpu_weights)
        total_accumulated_influence = cp.zeros(self.gpu_weights.shape[0])
        total_samples_processed = 0
        
        total_available_chunks = self.get_num_chunks()
        if chunk_indices is None:
            chunk_order = np.arange(total_available_chunks, dtype=np.int32)
        else:
            chunk_order = np.asarray(chunk_indices, dtype=np.int32)
        num_chunks = int(chunk_order.size)
        if timings is not None:
            timings.setdefault("num_chunks", int(num_chunks))
            timings.setdefault("total_available_chunks", int(total_available_chunks))
        if num_chunks == 0:
            return total_accumulated_updates, total_accumulated_influence, total_samples_processed

        if chunk_indices is not None and self._using_async_loader():
            self._ensure_async_loader_order(chunk_order, force_restart=True)
        
        # Get weight update frequency from params (default to 10 if not set)
        weight_update_frequency = getattr(params, 'weight_update_frequency', 10)
        if timings is not None:
            timings.setdefault("weight_update_frequency", int(weight_update_frequency))
        processing_config = getattr(params, "processing_config", None)
        metric = getattr(processing_config, "distance_metric", "euclidean") if processing_config else "euclidean"
        metric_kwargs = getattr(processing_config, "distance_metric_params", {}) or {}
        
        # Lists to accumulate updates within super-chunks
        super_chunk_updates = []
        super_chunk_influences = []
        
        logger.debug(f"Worker {self.worker_id}: Processing {num_chunks} chunks with update frequency {weight_update_frequency}")
        
        # Check if multi-buffering is enabled
        if self.multi_buffering_enabled:
            # Process with multi-buffering
            num_buffers = int(self.multi_buffering_num_buffers)
            for chunk_position, chunk_idx in enumerate(chunk_order):
                if chunk_position % 10 == 0:
                    logger.info(f"Processing chunk {chunk_position} of {num_chunks}")
                next_stop = chunk_position + max(1, num_buffers)
                prefetch_indices = [
                    int(next_idx)
                    for next_idx in chunk_order[chunk_position + 1:next_stop]
                ]
                # Load chunk with multi-buffering
                schedule_start = time.perf_counter() if timings is not None else None
                chunk_data, chunk_info = self.load_chunk_multi_buffered(
                    int(chunk_idx),
                    prefetch_indices=prefetch_indices,
                )
                if timings is not None and schedule_start is not None:
                    timings["chunk_schedule_s"] = float(
                        timings.get("chunk_schedule_s", 0.0) + (time.perf_counter() - schedule_start)
                    )
                buffer_id = chunk_info.get('buffer_id') if chunk_info else None
                if buffer_id is not None:
                    wait_start = time.perf_counter() if timings is not None else None
                    self._wait_for_buffer_ready(buffer_id)
                    if timings is not None and wait_start is not None:
                        timings["chunk_wait_ready_s"] = float(
                            timings.get("chunk_wait_ready_s", 0.0) + (time.perf_counter() - wait_start)
                        )

                # Handle padding if present
                if chunk_info.get('padded', False):
                    actual_size = chunk_info['local_size']
                    chunk_data = chunk_data[:actual_size]
                if buffer_id is not None:
                    chunk_data, chunk_info = self._subsample_loaded_chunk(chunk_data, chunk_info)
                
                if timings is not None and isinstance(chunk_info, dict):
                    source = chunk_info.get("source")
                    if isinstance(source, str) and source:
                        key = f"chunk_source_{source}_count"
                        timings[key] = int(timings.get(key, 0) or 0) + 1
                
                if len(chunk_data) == 0:
                    continue
                
                # All computation on compute stream
                with self.compute_stream:
                    # Find BMUs for this chunk
                    bmus = find_bmus(
                        chunk_data, self.gpu_weights, return_distances=False,
                        metric=metric, **metric_kwargs
                    )
                    
                    # Get raw accumulated updates and influence for this chunk
                    use_sparse = bool(getattr(processing_config, "use_sparse_influence", False))
                    if use_sparse:
                        chunk_updates, chunk_influence = compute_weight_updates_sparse(
                            batch=chunk_data,
                            bmus=bmus,
                            weights=self.gpu_weights,
                            influence_matrix=self.influence_matrix,
                            learning_rate=params.current_learning_rate,
                            apply_lr_in_update=(getattr(processing_config, "normalization", None) != "xpysom"),
                            verbose=False
                        )
                    else:
                        chunk_updates, chunk_influence = compute_weight_updates(
                            batch=chunk_data,
                            bmus=bmus,
                            weights=self.gpu_weights,
                            influence_matrix=self.influence_matrix,
                            learning_rate=params.current_learning_rate,
                            apply_lr_in_update=(getattr(processing_config, "normalization", None) != "xpysom"),
                            chunk_size=self.chunk_size or 1000,
                            verbose=False
                        )
                    
                    # Add to super-chunk lists instead of accumulating immediately
                    super_chunk_updates.append(chunk_updates)
                    super_chunk_influences.append(chunk_influence)
                    total_samples_processed += len(chunk_data)
                    
                    # Accumulate super-chunk when we reach the frequency threshold
                    if (chunk_position + 1) % weight_update_frequency == 0 or chunk_position == num_chunks - 1:
                        # Sum all updates in the super-chunk
                        if super_chunk_updates:
                            super_update = cp.zeros_like(self.gpu_weights)
                            super_influence = cp.zeros(self.gpu_weights.shape[0])
                            for update, influence in zip(super_chunk_updates, super_chunk_influences):
                                super_update += update
                                super_influence += influence
                            
                            # Now do the single accumulation for this super-chunk
                            total_accumulated_updates += super_update
                            total_accumulated_influence += super_influence
                            
                            # Clear the lists for next super-chunk
                            super_chunk_updates = []
                            super_chunk_influences = []
                            
                            # Clear memory after super-chunk
                            del super_update, super_influence
                    
                    # Free temporary variables explicitly
                    del bmus
                    if buffer_id is not None:
                        self._mark_buffer_done(buffer_id)
                
                # Clear GPU memory periodically at super-chunk boundaries
                if (chunk_position + 1) % weight_update_frequency == 0:
                    sync_start = time.perf_counter() if timings is not None else None
                    self.compute_stream.synchronize()
                    if timings is not None and sync_start is not None:
                        timings["compute_stream_sync_s"] = float(
                            timings.get("compute_stream_sync_s", 0.0) + (time.perf_counter() - sync_start)
                        )
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
            
            # Ensure compute stream completes before returning
            final_sync_start = time.perf_counter() if timings is not None else None
            self.compute_stream.synchronize()
            if timings is not None and final_sync_start is not None:
                timings["compute_stream_sync_s"] = float(
                    timings.get("compute_stream_sync_s", 0.0) + (time.perf_counter() - final_sync_start)
                )
            
        else:
            # Original synchronous path with super-chunk accumulation
            for chunk_position, chunk_idx in enumerate(chunk_order):
                # Load chunk
                load_start = time.perf_counter() if timings is not None else None
                chunk_data = self.load_chunk(int(chunk_idx))
                if timings is not None and load_start is not None:
                    timings["chunk_load_s"] = float(
                        timings.get("chunk_load_s", 0.0) + (time.perf_counter() - load_start)
                    )
                
                if len(chunk_data) == 0:
                    continue
                    
                # Find BMUs for this chunk
                bmus = find_bmus(
                    chunk_data, self.gpu_weights, return_distances=False,
                    metric=metric, **metric_kwargs
                )
                
                # Get raw accumulated updates and influence for this chunk
                use_sparse = bool(getattr(processing_config, "use_sparse_influence", False))
                if use_sparse:
                    chunk_updates, chunk_influence = compute_weight_updates_sparse(
                        batch=chunk_data,
                        bmus=bmus,
                        weights=self.gpu_weights,
                        influence_matrix=self.influence_matrix,
                        learning_rate=params.current_learning_rate,
                        apply_lr_in_update=(getattr(processing_config, "normalization", None) != "xpysom"),
                        verbose=False
                    )
                else:
                    chunk_updates, chunk_influence = compute_weight_updates(
                        batch=chunk_data,
                        bmus=bmus,
                        weights=self.gpu_weights,
                        influence_matrix=self.influence_matrix,
                        learning_rate=params.current_learning_rate,
                        apply_lr_in_update=(getattr(processing_config, "normalization", None) != "xpysom"),
                        chunk_size=self.chunk_size or 1000,
                        verbose=False
                    )
                
                # Add to super-chunk lists instead of accumulating immediately
                super_chunk_updates.append(chunk_updates)
                super_chunk_influences.append(chunk_influence)
                total_samples_processed += len(chunk_data)
                
                # Free chunk data and BMUs immediately
                del chunk_data, bmus
                
                # Accumulate super-chunk when we reach the frequency threshold
                if (chunk_position + 1) % weight_update_frequency == 0 or chunk_position == num_chunks - 1:
                    # Sum all updates in the super-chunk
                    if super_chunk_updates:
                        super_update = cp.zeros_like(self.gpu_weights)
                        super_influence = cp.zeros(self.gpu_weights.shape[0])
                        for update, influence in zip(super_chunk_updates, super_chunk_influences):
                            super_update += update
                            super_influence += influence
                        
                        # Now do the single accumulation for this super-chunk
                        total_accumulated_updates += super_update
                        total_accumulated_influence += super_influence
                        
                        # Clear the lists for next super-chunk
                        super_chunk_updates = []
                        super_chunk_influences = []
                        
                        # Clear memory after super-chunk
                        del super_update, super_influence
                        sync_start = time.perf_counter() if timings is not None else None
                        cp.cuda.get_current_stream().synchronize()
                        if timings is not None and sync_start is not None:
                            timings["default_stream_sync_s"] = float(
                                timings.get("default_stream_sync_s", 0.0)
                                + (time.perf_counter() - sync_start)
                            )
                        mempool = cp.get_default_memory_pool()
                        mempool.free_all_blocks()
        
        # Final GPU memory cleanup
        final_sync_start = time.perf_counter() if timings is not None else None
        cp.cuda.get_current_stream().synchronize()
        if timings is not None and final_sync_start is not None:
            timings["default_stream_sync_s"] = float(
                timings.get("default_stream_sync_s", 0.0) + (time.perf_counter() - final_sync_start)
            )
        cp.get_default_memory_pool().free_all_blocks()
        
        return total_accumulated_updates, total_accumulated_influence, total_samples_processed
    
    
    def _apply_nccl_reduction(self, accumulated_updates: cp.ndarray, 
                             accumulated_influence: cp.ndarray) -> None:
        """
        Apply NCCL AllReduce to sum across all GPUs.
        
        Args:
            accumulated_updates: Updates to reduce
            accumulated_influence: Influence to reduce
        """
        collective.allreduce(
            accumulated_updates,
            group_name=self.collective_group,
            op=ReduceOp.SUM
        )
        collective.allreduce(
            accumulated_influence,
            group_name=self.collective_group,
            op=ReduceOp.SUM
        )
    
    def _finalize_weight_updates(self, accumulated_updates: cp.ndarray,
                                accumulated_influence: cp.ndarray,
                                params: Dict, total_samples: int) -> cp.ndarray:
        """
        Apply normalization and momentum to get final weight changes.
        
        Args:
            accumulated_updates: Accumulated weight updates
            accumulated_influence: Accumulated influence values
            params: Training parameters
            total_samples: Total samples processed
            
        Returns:
            Final weight changes to apply
        """
        # Apply normalization
        normalized_updates = apply_normalization(
            accumulated_summed_updates=accumulated_updates,
            accumulated_influence_sum=accumulated_influence,
            original_batch_size=total_samples,
            normalization=params.processing_config.normalization,
            learning_rate=params.current_learning_rate,
            current_weights=self.gpu_weights,
            full_bmu_counts=None,
            norm_alpha=params.processing_config.norm_alpha,
            norm_clamp_factor=params.processing_config.norm_clamp_factor,
            norm_percentile=params.processing_config.norm_percentile,
            norm_max_update_threshold=params.processing_config.norm_max_update_threshold,
            training_progress=params.processing_config.training_progress,
            current_epoch=params.processing_config.current_epoch,
            total_epochs=params.processing_config.total_epochs,
            virtual_ratio=getattr(params.processing_config, 'virtual_ratio', 0.5),
            verbose=False
        )
        
        # Apply momentum
        momentum = params.current_momentum
        weight_changes, new_delta = apply_momentum(
            normalized_updates=normalized_updates,
            momentum_coefficient=momentum,
            delta_weights=self.delta_weights
        )
        
        # Update persistent delta_weights
        if new_delta is not None:
            self.delta_weights = new_delta
        
        return weight_changes
    
    def __init__(self, worker_id: int, num_gpus: int, worker_config: Dict[str, Any], 
                 gpu_id: Optional[int] = None, chunk_size: Optional[int] = None):
        """
        Initialize batch worker
        
        Args:
            worker_id: Worker ID for identification
            num_gpus: Total number of GPUs in cluster
            worker_config: Worker configuration dictionary
            gpu_id: Optional specific GPU ID to use (defaults to Ray-assigned GPU 0)
            chunk_size: Optimal chunk size in number of samples
        """
        super().__init__(worker_id, num_gpus, worker_config, gpu_id, chunk_size)

        self.randomize_chunk_order = False

        # Persistent state for momentum - stays on GPU between iterations
        self.delta_weights = None  # Persistent momentum state on GPU
        self.weights_shape = None  # Will be set on first iteration
        # Note: gpu_weights and iteration_count are now in base class
        
        logger.info(f"RayBatchWorker {worker_id} initialized on GPU {self.assigned_gpu} with batch processing support")

    def _ensure_async_loader_started(self) -> None:
        """
        Kick off the async loader when running in RAM async mode.

        The manager sets up async context but defers starting the loader; batch
        workers need to start it before requesting chunks to avoid loader-less
        access errors.
        """
        if not getattr(self, "async_mode", False):
            return
        if getattr(self, "async_loader", None) is not None:
            return

        async_context = getattr(self, "_async_loader_context", None)
        if not async_context:
            return

        num_chunks = int(async_context.get("num_chunks", 0) or 0)
        if num_chunks <= 0:
            return

        base_order = np.arange(num_chunks, dtype=np.int32)
        self._ensure_async_loader_order(base_order, force_restart=True)

    def _resolve_worker_global_span(self) -> Tuple[int, int]:
        """Resolve this worker's global index span [start, end)."""
        global_start = int(getattr(self, "data_start_idx", 0) or 0)
        global_end = int(getattr(self, "data_end_idx", global_start) or global_start)
        if global_end > global_start:
            return global_start, global_end

        local_n_samples = 0
        if self.data_loader is not None:
            try:
                info = self.data_loader.get_info()
                local_n_samples = int(info.get("n_samples", 0) or 0)
            except Exception:
                local_n_samples = 0
        if local_n_samples <= 0 and hasattr(self, "n_samples"):
            local_n_samples = int(getattr(self, "n_samples", 0) or 0)
        return global_start, global_start + max(0, local_n_samples)

    def _group_selected_indices_by_chunk(self, selected_indices: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Map global selected indices to per-chunk local row positions for this worker.
        """
        indices = np.asarray(selected_indices)
        if indices.ndim == 0:
            indices = indices.reshape(1)
        elif indices.ndim != 1:
            raise ValueError("selected_indices must be a 1D array")
        if indices.size == 0:
            return {}
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("selected_indices must contain integer values")

        global_start, global_end = self._resolve_worker_global_span()
        if global_end <= global_start:
            return {}

        indices_int = indices.astype(np.int64, copy=False)
        in_span = indices_int[(indices_int >= global_start) & (indices_int < global_end)]
        if in_span.size == 0:
            return {}

        local_offsets = np.unique((in_span - global_start).astype(np.int64, copy=False))
        loader_chunk = max(
            1,
            int(getattr(self, "loader_chunk_size", 0) or getattr(self, "chunk_size", 1) or 1),
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
        return chunk_positions

    def _configure_async_loader_for_selected_mode(self, selected_mode: bool) -> None:
        """
        Ensure async loader sampling/truncation never drops selector-targeted rows.
        """
        if selected_mode:
            self._async_loader_sampling_fraction_override = 1.0
            # Sentinel: 0 disables target_rows in base worker async loader startup.
            self._async_loader_target_rows_override = 0
        else:
            self._async_loader_sampling_fraction_override = None
            self._async_loader_target_rows_override = None

        if not getattr(self, "async_mode", False):
            return
        async_context = getattr(self, "_async_loader_context", None)
        if not async_context or getattr(self, "async_loader", None) is None:
            return

        if selected_mode:
            desired_sampling_fraction = 1.0
            desired_target_rows = None
        else:
            desired_sampling_fraction = float(getattr(self, "sampling_fraction", 1.0))
            if self.uses_shard_local_random_sampling():
                desired_target_rows = None
            else:
                desired_target_rows = int(self.chunk_size) if self.chunk_size is not None else None

        current_sampling = float(
            getattr(self.async_loader, "sampling_fraction", desired_sampling_fraction)
        )
        current_target_rows = getattr(self.async_loader, "target_rows", desired_target_rows)
        if (
            abs(current_sampling - desired_sampling_fraction) <= 1e-9
            and current_target_rows == desired_target_rows
        ):
            return

        current_order = getattr(self, "_current_async_order", None)
        if current_order is not None and len(current_order) > 0:
            base_order = np.asarray(current_order, dtype=np.int32)
        else:
            num_chunks = int(async_context.get("num_chunks", 0) or 0)
            if num_chunks <= 0:
                return
            base_order = np.arange(num_chunks, dtype=np.int32)
        self._ensure_async_loader_order(base_order, force_restart=True)
    
    def _uses_whole_chunk_random(self, params: Dict, selected_indices: Optional[np.ndarray]) -> bool:
        if selected_indices is not None or not bool(getattr(self, "whole_chunk_random", False)):
            return False

        sampling_cfg = getattr(params, "sampling_config", None)
        if sampling_cfg is None:
            return False

        sampling_method = str(getattr(sampling_cfg, "method", "full") or "full").lower()
        return sampling_method == "random" and bool(getattr(sampling_cfg, "whole_chunk_random", False))

    def _build_whole_chunk_order(self, params: Dict) -> np.ndarray:
        num_chunks = int(self.get_num_chunks())
        if num_chunks <= 0:
            return np.empty(0, dtype=np.int32)

        global_start, global_end = self._resolve_worker_global_span()
        local_rows = max(0, int(global_end) - int(global_start))
        if local_rows <= 0:
            return np.empty(0, dtype=np.int32)

        loader_chunk = max(
            1,
            int(getattr(self, "loader_chunk_size", 0) or getattr(self, "chunk_size", 1) or 1),
        )
        fraction = float(getattr(self, "sampling_fraction", 1.0) or 1.0)
        if fraction >= 1.0:
            return np.arange(num_chunks, dtype=np.int32)

        requested_rows = max(1, int(math.ceil(local_rows * fraction)))
        requested_chunks = min(num_chunks, max(1, int(math.ceil(requested_rows / loader_chunk))))
        if requested_chunks >= num_chunks:
            return np.arange(num_chunks, dtype=np.int32)

        sampling_cfg = getattr(params, "sampling_config", None)
        base_seed = getattr(sampling_cfg, "random_seed", None)
        seed = (
            (int(self.worker_id) << 48)
            ^ (int(self.iteration_count) << 16)
            ^ int(base_seed or 0)
        )
        rng = np.random.default_rng(seed)
        return rng.choice(num_chunks, size=requested_chunks, replace=False).astype(np.int32, copy=False)

    def process_batch_iteration(
        self,
        som_weights: Optional[np.ndarray],
        topology_data: Optional[Any],
        params: Dict,
        collective_group: str,
        chunk_index: Optional[int] = None,
        selected_indices: Optional[np.ndarray] = None,
        is_first_iteration: bool = False,
        driver_submit_ts: Optional[float] = None,
        collect_timing: bool = False,
    ) -> Dict[str, Any]:
        """
        Process batch iteration with NCCL synchronization.
        Keeps weights on GPU between iterations.
        
        Args:
            som_weights: Current SOM weights (only used in first iteration, can be None after)
            topology_data: Topology information
            params: Training parameters
            collective_group: NCCL collective group name
            chunk_index: Optional chunk index to process (if None, processes all chunks)
            selected_indices: Optional array of indices to process (if None, processes all data)
            is_first_iteration: Whether this is the first iteration
            
        Returns:
            Dictionary with status and metadata (no weight transfer)
        """
        timings: Optional[Dict[str, Any]] = {} if collect_timing else None
        iter_perf_start = time.perf_counter() if collect_timing else None

        if timings is not None and driver_submit_ts is not None:
            try:
                timings["queue_delay_s"] = float(time.time() - float(driver_submit_ts))
            except Exception:
                pass

        with self.device:
            if topology_data is not None:
                self._cached_topology_data = topology_data
            else:
                topology_data = getattr(self, "_cached_topology_data", None)
                if topology_data is None:
                    raise ValueError("topology_data must be provided on the first iteration")

            self._start_worker_profile(params, label="batch")
            self.collective_group = collective_group
            self.iteration_count += 1
            
            # Optional extra sync point (debug/stability; hurts multi-node performance)
            if self.collective_barriers_enabled():
                try:
                    barrier_start = time.perf_counter() if timings is not None else None
                    collective.barrier(group_name=collective_group)
                    if timings is not None and barrier_start is not None:
                        timings["barrier_start_s"] = float(time.perf_counter() - barrier_start)
                    logger.debug(f"Worker {self.worker_id}: Synchronized at iteration start (batch mode)")
                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Start barrier synchronization failed: {e}")
                    raise

            # Cache influence matrix on GPU once per iteration (align with colors worker)
            influence_lookup_start = time.perf_counter() if timings is not None else None
            influence_matrix = topology_data.get_precomputed_influence_matrix(
                params.current_radius, 'gaussian'
            )
            if timings is not None and influence_lookup_start is not None:
                timings["influence_lookup_s"] = float(time.perf_counter() - influence_lookup_start)

            influence_to_gpu_start = time.perf_counter() if timings is not None else None
            use_sparse = bool(getattr(params.processing_config, "use_sparse_influence", False))
            if use_sparse:
                self.influence_matrix = _ensure_sparse_influence_csr(influence_matrix)
            else:
                self.influence_matrix = cp.asarray(influence_matrix)
            if timings is not None and influence_to_gpu_start is not None:
                cp.cuda.get_current_stream().synchronize()
                timings["influence_to_gpu_s"] = float(time.perf_counter() - influence_to_gpu_start)
            
            # Reset for new iteration
            if not is_first_iteration:
                reset_start = time.perf_counter() if timings is not None else None
                self.reset_for_iteration()
                if timings is not None and reset_start is not None:
                    timings["reset_for_iteration_s"] = float(time.perf_counter() - reset_start)
            
            # Initialize or reuse persistent weights
            if is_first_iteration:
                # First iteration: initialize from provided weights
                if som_weights is None:
                    raise ValueError("som_weights required for first iteration")
                init_weights_start = time.perf_counter() if timings is not None else None
                self.initialize_gpu_weights(som_weights, collective_group)
                if timings is not None and init_weights_start is not None:
                    cp.cuda.get_current_stream().synchronize()
                    timings["init_weights_s"] = float(time.perf_counter() - init_weights_start)
                logger.info(f"Worker {self.worker_id}: Initialized GPU weights for first iteration")
                
                # Enable double buffering if configured
                num_buffers = getattr(params.processing_config, 'enable_multi_buffering', 0)
                if num_buffers and int(num_buffers) > 1 and not self.multi_buffering_enabled:
                    n_features = som_weights.shape[1]
                    enable_mb_start = time.perf_counter() if timings is not None else None
                    self.enable_multi_buffering(n_features, num_buffers=int(num_buffers))
                    if timings is not None and enable_mb_start is not None:
                        cp.cuda.get_current_stream().synchronize()
                        timings["enable_multi_buffering_s"] = float(time.perf_counter() - enable_mb_start)
                    logger.info(
                        "Worker %d: Multi-buffering enabled (%d buffers) for batch processing",
                        self.worker_id,
                        int(num_buffers),
                    )
            else:
                # Subsequent iterations: weights already on GPU from previous iteration
                if self.gpu_weights is None:
                    raise RuntimeError("GPU weights not initialized")
                logger.debug(f"Worker {self.worker_id}: Using persistent GPU weights from iteration {self.iteration_count-1}")
            
            # Initialize delta_weights on first iteration if needed
            if self.delta_weights is None or self.weights_shape != self.gpu_weights.shape:
                self.weights_shape = self.gpu_weights.shape
                # Initialize from params if provided, otherwise zeros
                if hasattr(params, 'delta_weights') and params.delta_weights is not None:
                    self.delta_weights = cp.asarray(params.delta_weights)
                else:
                    self.delta_weights = cp.zeros_like(self.gpu_weights)

            selected_mode = selected_indices is not None
            self._configure_async_loader_for_selected_mode(selected_mode)
            whole_chunk_order = None
            if self._uses_whole_chunk_random(params, selected_indices):
                whole_chunk_order = self._build_whole_chunk_order(params)
                if self._using_async_loader():
                    self._ensure_async_loader_order(whole_chunk_order, force_restart=True)

            # Ensure async RAM loaders are running before fetching any chunks
            async_start = time.perf_counter() if timings is not None else None
            self._ensure_async_loader_started()
            if timings is not None and async_start is not None:
                timings["ensure_async_loader_started_s"] = float(time.perf_counter() - async_start)
            
            # Process data based on mode
            chunk_timings: Optional[Dict[str, Any]] = {} if timings is not None else None
            process_chunks_start = time.perf_counter() if timings is not None else None
            if chunk_index is None:
                if selected_indices is not None:
                    # Process selected indices aligned to chunks
                    accumulated_updates, accumulated_influence, local_samples_count = \
                        self._process_selected_indices(selected_indices, params, topology_data)
                elif whole_chunk_order is not None:
                    accumulated_updates, accumulated_influence, local_samples_count = \
                        self._process_all_chunks(
                            params,
                            topology_data,
                            chunk_indices=whole_chunk_order,
                            timings=chunk_timings,
                        )
                else:
                    # Process all chunks
                    accumulated_updates, accumulated_influence, local_samples_count = \
                        self._process_all_chunks(params, topology_data, timings=chunk_timings)
            else:
                # Process single chunk
                accumulated_updates, accumulated_influence, local_samples_count = \
                    self._process_single_chunk(chunk_index, params, topology_data)
            if timings is not None and process_chunks_start is not None:
                timings["process_chunks_s"] = float(time.perf_counter() - process_chunks_start)
            if timings is not None and isinstance(chunk_timings, dict):
                for key, value in chunk_timings.items():
                    timings[f"chunks_{key}"] = value

            sample_count_allreduce_start = time.perf_counter() if timings is not None else None
            sample_count = cp.array([int(local_samples_count)], dtype=cp.int32)
            collective.allreduce(
                sample_count,
                group_name=self.collective_group,
                op=ReduceOp.SUM,
            )
            total_samples_all_gpus = int(max(int(sample_count[0].get()), 1))
            if timings is not None and sample_count_allreduce_start is not None:
                timings["sample_count_allreduce_s"] = float(
                    time.perf_counter() - sample_count_allreduce_start
                )
            
            # Apply NCCL reduction across all GPUs
            allreduce_start = time.perf_counter() if timings is not None else None
            self._apply_nccl_reduction(accumulated_updates, accumulated_influence)
            if timings is not None and allreduce_start is not None:
                cp.cuda.get_current_stream().synchronize()
                timings["nccl_allreduce_s"] = float(time.perf_counter() - allreduce_start)
            
            # Finalize weight updates with normalization and momentum
            finalize_start = time.perf_counter() if timings is not None else None
            weight_changes = self._finalize_weight_updates(
                accumulated_updates, accumulated_influence, params, total_samples_all_gpus
            )
            if timings is not None and finalize_start is not None:
                cp.cuda.get_current_stream().synchronize()
                timings["finalize_updates_s"] = float(time.perf_counter() - finalize_start)
            
            # Apply weight changes directly to persistent GPU weights
            apply_start = time.perf_counter() if timings is not None else None
            self.gpu_weights = self.gpu_weights + weight_changes
            if timings is not None and apply_start is not None:
                cp.cuda.get_current_stream().synchronize()
                timings["apply_weight_changes_s"] = float(time.perf_counter() - apply_start)

            # Compute scalar norm of applied weight change for lightweight reporting
            try:
                weight_change_norm = float(cp.linalg.norm(weight_changes))
            except Exception:
                weight_change_norm = 0.0
            
            # Clear GPU memory for temporary data
            cleanup_start = time.perf_counter() if timings is not None else None
            cp.cuda.get_current_stream().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            if timings is not None and cleanup_start is not None:
                timings["cleanup_s"] = float(time.perf_counter() - cleanup_start)
            
            # Optional extra sync point (debug/stability; hurts multi-node performance)
            if self.collective_barriers_enabled():
                try:
                    barrier_end = time.perf_counter() if timings is not None else None
                    collective.barrier(group_name=collective_group)
                    if timings is not None and barrier_end is not None:
                        timings["barrier_end_s"] = float(time.perf_counter() - barrier_end)
                    logger.debug(f"Worker {self.worker_id}: Synchronized at iteration end (batch mode)")
                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: End barrier synchronization failed: {e}")
                    raise
            
            # Return status only - weights stay on GPU
            result: Dict[str, Any] = {
                'status': 'success',
                'worker_id': self.worker_id,
                'iteration': self.iteration_count,
                'samples_processed': local_samples_count,
                # Report scalar magnitude of the applied update; avoids per-iteration weight transfer
                'weight_change_norm': weight_change_norm
            }

            if timings is not None and iter_perf_start is not None:
                timings["iteration_total_s"] = float(time.perf_counter() - iter_perf_start)
                result["timing"] = timings
                result["timing_meta"] = {
                    "hostname": socket.gethostname(),
                    "assigned_gpu": int(self.assigned_gpu),
                    "multi_buffering_enabled": bool(self.multi_buffering_enabled),
                    "async_mode": bool(getattr(self, "async_mode", False)),
                }

            return result
    
    def _process_selected_indices(self, selected_indices: np.ndarray, params: Dict,
                                 topology_data: Dict) -> Tuple[cp.ndarray, cp.ndarray, int]:
        """
        Process only selected indices aligned to chunks.
        
        Args:
            selected_indices: Array of indices to process
            params: Training parameters
            topology_data: Topology information
            
        Returns:
            Tuple of (accumulated_updates, accumulated_influence, total_samples_processed)
        """
        # Initialize accumulated updates
        total_accumulated_updates = cp.zeros_like(self.gpu_weights)
        total_accumulated_influence = cp.zeros(self.gpu_weights.shape[0])
        total_samples_processed = 0
        
        selected_by_chunk = self._group_selected_indices_by_chunk(selected_indices)
        if not selected_by_chunk:
            return total_accumulated_updates, total_accumulated_influence, total_samples_processed

        global_start, global_end = self._resolve_worker_global_span()
        if global_end <= global_start:
            return total_accumulated_updates, total_accumulated_influence, total_samples_processed

        loader_chunk = max(
            1,
            int(getattr(self, "loader_chunk_size", 0) or getattr(self, "chunk_size", 1) or 1),
        )

        for chunk_id in sorted(selected_by_chunk.keys()):
            chunk_start = global_start + int(chunk_id) * loader_chunk
            chunk_end = min(chunk_start + loader_chunk, global_end)
            if chunk_end <= chunk_start:
                continue

            chunk_updates, chunk_influence, samples_count = self._process_chunk_range(
                chunk_start,
                chunk_end,
                params,
                topology_data,
                selected_positions=selected_by_chunk[chunk_id],
            )

            if samples_count <= 0:
                continue

            # Accumulate updates only for truly processed selected rows
            total_accumulated_updates += chunk_updates
            total_accumulated_influence += chunk_influence
            total_samples_processed += samples_count
        
        return total_accumulated_updates, total_accumulated_influence, total_samples_processed
