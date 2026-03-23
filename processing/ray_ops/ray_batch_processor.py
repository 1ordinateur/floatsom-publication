"""
Ray-enabled batch processor with multi-GPU support and NCCL collective communication
"""

import ray
from ray.util import collective
from ray.util.collective.types import ReduceOp
import cupy as cp
import numpy as np
from typing import Any, Union, Tuple, Optional, Dict, List
import logging
import time
from dataclasses import dataclass

from ..base_processor import ProcessingMethod
from ...floatsom_params import RayConfig, ProcessingConfig
from ..processing_params import BatchConfig
from .ray_pipeline_base import RayWorkerManager
from .selection_utils import extract_selected_indices

logger = logging.getLogger(__name__)

class RayBatchProcessor(ProcessingMethod):
    """
    Ray-enabled batch processor for distributed multi-GPU training.
    Uses composition instead of multiple inheritance for cleaner architecture.
    """
    
    def __init__(self, batch_config: BatchConfig, ray_config: Optional[RayConfig] = None, processing_config: Optional[ProcessingConfig] = None):
        """
        Initialize Ray batch processor
        
        Args:
            batch_config: Batch processing configuration
            ray_config: RayConfig instance with Ray-specific configuration
        """
        super().__init__()
        
        # Store configurations
        self.batch_config = batch_config
        self.ray_config = ray_config
        
        # Extract batch parameters
        self.mode = batch_config.batch_mode
        self.chunk_size = batch_config.chunk_size  # Always set in config, also used as minibatch size
        self.ray_config.chunk_size = batch_config.chunk_size
        
        # Initialize worker manager using composition
        self.worker_manager = RayWorkerManager(
            num_gpus=self.ray_config.num_gpus,
            ray_config=self.ray_config,
            processing_config=processing_config
        )
        
        # Selector callback for HDSSSOM support
        self.selector_callback = None

        # Avoid shipping large topology objects every iteration (multi-node perf).
        self._send_topology_next = False
                
        logger.info(f"RayBatchProcessor initialized with CPU-GPU pipeline support")
    
    def set_selector(self, selector):
        """
        Set the sample selector for stochastic sampling
        """
        self.selector = selector

    def supports_selection_result(self) -> bool:
        return True

    def prefers_processor_local_sampling(self, data_reference, params) -> bool:
        sampling_cfg = getattr(params, "sampling_config", None)
        sampling_method = getattr(sampling_cfg, "method", "full") if sampling_cfg else "full"
        if str(sampling_method).lower() != "random":
            return False
        return isinstance(data_reference, str)
    
    def set_selector_callback(self, callback):
        """
        Set callback function for HDSSSOM metadata updates
        """
        self.selector_callback = callback
    
    def initialize(self, som_weights, topology, params, data_source):
        """
        Hook called by FloatSOM's _initialize_with_data_source().
        Delegates to base class for one-time worker and data setup.
        
        Args:
            som_weights: Initial SOM weights
            topology: SOM topology
            params: Training parameters
        """
        # Calculate chunk size based on GPU memory and data shape
        # We need to get the data shape from params or data source        
        # Delegate to worker manager for worker/data setup
        self.worker_manager.initialize(som_weights, topology, params, data_source)
        
        # Store params directly (skip BatchProcessor.initialize to avoid circular reference)
        self.params = params
        
        # Store topology for later use if needed
        self.topology_data = topology
        
        # Update selector's total_samples if we have a selector
        if hasattr(self, 'selector') and hasattr(self.selector, 'total_samples'):
            self.selector.total_samples = self.worker_manager.total_samples

    def process_samples(self, samples, som_weights, topology, params):
        """
        Process samples across multiple GPUs with NCCL synchronization.
        Data has already been distributed during initialization.
        
        Args:
            samples: Input samples or indices from selector
            som_weights: Current SOM weights (n_nodes, n_features)
            topology: SOMTopology instance
            params: Training parameters
            
        Returns:
            Updated weights or (updated_weights, delta_weights) if momentum enabled
        """
        # Ensure async RAM distribution setup has completed on every worker.
        self.worker_manager.ensure_distribution_ready()

        # Check that initialization has been done
        if not self.worker_manager.data_distributed:
            raise RuntimeError("Data not distributed. Call initialize() first")
        
        if not self.worker_manager.workers:
            raise RuntimeError("Workers not initialized. Call initialize() first")
        
        # Extract selector-provided global row indices when available.
        selected_indices = extract_selected_indices(samples)
        if selected_indices is not None:
            selected_unique = np.unique(np.asarray(selected_indices, dtype=np.int64))
            selected_total = int(selected_unique.size)
            params.total_selected_samples = selected_total
            params.selected_total_samples = selected_total
        else:
            params.total_selected_samples = None
            params.selected_total_samples = None
        
        # Pass params object directly to workers
        # Workers will access params.current_radius, params.current_learning_rate, etc.
        # Add total_samples from worker manager
        params.total_samples = self.worker_manager.total_samples
        # Add weight update frequency from batch config
        params.weight_update_frequency = self.batch_config.weight_update_frequency
        
        # Check if this is the first iteration
        is_first_iteration = not hasattr(self, '_iteration_count')
        if is_first_iteration:
            self._iteration_count = 0
        self._iteration_count += 1
        
        # Avoid repeatedly serializing and shipping the full topology object over the
        # network. Workers cache the last topology they received.
        should_send_topology = bool(is_first_iteration or getattr(self, "_send_topology_next", False))
        topology_payload = topology if should_send_topology else None

        iteration_value: Optional[int] = None
        try:
            iteration_value = int(getattr(params, "current_iteration", None))
        except Exception:
            iteration_value = None

        log_every = 0
        if isinstance(self.ray_config, dict):
            log_every = int(self.ray_config.get("timing_log_every_n_iterations", 0) or 0)
        else:
            log_every = int(getattr(self.ray_config, "timing_log_every_n_iterations", 0) or 0)

        should_log_timing = False
        if log_every > 0:
            if iteration_value is None:
                should_log_timing = bool(is_first_iteration)
            else:
                should_log_timing = bool(iteration_value == 0 or (iteration_value % log_every == 0))

        if should_log_timing:
            iter_label = iteration_value if iteration_value is not None else -1
            if is_first_iteration and som_weights is not None:
                try:
                    weights_nbytes = int(getattr(som_weights, "nbytes", 0) or 0)
                    weights_shape = getattr(som_weights, "shape", None)
                    logger.info(
                        "[RayTiming] iter=%d weights_payload: type=%s shape=%s nbytes=%.2fMB",
                        iter_label,
                        type(som_weights).__name__,
                        str(weights_shape),
                        weights_nbytes / (1024**2),
                    )
                except Exception:
                    logger.info(
                        "[RayTiming] iter=%d weights_payload: type=%s (failed to compute nbytes)",
                        iter_label,
                        type(som_weights).__name__,
                    )

            if topology_payload is not None:
                topo_use_cpu_storage = getattr(topology_payload, "_use_cpu_storage", None)
                topo_distance = getattr(topology_payload, "_distance_matrix", None)
                topo_influence_cache = getattr(topology_payload, "_influence_map_cache", None)

                distance_nbytes = 0
                distance_type = None
                if topo_distance is not None:
                    distance_type = type(topo_distance).__name__
                    try:
                        distance_nbytes = int(getattr(topo_distance, "nbytes", 0) or 0)
                    except Exception:
                        distance_nbytes = 0

                influence_count = None
                influence_nbytes = None
                influence_value_type = None
                if isinstance(topo_influence_cache, dict):
                    influence_count = len(topo_influence_cache)
                    total_bytes = 0
                    first_type = None
                    for value in topo_influence_cache.values():
                        if first_type is None and value is not None:
                            first_type = type(value).__name__
                        try:
                            total_bytes += int(getattr(value, "nbytes", 0) or 0)
                        except Exception:
                            continue
                    influence_value_type = first_type
                    influence_nbytes = total_bytes

                distance_mb = "n/a"
                if distance_nbytes:
                    distance_mb = f"{distance_nbytes / (1024**2):.2f}MB"

                influence_mb = "n/a"
                if influence_nbytes is not None:
                    influence_mb = f"{float(influence_nbytes) / (1024**2):.2f}MB"

                logger.info(
                    "[RayTiming] iter=%d topology_payload: type=%s use_cpu_storage=%s distance_type=%s distance=%s influence_maps=%s influence_type=%s influence_total=%s",
                    iter_label,
                    type(topology_payload).__name__,
                    str(topo_use_cpu_storage),
                    str(distance_type),
                    distance_mb,
                    str(influence_count) if influence_count is not None else "n/a",
                    str(influence_value_type) if influence_value_type is not None else "n/a",
                    influence_mb,
                )

        # Process iteration on all workers (they read from Zarr via CPU-GPU pipeline)
        submit_start = time.perf_counter()
        futures = []
        for worker in self.worker_manager.workers:
            driver_submit_ts = time.time() if should_log_timing else None
            futures.append(
                worker.process_batch_iteration.remote(
                    som_weights if is_first_iteration else None,  # Only send weights on first iteration
                    topology_payload,
                    params,
                    self.worker_manager.collective_group_name,
                    selected_indices=selected_indices,  # Pass selected indices
                    is_first_iteration=is_first_iteration,
                    driver_submit_ts=driver_submit_ts,
                    collect_timing=should_log_timing,
                )
            )
        submit_s = time.perf_counter() - submit_start

        # Collect status from all GPUs (lightweight dicts only)
        get_start = time.perf_counter()
        results = self.worker_manager.wait_for_worker_futures(
            futures,
            phase="batch_iteration",
        )
        get_s = time.perf_counter() - get_start

        if should_log_timing:
            iter_label = iteration_value if iteration_value is not None else -1
            logger.info(
                "[RayTiming] iter=%d workers=%d submit_s=%.3f ray_get_s=%.3f send_weights=%s send_topology=%s",
                iter_label,
                len(results),
                submit_s,
                get_s,
                bool(is_first_iteration),
                bool(topology_payload is not None),
            )

            def _summarize_timing_key(key: str) -> Optional[Tuple[float, float, float]]:
                values: List[float] = []
                for result in results:
                    timing = result.get("timing")
                    if not isinstance(timing, dict):
                        continue
                    value = timing.get(key)
                    if value is None:
                        continue
                    if isinstance(value, (int, float, np.floating)):
                        values.append(float(value))
                if not values:
                    return None
                return min(values), sum(values) / len(values), max(values)

            slowest_worker_id = None
            slowest_total_s = None
            for result in results:
                timing = result.get("timing")
                if not isinstance(timing, dict):
                    continue
                total_s = timing.get("iteration_total_s")
                if not isinstance(total_s, (int, float, np.floating)):
                    continue
                if slowest_total_s is None or float(total_s) > slowest_total_s:
                    slowest_total_s = float(total_s)
                    slowest_worker_id = result.get("worker_id")

            breakdown_keys = [
                "queue_delay_s",
                "barrier_start_s",
                "barrier_end_s",
                "influence_lookup_s",
                "influence_to_gpu_s",
                "reset_for_iteration_s",
                "init_weights_s",
                "enable_multi_buffering_s",
                "ensure_async_loader_started_s",
                "process_chunks_s",
                "chunks_chunk_schedule_s",
                "chunks_chunk_wait_ready_s",
                "chunks_chunk_load_s",
                "chunks_compute_stream_sync_s",
                "chunks_default_stream_sync_s",
                "nccl_allreduce_s",
                "finalize_updates_s",
                "apply_weight_changes_s",
                "cleanup_s",
                "iteration_total_s",
            ]

            breakdown_parts: List[str] = []
            breakdown_summary: Dict[str, Dict[str, float]] = {}
            for key in breakdown_keys:
                summary = _summarize_timing_key(key)
                if summary is None:
                    continue
                _, mean_s, max_s = summary
                breakdown_summary[key] = {
                    "mean_s": float(mean_s),
                    "max_s": float(max_s),
                }
                breakdown_parts.append(f"{key} mean={mean_s:.3f}s max={max_s:.3f}s")

            if breakdown_parts:
                slowest_label = ""
                if slowest_worker_id is not None and slowest_total_s is not None:
                    slowest_label = f" slowest_worker={slowest_worker_id} total={slowest_total_s:.3f}s"
                logger.info(
                    "[RayTiming] iter=%d worker_breakdown:%s %s",
                    iter_label,
                    slowest_label,
                    " | ".join(breakdown_parts),
                )

            # Persist the timing breakdown so benchmark harnesses can write it to disk.
            timing_summary: Dict[str, Any] = {
                "iteration": int(iter_label),
                "workers": int(len(results)),
                "driver_submit_s": float(submit_s),
                "driver_ray_get_s": float(get_s),
                "send_weights": bool(is_first_iteration),
                "send_topology": bool(topology_payload is not None),
                "worker_breakdown_s": breakdown_summary,
            }

            if slowest_worker_id is not None:
                try:
                    timing_summary["slowest_worker_id"] = int(slowest_worker_id)
                except Exception:
                    timing_summary["slowest_worker_id"] = slowest_worker_id
            if slowest_total_s is not None:
                timing_summary["slowest_worker_total_s"] = float(slowest_total_s)

            # Include per-worker metadata (hostname, assigned GPU, async/buffering flags).
            worker_meta: List[Dict[str, Any]] = []
            for result in results:
                meta = result.get("timing_meta")
                if not isinstance(meta, dict):
                    continue
                worker_meta.append(
                    {
                        "worker_id": int(result.get("worker_id", -1))
                        if isinstance(result.get("worker_id"), (int, np.integer))
                        else result.get("worker_id"),
                        "hostname": meta.get("hostname"),
                        "assigned_gpu": meta.get("assigned_gpu"),
                        "multi_buffering_enabled": meta.get("multi_buffering_enabled"),
                        "async_mode": meta.get("async_mode"),
                    }
                )
            if worker_meta:
                timing_summary["worker_meta"] = worker_meta

            # Chunk source counters (disk mode vs cache hits) are recorded as counts.
            chunk_source_keys = set()
            for result in results:
                timing = result.get("timing")
                if not isinstance(timing, dict):
                    continue
                for key in timing.keys():
                    if isinstance(key, str) and key.startswith("chunks_chunk_source_") and key.endswith("_count"):
                        chunk_source_keys.add(key)

            chunk_sources_summary: Dict[str, Dict[str, float]] = {}
            for key in sorted(chunk_source_keys):
                summary = _summarize_timing_key(key)
                if summary is None:
                    continue
                _, mean_s, max_s = summary
                chunk_sources_summary[key] = {"mean": float(mean_s), "max": float(max_s)}
            if chunk_sources_summary:
                timing_summary["chunk_source_counts"] = chunk_sources_summary

            num_chunks = None
            weight_update_frequency = None
            for result in results:
                timing = result.get("timing")
                if not isinstance(timing, dict):
                    continue
                if num_chunks is None and isinstance(timing.get("chunks_num_chunks"), (int, float, np.floating)):
                    num_chunks = int(timing["chunks_num_chunks"])
                if weight_update_frequency is None and isinstance(
                    timing.get("chunks_weight_update_frequency"), (int, float, np.floating)
                ):
                    weight_update_frequency = int(timing["chunks_weight_update_frequency"])

            if num_chunks is not None or weight_update_frequency is not None:
                timing_summary["chunking"] = {
                    "num_chunks": int(num_chunks) if num_chunks is not None else None,
                    "weight_update_frequency": int(weight_update_frequency)
                    if weight_update_frequency is not None
                    else None,
                }
                logger.info(
                    "[RayTiming] iter=%d chunking: num_chunks=%s weight_update_frequency=%s",
                    iter_label,
                    str(num_chunks) if num_chunks is not None else "n/a",
                    str(weight_update_frequency) if weight_update_frequency is not None else "n/a",
                )

            self.worker_manager.last_iteration_timing_stats = timing_summary
            history = getattr(self.worker_manager, "iteration_timing_history", None)
            if isinstance(history, list):
                history.append(timing_summary)
            else:
                self.worker_manager.iteration_timing_history = [timing_summary]
        
        # All workers should report success
        for result in results:
            if result['status'] != 'success':
                raise RuntimeError(f"Worker {result['worker_id']} failed")
        
        # Aggregate lightweight metrics and avoid per-iteration weight transfer
        total_samples = sum(r.get('samples_processed', 0) for r in results)
        # All workers see identical global updates after NCCL reduction; take first norm
        weight_change_norm = results[0].get('weight_change_norm', 0.0) if results else 0.0

        if should_send_topology:
            self._send_topology_next = False

        return {
            'update_type': 'inplace',
            'weight_change_norm': float(weight_change_norm),
            'samples_processed': int(total_samples)
        }

    def update_topology(self, topology) -> None:
        """
        Mark that workers need an updated topology object on the next iteration.

        This is used for dynamic topologies (e.g., MST) to avoid shipping the
        topology every iteration on multi-node jobs.
        """
        self.topology_data = topology
        self._send_topology_next = True
    
    # Note: _nccl_sum_across_gpus and _apply_global_normalization_and_momentum methods removed
    # NCCL synchronization and normalization are now done inside RayBatchWorker
    
    def cleanup(self):
        """Clean up Ray resources"""
        # Delegate cleanup to worker manager
        if hasattr(self, 'worker_manager'):
            self.worker_manager.cleanup()
        
        logger.info("[Ray] Batch processor cleanup completed")

    # Lightweight accessor for driver to fetch current weights when needed
    def get_current_weights(self):
        """
        Fetch current SOM weights from the first worker.
        Used sparingly (e.g., for MST topology updates) to avoid per-iteration transfers.
        """
        if not hasattr(self, 'worker_manager') or not self.worker_manager.workers:
            raise RuntimeError("No workers initialized")
        result = self.worker_manager.get_gpu_weights()

        # Ray serializes CuPy arrays when transferring back to the driver; ensure
        # we always restore GPU arrays so downstream topology updates stay in CuPy.
        if isinstance(result, tuple):
            weights, delta = result
            weights_gpu = cp.asarray(weights)
            delta_gpu = cp.asarray(delta) if delta is not None else None
            return weights_gpu if delta_gpu is None else (weights_gpu, delta_gpu)

        return cp.asarray(result)
