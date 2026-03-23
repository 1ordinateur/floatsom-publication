"""
Ray-enabled multi-GPU color set processor with memory-efficient chunking
Extends colors_processor.py for distributed processing across multiple GPUs
"""

import ray
from ray.util import collective
from ray.util.collective.types import ReduceOp
import cupy as cp
import numpy as np
from typing import Union, Tuple, Optional, Dict, List, Any
import logging
import statistics
import time

from ..base_processor import ProcessingMethod
from ...floatsom_params import RayConfig, ProcessingConfig
from ..processing_params import ColorsConfig
from ..colour_operations.colour_set_creation import (
    calculate_equal_sized_color_sets, 
    calculate_batch_all_color_sets,
)
from .ray_pipeline_base import RayWorkerManager
from .selection_utils import extract_selected_indices

logger = logging.getLogger(__name__)


# ColorGPUWorker class moved to ray_color_worker.py as RayColorWorker
# This reduces duplication and follows the unified worker architecture


@ray.remote
class _IterationProgressTracker:
    """Driver-side actor that records last-seen progress heartbeats from workers."""

    def __init__(self) -> None:
        self._last_progress = time.monotonic()
        self._revision = 0
        self._last_worker_id = None
        self._last_stage = None
        self._active_partitions: Dict[int, Tuple[int, float]] = {}
        self._partition_durations_s: Dict[int, List[float]] = {}

    def reset(self) -> None:
        self._last_progress = time.monotonic()
        self._revision = 0
        self._last_worker_id = None
        self._last_stage = None
        self._active_partitions = {}
        self._partition_durations_s = {}

    @staticmethod
    def _parse_partition_stage(stage: Optional[str]) -> Optional[Tuple[int, str]]:
        if not stage or not stage.startswith("partition_"):
            return None

        parts = stage.split("_", 2)
        if len(parts) != 3:
            return None

        try:
            partition_idx = int(parts[1])
        except (TypeError, ValueError):
            return None

        marker = parts[2]
        if marker not in {"start", "done"}:
            return None

        return partition_idx, marker

    def report(self, worker_id: int, stage: Optional[str] = None) -> None:
        self._revision += 1
        now = time.monotonic()
        worker_id_int = int(worker_id)
        self._last_progress = now
        self._last_worker_id = worker_id_int
        self._last_stage = stage

        partition_event = self._parse_partition_stage(stage)
        if partition_event is None:
            return

        partition_idx, marker = partition_event
        if marker == "start":
            self._active_partitions[worker_id_int] = (partition_idx, now)
            return

        active_partition = self._active_partitions.get(worker_id_int)
        if active_partition is None:
            return

        active_idx, start_t = active_partition
        self._active_partitions.pop(worker_id_int, None)
        if active_idx != partition_idx:
            return

        elapsed_s = max(0.0, now - start_t)
        self._partition_durations_s.setdefault(partition_idx, []).append(elapsed_s)

    def snapshot(self) -> Dict[str, object]:
        now = time.monotonic()
        active_partition_elapsed: Dict[int, Dict[str, float]] = {}
        for worker_id, (partition_idx, started_at) in self._active_partitions.items():
            active_partition_elapsed[int(worker_id)] = {
                "partition_idx": int(partition_idx),
                "elapsed_s": max(0.0, now - started_at),
            }

        partition_median_s: Dict[int, float] = {}
        partition_peer_counts: Dict[int, int] = {}
        for partition_idx, durations in self._partition_durations_s.items():
            if not durations:
                continue
            partition_median_s[int(partition_idx)] = float(statistics.median(durations))
            partition_peer_counts[int(partition_idx)] = int(len(durations))

        return {
            "now": now,
            "last_progress": self._last_progress,
            "revision": self._revision,
            "last_worker_id": self._last_worker_id,
            "last_stage": self._last_stage,
            "active_partitions": active_partition_elapsed,
            "partition_median_s": partition_median_s,
            "partition_peer_counts": partition_peer_counts,
        }


class RayColorsProcessor(ProcessingMethod):
    """
    Ray-enabled multi-GPU color set processor using COMPOSITION.
    Single inheritance only - uses worker_manager as a member.
    """

    _DEFAULT_ITERATION_STALL_TIMEOUT_S = 300.0
    _STRAGGLER_TIMEOUT_MULTIPLIER = 4.0
    _STRAGGLER_MIN_ELAPSED_S = 5.0
    _STRAGGLER_MIN_PEER_COMPLETIONS = 2
    
    def __init__(self, colors_config: ColorsConfig, ray_config: Optional[RayConfig] = None, processing_config: Optional[ProcessingConfig] = None):
        """
        Initialize Ray colors processor using COMPOSITION.
        
        Args:
            colors_config: ColorsConfig with color processing parameters
            ray_config: RayConfig dataclass instance with Ray-specific configuration
        """
        super().__init__()
        
        # Store configurations
        self.colors_config = colors_config
        self.ray_config = ray_config
        
        # Ray-specific configuration
        # Use config's chunk size, override with ray_config if specified
        self.chunk_size = colors_config.chunk_size
        self.ray_config.chunk_size = colors_config.chunk_size
        self.worker_manager = RayWorkerManager(
            num_gpus=ray_config.num_gpus,
            ray_config=ray_config,
            processing_config=processing_config
        )
        
        # Extract color processing parameters
        self.processing_mode = colors_config.processing_mode
        self.max_rounds = colors_config.max_rounds
        self.sample_order = colors_config.sample_order
        self.color_set_algorithm = colors_config.color_set_algorithm
        self.num_color_sets = colors_config.num_color_sets
        self.enable_adaptive_bmu = colors_config.enable_adaptive_bmu
        self.bmu_recalc_initial = colors_config.bmu_recalc_initial
        self.bmu_recalc_decay_type = colors_config.bmu_recalc_decay_type
        self.bmu_recalc_min_samples = colors_config.bmu_recalc_min_samples
        
        # Color-specific utilities
        self.color_sets = None
        self._cached_color_sets = None
        self._cached_radius = None
        self.bmu_scheduler = None
        self.use_momentum = False
        self.delta_weights = None
        self.verbose = False
        
        self.collective_group_name = ray_config.collective_group_name
        self._base_collective_group_name = self.collective_group_name or "default"
        self._recovery_attempt = 0
        # NOTE: `ray_config.num_gpus` may be None ("auto") for multi-node clusters.
        # The effective worker/GPU count is resolved after `worker_manager.initialize()`.
        self.num_gpus = ray_config.num_gpus

        # Recovery helpers (driver-side only)
        self._data_source = None
        self._last_good_weights_cpu = None
        self._last_good_delta_cpu = None
        self._progress_tracker = None
        self._watchdog_default_logged = False
        
        logger.info(f"RayColorsProcessor initialized with composition")

    def supports_selection_result(self) -> bool:
        return True

    def _ensure_progress_tracker(self):
        if self._progress_tracker is None:
            self._progress_tracker = _IterationProgressTracker.remote()
        return self._progress_tracker

    def _resolve_iteration_timeout_s(self) -> float:
        configured_timeout = getattr(self.ray_config, "iteration_timeout_s", 0.0)
        try:
            timeout_s = float(configured_timeout)
        except (TypeError, ValueError):
            timeout_s = 0.0

        if timeout_s > 0:
            return timeout_s

        timeout_s = float(self._DEFAULT_ITERATION_STALL_TIMEOUT_S)
        if not self._watchdog_default_logged:
            logger.warning(
                "Ray iteration watchdog timeout is unset/non-positive (%r); "
                "using default %.1fs to avoid unbounded waits.",
                configured_timeout,
                timeout_s,
            )
            self._watchdog_default_logged = True
        return timeout_s

    def _requires_selected_indices_for_random_subsampling(self, params: Any) -> bool:
        del params
        return False

    def _await_iteration_with_watchdog(
        self,
        iteration_futures,
        stall_timeout_s: float,
        progress_tracker,
    ):
        def _dict_lookup(mapping: Any, key: int, default: Any = None):
            if not isinstance(mapping, dict):
                return default
            if key in mapping:
                return mapping[key]
            key_str = str(key)
            if key_str in mapping:
                return mapping[key_str]
            return default

        def _as_float(value: Any, fallback: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        def _as_int(value: Any, fallback: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        poll_interval_s = min(5.0, max(0.5, stall_timeout_s / 10.0))
        last_snapshot = None
        last_snapshot_ok = time.monotonic()
        pending = list(iteration_futures)
        future_index = {ref: idx for idx, ref in enumerate(iteration_futures)}
        results = [None] * len(iteration_futures)
        completed_worker_ids = set()
        required_peer_completions = max(
            1,
            min(self._STRAGGLER_MIN_PEER_COMPLETIONS, max(len(iteration_futures) - 1, 1)),
        )

        while pending:
            ready, pending = ray.wait(
                pending,
                num_returns=1,
                timeout=poll_interval_s,
            )
            if ready:
                for ref in ready:
                    idx = future_index[ref]
                    try:
                        results[idx] = ray.get(ref)
                        if isinstance(results[idx], dict):
                            worker_id = _as_int(results[idx].get("worker_id"), fallback=-1)
                            if worker_id >= 0:
                                completed_worker_ids.add(worker_id)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Ray color worker future {idx} failed during distributed iteration."
                        ) from exc
                # Completed work is forward progress when tracker snapshots are unavailable.
                last_snapshot_ok = time.monotonic()
                continue

            if progress_tracker is None:
                if (time.monotonic() - last_snapshot_ok) > stall_timeout_s:
                    raise TimeoutError(
                        f"Ray color iteration made no completions for >{stall_timeout_s:.1f}s "
                        "without a progress tracker; treating as stalled."
                    )
                continue

            try:
                last_snapshot = ray.get(progress_tracker.snapshot.remote())
                last_snapshot_ok = time.monotonic()
            except Exception as exc:
                logger.debug("Ray iteration progress snapshot failed: %s", exc)
                if (time.monotonic() - last_snapshot_ok) > stall_timeout_s:
                    raise TimeoutError(
                        f"Ray progress watchdog snapshot failed for >{stall_timeout_s:.1f}s; "
                        "treating as stalled iteration."
                    ) from exc
                continue

            try:
                now = float(last_snapshot.get("now", 0.0))
                last_progress = float(last_snapshot.get("last_progress", 0.0))
            except (TypeError, ValueError):
                continue

            stall_s = now - last_progress
            if stall_s > stall_timeout_s:
                worker_id = last_snapshot.get("last_worker_id")
                stage = last_snapshot.get("last_stage")
                revision = last_snapshot.get("revision")
                completed = len(iteration_futures) - len(pending)
                raise TimeoutError(
                    f"No Ray color iteration progress for {stall_s:.1f}s "
                    f"(stall limit {stall_timeout_s:.1f}s). "
                    f"Last heartbeat: worker={worker_id} stage={stage} revision={revision} "
                    f"({completed}/{len(iteration_futures)} workers completed)."
                )

            active_partitions = last_snapshot.get("active_partitions", {})
            partition_median_s = last_snapshot.get("partition_median_s", {})
            partition_peer_counts = last_snapshot.get("partition_peer_counts", {})

            for worker_key, active in active_partitions.items():
                worker_id = _as_int(worker_key, fallback=-1)
                if not isinstance(active, dict):
                    continue
                if worker_id in completed_worker_ids:
                    # A completed worker can leave behind stale partition-start heartbeats
                    # because reporting is best-effort/throttled; only evaluate pending workers.
                    continue

                partition_idx = _as_int(active.get("partition_idx"), fallback=-1)
                elapsed_s = _as_float(active.get("elapsed_s"), fallback=0.0)
                if partition_idx < 0 or elapsed_s <= 0.0:
                    continue

                peer_count = _as_int(_dict_lookup(partition_peer_counts, partition_idx, 0), fallback=0)
                if peer_count < required_peer_completions:
                    continue

                peer_median_s = _as_float(_dict_lookup(partition_median_s, partition_idx, 0.0), fallback=0.0)
                if peer_median_s <= 0.0:
                    continue

                straggler_limit_s = max(
                    self._STRAGGLER_MIN_ELAPSED_S,
                    self._STRAGGLER_TIMEOUT_MULTIPLIER * peer_median_s,
                )
                if elapsed_s > straggler_limit_s:
                    completed = len(iteration_futures) - len(pending)
                    raise TimeoutError(
                        f"Ray color iteration detected worker straggler: worker={worker_id} "
                        f"partition={partition_idx} elapsed={elapsed_s:.1f}s exceeded "
                        f"{self._STRAGGLER_TIMEOUT_MULTIPLIER:.1f}x peer median "
                        f"({peer_median_s:.1f}s, limit {straggler_limit_s:.1f}s, peers={peer_count}). "
                        f"({completed}/{len(iteration_futures)} workers completed)."
                    )

        return results
    
    def initialize(self, som_weights, topology, params, data_source):
        """
        Hook called by FloatSOM's _initialize_with_data_source().
        Delegates to base class for worker/data setup, then prepares color sets.
        
        Args:
            som_weights: Initial SOM weights
            topology: SOM topology  
            params: Training parameters
        """
        # Delegate to worker_manager for worker/data setup
        self.worker_manager.initialize(som_weights, topology, params, data_source)
        self._data_source = data_source
        
        # Initialize color processing parameters directly (skip ColorsProcessor.initialize
        # to avoid circular reference where ColorsProcessor would call self._ray_processor.initialize)
        self.params = params
        self.verbose = params.verbose
        processing_config = params.processing_config
        self.processing_mode = processing_config.processing_mode
        self.use_momentum = processing_config.enable_momentum
        
        # Initialize BMU scheduler
        from ..colour_operations.bmu_scheduler import BMUScheduler
        self.bmu_scheduler = BMUScheduler(
            enabled=self.enable_adaptive_bmu,
            initial_frequency=self.bmu_recalc_initial,
            decay_type=self.bmu_recalc_decay_type,
            min_samples=self.bmu_recalc_min_samples,
            verbose=self.verbose
        )
        
        # Initialize momentum if needed
        if som_weights is not None and self.use_momentum:
            self.delta_weights = cp.zeros_like(som_weights)
            if self.verbose:
                logger.info("Momentum enabled for Ray color set processing")

        # Avoid forcing GPU->CPU transfers during initialization.
        # A CPU checkpoint is materialized lazily only if/when the watchdog triggers.
        if som_weights is not None and not isinstance(som_weights, cp.ndarray):
            self._last_good_weights_cpu = np.asarray(som_weights)
        
        # Prepare initial color sets if we have a radius
        initial_radius = getattr(params, 'current_radius', None) or getattr(params, 'initial_radius', 1.0)
        self.influence_matrix, cached_radius = topology.get_precomputed_influence_matrix(
            initial_radius, 'gaussian', return_radius=True
        )
        self._prepare_color_sets(som_weights, topology, params, cached_radius)
    
    def _prepare_color_sets(self, som_weights, topology, params, cached_radius):
        """
        Prepare color sets using parent's caching mechanism
        """
        # Create params object for color set calculation
        color_params = type('obj', (object,), {
            'color_set_algorithm': self.color_set_algorithm,
            'num_color_sets': self.num_color_sets,
            'max_rounds': self.max_rounds
        })()
        
        # Check cache (inherited attributes from ColorsProcessor)
        if self._cached_color_sets is not None and self._cached_radius == cached_radius:
            self.color_sets = self._cached_color_sets
            if self.verbose:
                logger.debug(f"Using cached color sets (radius={cached_radius:.3f})")
        else:
            # Calculate new color sets using parent's operations
            if self.processing_mode == "equal_sized":
                self.color_sets = calculate_equal_sized_color_sets(
                    None, som_weights, topology, color_params, self.influence_matrix
                )
            else:
                self.color_sets = calculate_batch_all_color_sets(
                    None, som_weights, topology, color_params, self.influence_matrix
                )
            
            # Convert to CuPy arrays for NCCL
            self.color_sets = [cp.asarray(cs, dtype=cp.int32) for cs in self.color_sets]
            
            # Update cache (inherited attributes)
            self._cached_color_sets = self.color_sets
            self._cached_radius = cached_radius
            
            if self.verbose:
                logger.debug(f"Calculated new color sets and cached (radius={cached_radius:.3f})")

    def process_samples(self, samples, som_weights, topology, params):
        """
        Process samples using distributed color set algorithm
        """
        # Ensure async RAM distribution setup has completed on every worker.
        self.worker_manager.ensure_distribution_ready()

        # Check that initialization has been done via worker_manager
        if not self.worker_manager.data_distributed:
            raise RuntimeError("Data not distributed. Call initialize() first")
        
        if not self.worker_manager.workers:
            raise RuntimeError("Workers not initialized. Call initialize() first")
        
        # Get influence matrix and prepare color sets using parent's logic
        radius = params.current_radius
        self.influence_matrix, cached_radius = topology.get_precomputed_influence_matrix(
            radius, 'gaussian', return_radius=True
        )
        
        # Create/retrieve color sets using parent's caching logic
        self._prepare_color_sets(som_weights, topology, params, cached_radius)
        
        return self._process_samples(samples, som_weights, topology, params)
    
    def _process_samples(self, samples, som_weights, topology, params):
        """
        Process samples using distributed color set algorithm
        Uses single iteration method to keep everything on GPU
        """
        if self.verbose:
            start_time = time.time()
                
        processing_cfg = params.processing_config
        num_partitions = max(1, int(processing_cfg.max_rounds))
        sampling_cfg = getattr(params, "sampling_config", None)
        sampling_method = str(getattr(sampling_cfg, "method", "full") or "full").lower()
        selected_indices = extract_selected_indices(samples)
        if sampling_method == "random":
            # KISS path: random colors mirrors full-chunk setup and applies subsampling
            # inside the GPU chunk-processing stage.
            selected_indices = None
        params.total_samples = int(getattr(self.worker_manager, "total_samples", 0) or 0)
        if selected_indices is not None:
            selected_unique = np.unique(np.asarray(selected_indices, dtype=np.int64))
            selected_total = int(selected_unique.size)
            params.total_selected_samples = selected_total
            params.selected_total_samples = selected_total
        else:
            params.total_selected_samples = None
            params.selected_total_samples = None
        if self._requires_selected_indices_for_random_subsampling(params) and selected_indices is None:
            raise ValueError(
                "Random subsampling requires selector-provided selected_indices when "
                "effective sampling fraction is below 1.0."
            )
        shuffle_seed = None
        if processing_cfg.sample_order == 'random':
            base_seed = getattr(processing_cfg, "random_seed", None)
            if base_seed is None:
                if not hasattr(self, "_rng"):
                    self._rng = np.random.default_rng()
                base_seed = int(self._rng.integers(0, 2**32 - 1))
            shuffle_seed = int(base_seed)

        color_order_seed = getattr(processing_cfg, "random_seed", None)
        if color_order_seed is None:
            color_order_seed = getattr(params, "seed", None)
        if color_order_seed is None:
            if shuffle_seed is not None:
                color_order_seed = int(shuffle_seed)
            else:
                if not hasattr(self, "_rng"):
                    self._rng = np.random.default_rng()
                color_order_seed = int(self._rng.integers(0, 2**32 - 1))
        color_order_seed = int(color_order_seed)

        color_round_plan = {
            'num_partitions': num_partitions,
            'shuffle_seed': shuffle_seed,
            'color_order_seed': color_order_seed,
            'target_batch_size': int(self.chunk_size)
        }

        if self.verbose:
            effective_gpus = (
                len(self.worker_manager.workers)
                if getattr(self.worker_manager, "workers", None)
                else self.num_gpus
            )
            logger.info(
                f"Processing with {len(self.color_sets)} color sets across {effective_gpus} GPUs "
                f"using {num_partitions} partitions"
            )

        timeout_s = self._resolve_iteration_timeout_s()

        max_retries = getattr(self.ray_config, "iteration_timeout_max_retries", 1)
        try:
            max_retries = int(max_retries)
        except (TypeError, ValueError):
            max_retries = 1
        max_retries = max(0, max_retries)

        enable_recovery = max_retries > 0

        checkpoint_weights_cpu = None
        checkpoint_delta_cpu = None
        if enable_recovery:
            checkpoint_weights_cpu = self._last_good_weights_cpu
            checkpoint_delta_cpu = self._last_good_delta_cpu

        # Initialize weights only on the first iteration to avoid repeated transfers.
        is_first_iteration = not hasattr(self, '_iteration_count')
        if is_first_iteration:
            self._iteration_count = 0

        results = None
        for attempt in range(max_retries + 1):
            self._iteration_count += 1
            is_retry = attempt > 0

            if is_retry:
                if checkpoint_weights_cpu is None:
                    raise RuntimeError(
                        "Ray color iteration timed out, but no last-known-good weights are available for recovery."
                    )
                weights_for_workers = checkpoint_weights_cpu
                # Best-effort restore momentum state if available.
                if checkpoint_delta_cpu is not None:
                    params.delta_weights = checkpoint_delta_cpu
            else:
                weights_for_workers = som_weights if is_first_iteration else None

            params.som_weights = weights_for_workers

            progress_tracker = self._ensure_progress_tracker()
            try:
                ray.get(progress_tracker.reset.remote())
            except Exception:
                self._progress_tracker = None
                progress_tracker = self._ensure_progress_tracker()
                try:
                    ray.get(progress_tracker.reset.remote())
                except Exception:
                    progress_tracker = None

            try:
                # Setup workers for processing
                setup_futures = [
                    worker.setup_processing.remote(
                        weights_for_workers,
                        self.influence_matrix,
                        self.color_sets,
                        self.collective_group_name,
                        params,
                        color_round_plan,
                        selected_indices=selected_indices,
                        progress_tracker=progress_tracker,
                        progress_interval_s=5.0,
                    )
                    for worker in self.worker_manager.workers
                ]
                self.worker_manager.wait_for_worker_futures(
                    setup_futures,
                    phase="colors_iteration_setup",
                    timeout_s=timeout_s,
                )

                # Process entire iteration on workers (single call!)
                iteration_futures = [
                    worker.process_full_iteration.remote()
                    for worker in self.worker_manager.workers
                ]
                results = self._await_iteration_with_watchdog(
                    iteration_futures,
                    stall_timeout_s=timeout_s,
                    progress_tracker=progress_tracker,
                )
            except Exception as iteration_error:
                logger.warning(
                    "Ray color iteration attempt %d/%d failed (%s); retrying from checkpoint if available.",
                    attempt + 1,
                    max_retries + 1,
                    iteration_error,
                )

                if attempt >= max_retries or not enable_recovery:
                    raise

                # Lazily materialize a CPU recovery checkpoint only when a failure occurs.
                # This avoids an unconditional GPU->CPU transfer in the common case.
                if checkpoint_weights_cpu is None:
                    if weights_for_workers is None:
                        raise RuntimeError(
                            "Ray color iteration failed and no recovery checkpoint weights are available"
                        ) from iteration_error
                    try:
                        checkpoint_weights_cpu = (
                            cp.asnumpy(weights_for_workers)
                            if isinstance(weights_for_workers, cp.ndarray)
                            else np.asarray(weights_for_workers)
                        )
                        self._last_good_weights_cpu = checkpoint_weights_cpu
                    except Exception as exc:
                        logger.error(
                            "Failed to materialize recovery checkpoint weights after iteration failure: %s",
                            exc,
                        )
                        raise iteration_error from exc

                if checkpoint_delta_cpu is None:
                    delta = getattr(params, "delta_weights", None)
                    if delta is not None:
                        try:
                            checkpoint_delta_cpu = (
                                cp.asnumpy(delta) if isinstance(delta, cp.ndarray) else np.asarray(delta)
                            )
                            self._last_good_delta_cpu = checkpoint_delta_cpu
                        except Exception as exc:
                            logger.warning(
                                "Failed to materialize recovery checkpoint delta_weights after iteration failure: %s",
                                exc,
                            )

                # Hard-reset Ray resources and retry this iteration from the last checkpoint.
                self._recover_from_iteration_timeout(topology, params)

                # Refresh worker list after recovery.
                if not self.worker_manager.workers:
                    raise RuntimeError("Ray worker recovery completed but no workers were initialized")
                continue
            break

        if results is None:
            raise RuntimeError("Ray color iteration completed without results")
        
        # All workers should report success
        for result in results:
            if result['status'] != 'success':
                raise RuntimeError(f"Worker {result['worker_id']} failed")
        
        # Aggregate lightweight metrics and avoid per-iteration weight transfer
        total_samples = sum(r.get('samples_processed', 0) for r in results)
        weight_change_norm = results[0].get('weight_change_norm', 0.0) if results else 0.0

        if self.verbose:
            end_time = time.time()
            logger.info(
                f"Distributed color processing - ||dW||: {float(weight_change_norm):.8f} in {end_time - start_time:.2f}s"
            )

        if enable_recovery:
            # Update last-known-good checkpoint for future recovery attempts.
            # Note: this triggers GPU->CPU transfers and should only run when recovery is enabled.
            try:
                final_weights, final_delta = self.worker_manager.get_final_weights()
                self._last_good_weights_cpu = final_weights
                self._last_good_delta_cpu = final_delta
            except Exception as exc:
                logger.warning("Failed to refresh Ray recovery checkpoint weights: %s", exc)

        return {
            'update_type': 'inplace',
            'weight_change_norm': float(weight_change_norm),
            'samples_processed': int(total_samples)
        }

    def _recover_from_iteration_timeout(self, topology, params) -> None:
        """
        Best-effort hard reset after a suspected Ray/NCCL hang.

        Strategy:
        - Tear down workers/collective group via worker_manager.cleanup() (timeout-safe).
        - Reinitialize workers and redistribute data using the cached data_source.
        """
        if self._data_source is None:
            raise RuntimeError(
                "Cannot recover from Ray iteration timeout: no data_source cached. "
                "Ensure RayColorsProcessor.initialize(...) was called with a data_source."
            )

        logger.warning(
            "Ray color iteration watchdog triggered; performing hard reset and retrying."
        )

        if self._last_good_weights_cpu is None:
            raise RuntimeError(
                "Cannot recover from Ray iteration timeout: no CPU checkpoint weights are available."
            )

        try:
            self.worker_manager.cleanup(shutdown_ray=False)
        except Exception as exc:
            logger.warning("Ray worker-manager cleanup failed during recovery: %s", exc)

        # Use a fresh collective group name for each recovery attempt to avoid
        # colliding with stale NCCL communicator state from the failed attempt.
        self._recovery_attempt += 1
        next_group_name = (
            f"{self._base_collective_group_name}_recover_{self._recovery_attempt}"
        )
        self.collective_group_name = next_group_name
        self.worker_manager.collective_group_name = next_group_name
        logger.warning(
            "Ray recovery attempt %d: reinitializing with collective group '%s'.",
            self._recovery_attempt,
            next_group_name,
        )

        # Reinitialize worker manager + data distribution.
        self.worker_manager.initialize(
            self._last_good_weights_cpu,
            topology,
            params,
            self._data_source,
        )
    
    def cleanup(self):
        """Clean up Ray resources"""
        if self._progress_tracker is not None:
            should_kill_tracker = False
            try:
                has_live_workers = bool(
                    hasattr(self, "worker_manager")
                    and getattr(self.worker_manager, "workers", None)
                )
                should_kill_tracker = bool(ray.is_initialized() and has_live_workers)
            except Exception:
                should_kill_tracker = False

            # Avoid ray.kill() when Ray is not initialized or workers are already gone.
            # Calling ray.kill() in that state can trigger a fresh local Ray instance and
            # then fail on stale actor handles.
            if should_kill_tracker:
                try:
                    ray.kill(self._progress_tracker)
                except Exception:
                    pass
            self._progress_tracker = None
        # Delegate cleanup to worker manager
        if hasattr(self, 'worker_manager'):
            self.worker_manager.cleanup()
        
        logger.info("[Ray] Color processor cleanup completed")

    def get_current_weights(self):
        """
        Fetch current SOM weights from the first worker when needed (e.g., MST updates).
        Avoids per-iteration transfers; use sparingly.
        """
        if not hasattr(self, 'worker_manager') or not self.worker_manager.workers:
            raise RuntimeError("No workers initialized")
        result = self.worker_manager.get_gpu_weights()

        # Ray serializes CuPy arrays when they cross back to the driver.
        # Rewrap so downstream code continues working with GPU data.
        if isinstance(result, tuple):
            weights, delta = result
            weights_gpu = cp.asarray(weights)
            delta_gpu = cp.asarray(delta) if delta is not None else None
            return weights_gpu if delta_gpu is None else (weights_gpu, delta_gpu)

        return cp.asarray(result)
