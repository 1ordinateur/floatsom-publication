"""
Base infrastructure for Ray worker management with CPU-GPU pipeline
Implements DRY principle by consolidating all Ray worker initialization
"""

import os
import re
import math
import socket
import uuid
import shutil
import concurrent.futures
import ray
from ray.util import collective
import logging
from typing import List, Optional, Dict, Any, Tuple
from ..processing_params import AsyncLoadingConfig
import cupy as cp
import numpy as np
import psutil
from pathlib import Path
import time
import threading
from ...floatsom_params import ProcessingConfig
from ...data.fast_array_store import FastArrayStore
from ...data.cpugpu_fast_loader import CPUGPUFastLoader
from ...data.zarr_utils import open_array_read

logger = logging.getLogger(__name__)


class RayWorkerManager:
    """
    Base class for managing Ray workers with CPU-GPU pipeline support
    Follows Single Responsibility Principle: manages worker lifecycle only
    """
    
    def __init__(self, num_gpus: int, ray_config: Any, 
                 processing_config: Optional[ProcessingConfig] = None):
        """
        Initialize the Ray worker manager with CPU-GPU pipeline
        
        Args:
            num_gpus: Number of GPUs to use (None for auto-detect)
            ray_config: RayConfig object containing Ray configuration
            processing_config: Processing configuration
        """
        self.num_gpus = num_gpus
        # Chunk size must be provided in ray_config
        if ray_config and ray_config.chunk_size:
            self.chunk_size = ray_config.chunk_size
            self.target_chunk_size = self.chunk_size
            self.loader_chunk_size = self.chunk_size
            self.sampling_fraction = 1.0
        else:
            raise ValueError("chunk_size must be provided in RayConfig")
        
        # Store ray_config directly
        self.ray_config = ray_config
        
        # Store config for worker initialization
        self.worker_config = ray_config if ray_config else None

        self.processing_config = processing_config

        self.temp_root = self.ray_config.storage_path 
        self.hostname = socket.gethostname()
        node_root = os.path.join(self.temp_root, self.hostname)
        self.node_temp_root = self._ensure_directory(node_root)
        self.manager_temp_root = self._ensure_directory(os.path.join(self.node_temp_root, "manager"))
        self._allocated_temp_dirs: List[str] = []

        # Use collective_group_name from params or ray_config
        if self.ray_config.collective_group_name:
            self.collective_group_name = self.ray_config.collective_group_name
        else:
            self.collective_group_name = 'som_training'
        
        # Worker management
        self.workers = []
        self.collective_group_created = False
        
        # Data distribution state
        self.data_distributed = False
        self.worker_metadata = []
        self.total_samples = 0
        self.n_features = 0
        self.worker_placement_plan: List[Dict[str, Any]] = []
        self._pending_node_ram_stage_leaders: List[Any] = []
        
        # Async loading state
        self.async_loading_active = False
        self.distribution_futures = None
        self.monitor_thread = None

        # Populated during `initialize()` to support benchmark timing diagnostics.
        self.last_staging_stats: Optional[Dict[str, Any]] = None
        self.last_iteration_timing_stats: Optional[Dict[str, Any]] = None
        self.iteration_timing_history: List[Dict[str, Any]] = []
        
        logger.info(f"RayWorkerManager initialized with {num_gpus} GPUs, target chunk size: {self.chunk_size} samples")

    @staticmethod
    def _ensure_directory(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    def _allocate_temp_dir(self, prefix: str) -> str:
        unique_id = uuid.uuid4().hex
        path = os.path.join(self.manager_temp_root, f"{prefix}_{unique_id}")
        os.makedirs(path, exist_ok=True)
        self._allocated_temp_dirs.append(path)
        return path

    def _ray_config_value(self, key: str, default: Any = None) -> Any:
        if isinstance(self.ray_config, dict):
            return self.ray_config.get(key, default)
        if self.ray_config is None:
            return default
        return getattr(self.ray_config, key, default)

    def _resolve_phase_timeout_s(self, phase: str, default_s: float = 300.0) -> float:
        phase_key = re.sub(r"[^a-zA-Z0-9]+", "_", str(phase or "")).strip("_").lower()
        candidates = []
        if phase_key:
            candidates.append(f"{phase_key}_timeout_s")
        candidates.append("ray_wait_timeout_s")
        if "iteration" in phase_key:
            candidates.append("iteration_timeout_s")

        for key in candidates:
            raw_value = self._ray_config_value(key, None)
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                return value

        return float(default_s)

    def wait_for_worker_futures(
        self,
        futures: List[Any],
        *,
        phase: str,
        timeout_s: Optional[float] = None,
        poll_interval_s: float = 5.0,
    ) -> List[Any]:
        if not futures:
            return []

        resolved_timeout_s = self._resolve_phase_timeout_s(phase, default_s=30000.0)
        if timeout_s is not None:
            try:
                explicit_timeout_s = float(timeout_s)
            except (TypeError, ValueError):
                explicit_timeout_s = 0.0
            if explicit_timeout_s > 0.0:
                resolved_timeout_s = explicit_timeout_s
        poll_s = max(0.1, min(float(poll_interval_s), resolved_timeout_s))

        pending = list(futures)
        future_to_index = {future: idx for idx, future in enumerate(futures)}
        results: List[Any] = [None] * len(futures)
        start_t = time.monotonic()

        while pending:
            elapsed_s = time.monotonic() - start_t
            remaining_s = resolved_timeout_s - elapsed_s
            if remaining_s <= 0:
                pending_indices = sorted(
                    future_to_index.get(ref, -1) for ref in pending
                )
                raise TimeoutError(
                    f"Timed out waiting for Ray futures in phase='{phase}' after "
                    f"{resolved_timeout_s:.1f}s; pending worker indices={pending_indices}"
                )

            ready, pending = ray.wait(
                pending,
                num_returns=1,
                timeout=min(poll_s, remaining_s),
            )
            if not ready:
                continue

            for ref in ready:
                idx = future_to_index[ref]
                try:
                    results[idx] = ray.get(ref)
                except Exception as exc:
                    raise RuntimeError(
                        f"Ray future failed in phase='{phase}' for worker index {idx}"
                    ) from exc

        return results

    def ensure_distribution_ready(self, timeout: Optional[float] = None) -> bool:
        """
        Ensure worker-side data-distribution setup completed before iterations start.
        """
        if not self.distribution_futures:
            if self.total_samples > 0:
                self.data_distributed = True
            return True

        try:
            results = self.wait_for_worker_futures(
                self.distribution_futures,
                phase="async_distribution_setup",
                timeout_s=timeout,
            )
            self.worker_metadata = list(results)
            self.distribution_futures = None
            self.data_distributed = True
            self._cleanup_pending_node_ram_stage_leaders()
            return True
        except Exception:
            self.data_distributed = False
            self._cleanup_pending_node_ram_stage_leaders(suppress_exceptions=True)
            raise

    def _cleanup_pending_node_ram_stage_leaders(self, *, suppress_exceptions: bool = False) -> None:
        """Release node-local staged shards retained on worker leaders."""
        if not self._pending_node_ram_stage_leaders:
            return

        leader_futures: List[Tuple[Any, Any]] = []
        failed_leaders: List[Any] = []
        first_error: Optional[Exception] = None

        for leader in list(self._pending_node_ram_stage_leaders):
            try:
                future = leader.clear_node_staged_shards.remote()
            except Exception as exc:
                failed_leaders.append(leader)
                if suppress_exceptions:
                    logger.exception("Node-local RAM staged-shard cleanup dispatch failed")
                    continue
                if first_error is None:
                    first_error = exc
                continue
            leader_futures.append((leader, future))

        for leader, future in leader_futures:
            try:
                self.wait_for_worker_futures(
                    [future],
                    phase="node_ram_stage_cleanup",
                    timeout_s=self._resolve_phase_timeout_s(
                        "node_ram_stage_cleanup",
                        default_s=60.0,
                    ),
                )
            except Exception as exc:
                failed_leaders.append(leader)
                if suppress_exceptions:
                    logger.exception("Node-local RAM staged-shard cleanup failed")
                    continue
                if first_error is None:
                    first_error = exc

        self._pending_node_ram_stage_leaders = failed_leaders
        if first_error is not None:
            raise first_error

    @staticmethod
    def _node_affinity_options(node_resource_key: Optional[str]) -> Dict[str, Any]:
        """
        Return Ray actor `.options()` kwargs to keep an actor on a specific node.

        Uses the built-in per-node resource key ("node:<ip>" or equivalent).
        """
        if not node_resource_key:
            return {}
        return {"resources": {str(node_resource_key): 0.001}}

    def _query_node_cpu_affinity(self, node_resource_key: str) -> Optional[int]:
        """
        Best-effort query for the CPU affinity (cpuset) size on a specific Ray node.

        This matters on HPC systems where Ray may report the machine-wide CPU count
        even when the job is restricted to a subset of cores (via cpusets/cgroups).
        """
        if not node_resource_key:
            return None

        @ray.remote(num_cpus=0)
        def _detect_affinity() -> int:
            import os

            try:
                return len(os.sched_getaffinity(0))
            except Exception:
                return max(1, os.cpu_count() or 1)

        try:
            return int(
                ray.get(
                    _detect_affinity.options(resources={str(node_resource_key): 0.001}).remote()
                )
            )
        except Exception as exc:
            logger.debug(
                "Failed to query node CPU affinity for %s (%s)",
                node_resource_key,
                exc,
            )
            return None

    def _plan_worker_cpu_reservations(self, num_workers: int) -> List[Dict[str, Any]]:
        """
        Plan per-worker CPU reservations, dividing CPUs equally per node.

        Leaves 1 CPU spare per node when possible to avoid starving Ray/OS
        housekeeping. Returns a list (length=num_workers) of dicts containing:
          - node_resource_key: Ray node:<...> placement key or None
          - hostname: node hostname (for logging)
          - num_cpus: integer CPU reservation for the worker actor
        """
        if num_workers <= 0:
            return []

        cluster_resources = ray.cluster_resources()
        cluster_cpus = int(cluster_resources.get("CPU", 0) or 0)
        cluster_cpu_budget = max(1, cluster_cpus - 1)
        fallback_cpus_per_worker = max(1, cluster_cpu_budget // max(1, num_workers))
        fallback_plan = [
            {"node_resource_key": None, "hostname": None, "num_cpus": fallback_cpus_per_worker}
            for _ in range(num_workers)
        ]

        try:
            nodes = [node for node in ray.nodes() if node.get("Alive", False)]
        except Exception as exc:
            logger.warning(
                "Failed to query Ray nodes for per-node CPU allocation (%s); using cluster-wide CPU split",
                exc,
            )
            return fallback_plan

        gpu_nodes: List[Dict[str, Any]] = []
        for node in nodes:
            resources = node.get("Resources", {}) or {}
            node_gpus = int(resources.get("GPU", 0) or 0)
            if node_gpus <= 0:
                continue

            node_cpus = int(resources.get("CPU", 0) or 0)
            hostname = node.get("NodeManagerHostname") or node.get("NodeManagerAddress") or "unknown"
            node_resource_keys = [
                key for key in resources.keys()
                if isinstance(key, str) and key.startswith("node:")
            ]
            node_resource_key = node_resource_keys[0] if node_resource_keys else None

            affinity_cpus = None
            if node_resource_key:
                affinity_cpus = self._query_node_cpu_affinity(node_resource_key)
            effective_cpus = node_cpus
            if affinity_cpus is not None:
                effective_cpus = min(node_cpus, int(affinity_cpus))

            # Best-effort: keep 1 CPU per node free for system/Ray overhead.
            cpu_budget = max(1, effective_cpus - 1)
            max_workers_on_node = min(node_gpus, cpu_budget)

            gpu_nodes.append(
                {
                    "hostname": hostname,
                    "cpus": effective_cpus,
                    "cpu_budget": cpu_budget,
                    "gpu_slots": node_gpus,
                    "worker_slots": max_workers_on_node,
                    "node_resource_key": node_resource_key,
                }
            )

        if not gpu_nodes:
            logger.warning("No GPU-capable Ray nodes detected; using cluster-wide CPU split")
            return fallback_plan

        if any(item.get("node_resource_key") is None for item in gpu_nodes):
            logger.warning(
                "Ray node resource keys (node:<...>) not available for placement; using cluster-wide CPU split"
            )
            return fallback_plan

        gpu_nodes.sort(key=lambda item: (str(item.get("hostname")), str(item.get("node_resource_key"))))

        total_worker_slots = sum(int(item["worker_slots"]) for item in gpu_nodes)
        if total_worker_slots < num_workers:
            # Not enough CPU budget to keep 1 CPU spare per node *and* schedule all GPU workers.
            # Relax the spare-CPU constraint rather than deadlocking scheduling.
            logger.warning(
                "Per-node CPU budgeting (%d slots after reserving 1 CPU/node) cannot place %d worker(s); "
                "relaxing spare-CPU reservation for scheduling",
                total_worker_slots,
                num_workers,
            )
            for item in gpu_nodes:
                cpu_budget = max(1, int(item["cpus"]))
                item["cpu_budget"] = cpu_budget
                item["worker_slots"] = min(int(item["gpu_slots"]), cpu_budget)
            total_worker_slots = sum(int(item["worker_slots"]) for item in gpu_nodes)

        if total_worker_slots < num_workers:
            logger.warning(
                "Ray cluster CPU capacity (%d worker slot(s)) is insufficient for %d GPU worker(s); "
                "using cluster-wide CPU split and leaving placement to Ray",
                total_worker_slots,
                num_workers,
            )
            return fallback_plan

        remaining_slots = [int(item["worker_slots"]) for item in gpu_nodes]
        workers_per_node = [0 for _ in gpu_nodes]

        cursor = 0
        while sum(workers_per_node) < num_workers:
            if remaining_slots[cursor] > 0:
                remaining_slots[cursor] -= 1
                workers_per_node[cursor] += 1
            cursor = (cursor + 1) % len(gpu_nodes)

        cpus_per_worker_by_node: List[int] = []
        for item, count in zip(gpu_nodes, workers_per_node):
            if count <= 0:
                cpus_per_worker_by_node.append(0)
                continue
            cpu_budget = int(item["cpu_budget"])
            cpus_per_worker_by_node.append(max(1, cpu_budget // count))

        plan: List[Dict[str, Any]] = []
        for node_idx, worker_count in enumerate(workers_per_node):
            if worker_count <= 0:
                continue
            node_info = gpu_nodes[node_idx]
            num_cpus = int(cpus_per_worker_by_node[node_idx])
            for _ in range(worker_count):
                plan.append(
                    {
                        "hostname": node_info["hostname"],
                        "num_cpus": max(1, num_cpus),
                        "node_resource_key": node_info.get("node_resource_key"),
                    }
                )

        return plan

    def initialize_workers(self, worker_class, chunk_size=None) -> List:
        """
        Initialize Ray GPU workers with CPU-GPU fast loading

        Args:
            chunk_size: Chunk size in number of samples (optional override)
        
        Returns:
            List of worker actors
        """
        if self.workers:
            return self.workers  # Already initialized
        
        # Use provided chunk_size if given, otherwise keep the configured chunk_size
        if chunk_size is not None:
            self.chunk_size = chunk_size
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            logger.info("Initializing Ray - attempting to connect to existing cluster")
            ray_address = getattr(self.ray_config, "ray_address", None) or "auto"
            
            # Dynamically determine the project root based on current file location
            import os
            import sys
            current_file = os.path.abspath(__file__)
            # Navigate up: ray_ops -> processing -> floatsom -> project_root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
            
            # Set up runtime environment with Python path
            runtime_env = {
                "env_vars": {
                    "PYTHONPATH": f"{project_root}:{os.environ.get('PYTHONPATH', '')}"
                }
            }
            
            try:
                # 'auto' will connect to existing cluster if available
                ray.init(address=ray_address, runtime_env=runtime_env)
                logger.info(
                    "Successfully initialized Ray with address=%s and PYTHONPATH including %s",
                    ray_address,
                    project_root,
                )
            except Exception:
                if ray_address not in ("", "auto"):
                    raise
                # Fall back to local Ray if no cluster found when using default address
                logger.warning("No existing Ray cluster found with address=%s, starting local Ray instance", ray_address)
                ray.init(runtime_env=runtime_env)
                logger.info(f"Started local Ray instance with PYTHONPATH including {project_root}")

        resources = ray.cluster_resources()
        available_gpus = int(resources.get('GPU', 0))
        available_cpus = int(resources.get('CPU', 0))

        if self.num_gpus is not None and available_gpus < self.num_gpus:
            raise RuntimeError(
                f"Requested {self.num_gpus} GPU(s) but only {available_gpus} available. "
                "Adjust the Ray GPU count or ensure the cluster has the requested resources."
            )

        # Auto-detect GPUs if not specified
        if self.num_gpus is None:
            # Log cluster resources for debugging
            logger.info(f"Ray cluster resources detected: CPUs={available_cpus}, GPUs={available_gpus}")
            
            # Also log node information for multi-node awareness
            nodes = ray.nodes()
            alive_nodes = [n for n in nodes if n.get('Alive', False)]
            logger.info(f"Ray cluster has {len(alive_nodes)} alive node(s)")
            for i, node in enumerate(alive_nodes):
                node_gpus = int(node.get('Resources', {}).get('GPU', 0))
                node_cpus = int(node.get('Resources', {}).get('CPU', 0))
                logger.info(f"  Node {i+1}: {node.get('NodeManagerHostname', 'unknown')} - CPUs={node_cpus}, GPUs={node_gpus}")
            
            if available_gpus == 0:
                raise RuntimeError("No GPUs detected in Ray cluster")
            self.num_gpus = int(available_gpus)
            logger.info(f"Auto-detected {self.num_gpus} total GPUs from Ray cluster")
        else:
            # Log cluster resources for debugging
            logger.info(f"Ray cluster resources: CPUs={available_cpus}, GPUs={available_gpus}")
        
        # Create workers
        self.workers = []
        num_workers_to_create = int(self.num_gpus) if self.num_gpus else 1

        worker_cpu_plan = self._plan_worker_cpu_reservations(num_workers_to_create)
        self.worker_placement_plan = [dict(plan) for plan in worker_cpu_plan]
        if len(worker_cpu_plan) != num_workers_to_create:
            logger.warning(
                "Worker CPU plan length mismatch (%d vs %d); falling back to 1 CPU/worker",
                len(worker_cpu_plan),
                num_workers_to_create,
            )
            worker_cpu_plan = [
                {"node_resource_key": None, "hostname": None, "num_cpus": 1}
                for _ in range(num_workers_to_create)
            ]
            self.worker_placement_plan = [dict(plan) for plan in worker_cpu_plan]

        for worker_id, plan in enumerate(worker_cpu_plan):
            num_cpus = int(plan.get("num_cpus", 1) or 1)
            node_resource_key = plan.get("node_resource_key")
            hostname = plan.get("hostname")

            options: Dict[str, Any] = {"num_cpus": max(1, num_cpus)}
            options.update(self._node_affinity_options(node_resource_key))

            logger.info(
                "Creating Ray worker %d with num_cpus=%d%s",
                worker_id,
                options["num_cpus"],
                f" on node={hostname}" if hostname else "",
            )
            print(
                f"Creating Ray worker {worker_id} with num_cpus={options['num_cpus']}"
                f"{f' on node={hostname}' if hostname else ''}",
                flush=True,
            )

            worker = worker_class.options(**options).remote(
                worker_id=worker_id,
                num_gpus=self.num_gpus,
                worker_config=self.worker_config,
                gpu_id=None,  # Let Ray handle GPU assignment
                chunk_size=chunk_size,  # Pass optimal chunk size
            )
            self.workers.append(worker)
        
        logger.info(f"Created {len(self.workers)} Ray GPU workers with CPU-GPU fast loading")
        logger.info(f"Workers configured with effective chunk size: {self.chunk_size} samples")
        
        self._initialize_collective_group()
        
        return self.workers

    def _configure_sampling_parameters(self, params: Any, dataset_samples: int) -> None:
        """
        Determine loader chunk sizing and sampling fraction from training parameters.
        """
        sampling_fraction = 1.0
        sampling_method = "full"
        sampling_cfg = getattr(params, "sampling_config", None)
        if sampling_cfg:
            sampling_method = getattr(sampling_cfg, "method", "full") or "full"
            sampling_method = str(sampling_method).lower()
            if sampling_method == "random":
                samples_per_epoch = getattr(sampling_cfg, "samples_per_epoch", None)
                target_proportion = getattr(sampling_cfg, "target_proportion", None)
                if samples_per_epoch is not None and dataset_samples:
                    sampling_fraction = samples_per_epoch / dataset_samples
                elif target_proportion is not None:
                    sampling_fraction = float(target_proportion)
        sampling_fraction = max(min(sampling_fraction, 1.0), 1e-6)

        if sampling_method == "random":
            self.loader_chunk_size = self.target_chunk_size
        elif sampling_fraction < 1.0:
            inflated = math.ceil(self.target_chunk_size / sampling_fraction)
            if dataset_samples:
                inflated = min(inflated, dataset_samples)
            inflated = max(inflated, self.target_chunk_size)
            self.loader_chunk_size = int(inflated)
        else:
            self.loader_chunk_size = self.target_chunk_size

        self.sampling_fraction = sampling_fraction
        self.sampling_method = sampling_method
        logger.info(
            "Configured sampling: method=%s fraction=%.4f, effective chunk=%d, loader chunk=%d",
            sampling_method,
            self.sampling_fraction,
            self.target_chunk_size,
            self.loader_chunk_size,
        )
    
    def _initialize_collective_group(self):
        """Initialize NCCL collective group for multi-GPU communication"""
        if self.collective_group_created:
            return

        def _destroy_collective_group_everywhere() -> None:
            if self.workers:
                try:
                    destroy_futures = [
                        worker.destroy_collective_group.remote(self.collective_group_name)
                        for worker in self.workers
                    ]
                    ray.get(destroy_futures, timeout=10.0)
                except Exception as exc:
                    logger.debug(
                        "Best-effort worker collective group destroy failed for %s: %s",
                        self.collective_group_name,
                        exc,
                    )

            try:
                collective.destroy_collective_group(group_name=self.collective_group_name)
            except (RuntimeError, ValueError, AttributeError):
                # Group doesn't exist or destroy_collective_group not available
                pass

        last_error: Optional[RuntimeError] = None
        for attempt in range(2):
            try:
                collective.create_collective_group(
                    self.workers,
                    world_size=len(self.workers),
                    ranks=list(range(len(self.workers))),
                    backend="nccl",
                    group_name=self.collective_group_name,
                )
                self.collective_group_created = True
                logger.info(
                    "Initialized NCCL collective group: %s",
                    self.collective_group_name,
                )
                return
            except RuntimeError as exc:
                last_error = exc
                message = str(exc)
                if (
                    "Trying to initialize a group twice" not in message
                    and "initialize a group twice" not in message
                ):
                    raise

                logger.warning(
                    "Collective group %s already exists; destroying and recreating (attempt %d/2).",
                    self.collective_group_name,
                    attempt + 1,
                )
                _destroy_collective_group_everywhere()

        raise RuntimeError(
            f"Failed to initialize NCCL collective group '{self.collective_group_name}' "
            f"after cleanup attempts: {last_error}"
        )

    def _warmup_collective_group(self) -> Optional[Dict[str, Any]]:
        """
        Run a small NCCL collective warmup across all workers (optional).

        NCCL communicators are often lazily initialized on the first collective
        call; this warmup moves that cost into setup so iteration 0 timing is
        representative.
        """
        if not self.workers:
            return None
        if not self.collective_group_created:
            return None

        cfg = self.ray_config
        warmup_iters = 1
        tensor_elements = 1024
        if isinstance(cfg, dict):
            warmup_iters = int(cfg.get("collective_warmup_iters", warmup_iters) or 0)
            tensor_elements = int(
                cfg.get("collective_warmup_tensor_elements", tensor_elements) or tensor_elements
            )
        else:
            warmup_iters = int(getattr(cfg, "collective_warmup_iters", warmup_iters) or 0)
            tensor_elements = int(
                getattr(cfg, "collective_warmup_tensor_elements", tensor_elements) or tensor_elements
            )

        if warmup_iters <= 0:
            return None

        tensor_elements = max(1, tensor_elements)
        logger.info(
            "Warming up NCCL collectives: group=%s workers=%d iters=%d elements=%d",
            self.collective_group_name,
            len(self.workers),
            warmup_iters,
            tensor_elements,
        )

        warmup_start = time.perf_counter()
        futures = [
            worker.warmup_collectives.remote(self.collective_group_name, warmup_iters, tensor_elements)
            for worker in self.workers
        ]
        results = self.wait_for_worker_futures(
            futures,
            phase="collective_warmup",
            timeout_s=self._resolve_phase_timeout_s("collective_warmup", default_s=120.0),
        )
        warmup_wall_s = float(time.perf_counter() - warmup_start)

        worker_elapsed = []
        for result in results:
            if not isinstance(result, dict):
                continue
            elapsed = result.get("elapsed_s")
            if isinstance(elapsed, (int, float)):
                worker_elapsed.append(float(elapsed))

        stats: Dict[str, Any] = {
            "collective_warmup_s": warmup_wall_s,
            "collective_warmup_iters": int(warmup_iters),
            "collective_warmup_tensor_elements": int(tensor_elements),
        }
        if worker_elapsed:
            stats["collective_warmup_worker_elapsed_s_min"] = float(min(worker_elapsed))
            stats["collective_warmup_worker_elapsed_s_max"] = float(max(worker_elapsed))
            stats["collective_warmup_worker_elapsed_s_mean"] = float(
                sum(worker_elapsed) / len(worker_elapsed)
            )

        logger.info(
            "[RayWarmup] wall_s=%.3f worker_elapsed_s_max=%s",
            warmup_wall_s,
            f"{max(worker_elapsed):.3f}" if worker_elapsed else "n/a",
        )
        return stats
        
    
    def cleanup(self, *, shutdown_ray: bool = True):
        """Clean up all Ray resources"""
        logger.info("Cleaning up RayWorkerManager...")

        ray_initialized = False
        try:
            ray_initialized = bool(ray.is_initialized())
        except Exception:
            ray_initialized = False

        safe_cleanup = True
        cleanup_config = None
        if self.processing_config is not None:
            cleanup_config = getattr(self.processing_config, "cleanup_config", None)
        if cleanup_config is not None:
            safe_cleanup = bool(getattr(cleanup_config, "safe_cleanup", True))

        profile_enabled = False
        profile_config = None
        if self.processing_config is not None:
            profile_config = getattr(self.processing_config, "worker_profile_config", None)
        if profile_config is not None:
            profile_enabled = bool(getattr(profile_config, "enabled", False))
        
        # Destroy collective group if it was created
        if self.collective_group_created and ray_initialized:
            try:
                collective.destroy_collective_group(group_name=self.collective_group_name)
                logger.info(f"Destroyed collective group: {self.collective_group_name}")
            except (RuntimeError, ValueError, AttributeError):
                # Group might not exist or destroy method not available
                pass
        elif self.collective_group_created and not ray_initialized:
            logger.debug(
                "Skipping collective group destroy for %s because Ray is not initialized.",
                self.collective_group_name,
            )
        
        self._cleanup_pending_node_ram_stage_leaders(suppress_exceptions=True)
        self._pending_node_ram_stage_leaders = []

        # Clean up workers
        if self.workers:
            if ray_initialized:
                if safe_cleanup:
                    cleanup_futures = []
                    for worker in self.workers:
                        try:
                            cleanup_futures.append(worker.cleanup.remote())
                        except Exception:
                            # Worker might already be dead
                            pass

                    # Wait for cleanup with timeout
                    try:
                        ray.get(cleanup_futures, timeout=5.0)
                    except Exception:
                        pass
                else:
                    if profile_enabled:
                        profile_futures = []
                        for worker in self.workers:
                            try:
                                profile_futures.append(worker.finalize_profile.remote())
                            except Exception:
                                pass
                        try:
                            ray.get(profile_futures, timeout=5.0)
                        except Exception:
                            pass
                    logger.info("Fast cleanup requested; skipping worker cleanup calls.")
                
                # Kill workers
                for worker in self.workers:
                    try:
                        ray.kill(worker)
                    except Exception:
                        pass
            else:
                logger.info(
                    "Ray is not initialized; skipping remote cleanup for %d stale worker handle(s).",
                    len(self.workers),
                )
        
        # Clear references
        self.workers = []
        self.worker_metadata = []
        self.collective_group_created = False
        self.data_distributed = False
        # Stop async monitor if running
        try:
            self.async_loading_active = False
            if hasattr(self, 'monitor_thread') and self.monitor_thread is not None:
                if self.monitor_thread.is_alive():
                    self.monitor_thread.join(timeout=2.0)
        except Exception:
            pass
        # Clear async distribution futures
        self.distribution_futures = None
        self._cleanup_pending_node_ram_stage_leaders(suppress_exceptions=True)
        self._pending_node_ram_stage_leaders = []
        # Remove temporary directories created during distribution
        for temp_dir in reversed(self._allocated_temp_dirs):
            shutil.rmtree(temp_dir, ignore_errors=True)
        self._allocated_temp_dirs.clear()
        # Attempt to remove manager directory if empty
        if self.manager_temp_root and os.path.isdir(self.manager_temp_root):
            try:
                os.rmdir(self.manager_temp_root)
            except OSError:
                pass
        # Do not remove node_temp_root or storage_path directly to avoid deleting user data

        if shutdown_ray and ray_initialized:
            ray.shutdown()
        
        logger.info("RayWorkerManager cleanup complete")
    
    def initialize(self, som_weights, topology, params, data_source):
        """
        One-time initialization called by FloatSOM's _initialize_with_data_source().
        Initializes workers and distributes data as Zarr files.
        Uses CPU-GPU fast loading for data access.
        
        Args:
            som_weights: Initial SOM weights
            topology: SOM topology
            params: Training parameters containing data source
            data_source: Data source object with file reference
        """
        # 1. Get source info from data source FIRST - can be file or array
        source_info = self.get_dataset_metadata(data_source)
        
        if not source_info:
            raise ValueError("No data source found")
        
        # 2. Optionally adjust chunk size based on dimensions if not already configured
        n_samples = source_info.get('n_samples', 0)
        n_features = source_info.get('n_features', 1)
        
        # Only recalculate if chunk_size wasn't explicitly configured
        # Check if we should adjust chunk size based on data dimensions
        if self.chunk_size == 1000000 and n_samples > 0 and n_features > 0:  # 1M was the old default
            # Calculate optimal chunk size locally without modifying global
            TARGET_MEMORY_ELEMENTS = 750_000_000  # Same as in processing_params.py
            optimal_chunk_size = max(1000, TARGET_MEMORY_ELEMENTS // n_features)
            self.chunk_size = min(optimal_chunk_size, n_samples) if n_samples > 0 else optimal_chunk_size
            logger.info(f"Auto-adjusted chunk size to {self.chunk_size} for dataset with {n_samples} samples and {n_features} features")
        else:
            logger.info(f"Using configured chunk size: {self.chunk_size} samples")
        
        # Record effective chunk sizing before configuring sampling
        self.target_chunk_size = self.chunk_size
        self.loader_chunk_size = self.target_chunk_size
        self._configure_sampling_parameters(params, n_samples)

        # 3. NOW initialize workers with the correct chunk size
        if not self.workers:
            worker_class = self._get_worker_class()
            self.initialize_workers(worker_class, self.target_chunk_size)

        # Reset per-run diagnostics state (benchmarks may reuse the same manager).
        self.last_staging_stats = None
        self.last_iteration_timing_stats = None
        self.iteration_timing_history = []

        wipe_enabled = True
        if isinstance(self.ray_config, dict):
            wipe_enabled = bool(self.ray_config.get("wipe_local_storage_on_start", True))
        else:
            wipe_enabled = bool(getattr(self.ray_config, "wipe_local_storage_on_start", True))

        if wipe_enabled:
            wipe_futures = [worker.wipe_local_storage.remote() for worker in self.workers]
            if wipe_futures:
                self.wait_for_worker_futures(
                    wipe_futures,
                    phase="wipe_local_storage",
                )

        # Propagate sampling configuration to workers
        configure_futures = [
            worker.configure_chunk_sampling.remote(
                self.loader_chunk_size,
                self.sampling_fraction,
                getattr(self, "sampling_method", "full"),
                whole_chunk_random=bool(
                    getattr(getattr(params, "sampling_config", None), "whole_chunk_random", False)
                ),
            )
            for worker in self.workers
        ]
        if configure_futures:
            self.wait_for_worker_futures(
                configure_futures,
                phase="configure_chunk_sampling",
            )

        warmup_details = self._warmup_collective_group() or {}
        
        # If source is memory/array, it will be converted to FastArrayStore
        if source_info.get('type') == 'memory':
            logger.info("Detected array data source, will convert to FastArrayStore for high-performance loading")
        
        # 3. Always distribute data as FastArrayStore files to workers
        # (arrays will be converted to FastArrayStore inside this method)
        staging_start = time.perf_counter()
        staging_details = self._distribute_fast_array_once(source_info) or {}
        self.ensure_distribution_ready()
        staging_wall_s = time.perf_counter() - staging_start
        if not isinstance(staging_details, dict):
            staging_details = {}

        self.last_staging_stats = {
            "data_staging_s": float(staging_wall_s),
            **warmup_details,
            **staging_details,
        }
        self.data_distributed = True
        logger.info("Data distribution complete. Ready for high-performance FastArrayStore loading.")

        if self.last_staging_stats:
            parts: List[str] = []
            for key in sorted(self.last_staging_stats.keys()):
                value = self.last_staging_stats.get(key)
                if isinstance(value, float):
                    if key.endswith("_s"):
                        parts.append(f"{key}={value:.3f}s")
                    else:
                        parts.append(f"{key}={value:.3f}")
                else:
                    parts.append(f"{key}={value}")
            if parts:
                logger.info("[RayStaging] %s", " ".join(parts))
    
    def _get_worker_class(self):
        """
        Get the appropriate worker class for this processor.
        Subclasses must override this method.
        
        Returns:
            Worker class (Ray remote class)
        """
        from .workers.ray_batch_worker import RayBatchWorker
        from .workers.ray_color_worker import RayColorWorker

        if self.processing_config.method == 'batch':
            return RayBatchWorker
        elif self.processing_config.method == 'colors':
            return RayColorWorker
        else:
            raise ValueError(f"Invalid processing method: {self.processing_config.method}")  
    
    def get_dataset_metadata(self, data_source):
        """
        Extract the full dataset from params with source information.
        
        Args:
            params: Training parameters
            
        Returns:
            Tuple of (data, source_info) where source_info contains:
                - type: 'file' or 'memory'
                - path: file path if type is 'file'
                - format: 'zarr', 'npy', 'parquet', etc. if known
                - array: numpy/cupy array if type is 'memory'
            Returns (None, None) if no data found
        """
        # Check if data_source has array data directly
        if hasattr(data_source, 'data') and data_source.data is not None:
            # This is an array data source
            data = data_source.data
            if isinstance(data, (np.ndarray, cp.ndarray)):
                logger.warning(f"Detected array data source with shape {data.shape}")
                source_info = {
                    'type': 'memory',
                    'array': data if isinstance(data, np.ndarray) else data.get(),  # Convert cupy to numpy if needed
                    'format': 'array',
                    'n_samples': data.shape[0] if len(data.shape) > 0 else 0,
                    'n_features': data.shape[1] if len(data.shape) > 1 else 1
                }
                return source_info
        
        # Try to get data from various sources in params            
        file_path = str(data_source.get_reference())
        # Ray workers may run with a different working directory. Normalize to an
        # absolute path so multi-node jobs can open the same shared dataset.
        if "://" not in file_path:
            try:
                file_path = str(Path(file_path).expanduser().resolve())
            except Exception:
                file_path = os.path.abspath(file_path)
        file_format = 'auto'
        
        # Try to detect format from extension
        if file_path.endswith('.zarr'):
            file_format = 'zarr'
        elif file_path.endswith('.npy'):
            file_format = 'npy'
        elif file_path.endswith('.parquet'):
            file_format = 'parquet'
        
        # Get shape if available from data_source
        n_samples, n_features = 0, 1
        if hasattr(data_source, 'get_shape'):
            shape = data_source.get_shape()
            if shape:
                n_samples = shape[0] if len(shape) > 0 else 0
                n_features = shape[1] if len(shape) > 1 else 1
        
        source_info = {
            'type': 'file',
            'path': file_path,
            'format': file_format,
            'n_samples': n_samples,
            'n_features': n_features
        }
        return source_info

    def _distribute_zarr_to_worker_ram(self, zarr_path: str, *, n_samples: int) -> Dict[str, Any]:
        """
        Multi-node-safe RAM staging: each worker loads only its slice of the Zarr store
        into worker-local RAM (no `ray.put(z[:])` broadcast).
        """
        n_workers = len(self.workers)
        if n_workers <= 0:
            raise RuntimeError("No workers initialized")

        samples_per_worker = n_samples // n_workers if n_workers else n_samples
        distribution_futures = []
        for i, worker in enumerate(self.workers):
            start = i * samples_per_worker
            end = start + samples_per_worker if i < n_workers - 1 else n_samples
            distribution_futures.append(worker.use_zarr_ram_slice.remote(zarr_path, start, end))

        self.worker_metadata = self.wait_for_worker_futures(
            distribution_futures,
            phase="zarr_worker_ram_distribution",
            timeout_s=self._resolve_phase_timeout_s(
                "zarr_worker_ram_distribution",
                default_s=3600.0,
            ),
        )

        elapsed_values: List[float] = []
        throughput_values: List[float] = []
        for meta in self.worker_metadata:
            if not isinstance(meta, dict):
                continue
            elapsed = meta.get("elapsed_s")
            if isinstance(elapsed, (int, float)):
                elapsed_values.append(float(elapsed))
            throughput = meta.get("throughput_gb_s")
            if isinstance(throughput, (int, float)):
                throughput_values.append(float(throughput))

        details: Dict[str, Any] = {
            "data_staging_strategy": "zarr_to_worker_ram",
            "data_staging_async": False,
            "data_staging_workers": int(n_workers),
            "data_staging_worker_metadata_count": int(len(self.worker_metadata) if self.worker_metadata else 0),
        }

        if elapsed_values:
            details["data_staging_worker_elapsed_s_min"] = float(min(elapsed_values))
            details["data_staging_worker_elapsed_s_max"] = float(max(elapsed_values))
            details["data_staging_worker_elapsed_s_mean"] = float(sum(elapsed_values) / max(1, len(elapsed_values)))

        if throughput_values:
            details["data_staging_worker_throughput_gb_s_min"] = float(min(throughput_values))
            details["data_staging_worker_throughput_gb_s_max"] = float(max(throughput_values))
            details["data_staging_worker_throughput_gb_s_mean"] = float(
                sum(throughput_values) / max(1, len(throughput_values))
            )

        return details

    def _group_worker_spans_by_node(self, n_samples: int) -> List[Dict[str, Any]]:
        """Group per-worker global spans by Ray node for node-local RAM staging."""
        n_workers = len(self.workers)
        if n_workers <= 0:
            raise RuntimeError("No workers initialized")
        if len(self.worker_placement_plan) != n_workers:
            raise RuntimeError(
                "Worker placement metadata unavailable for node-local RAM staging"
            )

        spans = self._compute_worker_spans(int(n_samples), n_workers)
        grouped: Dict[str, Dict[str, Any]] = {}
        for worker_idx, ((start, end), placement) in enumerate(zip(spans, self.worker_placement_plan)):
            node_key = placement.get("node_resource_key")
            hostname = placement.get("hostname")
            if not node_key or not hostname:
                raise RuntimeError(
                    "Node-local RAM staging requires explicit node placement metadata for every worker"
                )
            group = grouped.setdefault(
                str(node_key),
                {
                    "node_resource_key": str(node_key),
                    "hostname": str(hostname),
                    "workers": [],
                },
            )
            group["workers"].append(
                {
                    "worker_idx": int(worker_idx),
                    "start": int(start),
                    "end": int(end),
                }
            )

        groups = []
        for node_key in sorted(grouped.keys()):
            group = grouped[node_key]
            workers = sorted(group["workers"], key=lambda item: item["worker_idx"])
            if workers:
                expected_start = int(workers[0]["start"])
                for item in workers:
                    if int(item["start"]) != expected_start:
                        raise RuntimeError(
                            "Node-local RAM staging requires contiguous worker placement per node; "
                            f"node={group['hostname']} worker_idx={item['worker_idx']} "
                            f"start={item['start']} expected_start={expected_start}"
                        )
                    expected_start = int(item["end"])
                node_start = int(workers[0]["start"])
                node_end = int(workers[-1]["end"])
            else:
                node_start = 0
                node_end = 0
            prepared_workers = []
            for item in workers:
                prepared_workers.append(
                    {
                        **item,
                        "local_start": int(item["start"]) - int(node_start),
                        "local_end": int(item["end"]) - int(node_start),
                    }
                )
            groups.append(
                {
                    "node_resource_key": group["node_resource_key"],
                    "hostname": group["hostname"],
                    "node_start": int(node_start),
                    "node_end": int(node_end),
                    "leader_worker_idx": int(workers[0]["worker_idx"]) if workers else -1,
                    "workers": prepared_workers,
                }
            )
        return groups

    def _distribute_zarr_to_node_ram_shards(self, zarr_path: str, *, n_samples: int) -> Dict[str, Any]:
        """
        Multi-node RAM staging: stage one Zarr span per node, then fan out local worker shards.
        """
        n_workers = len(self.workers)
        if n_workers <= 0:
            raise RuntimeError("No workers initialized")

        groups = self._group_worker_spans_by_node(int(n_samples))
        if not groups:
            raise RuntimeError("No node groups available for node-local RAM staging")

        leaders: List[Any] = []
        stage_futures = []
        for group in groups:
            leader = self.workers[int(group["leader_worker_idx"])]
            leaders.append(leader)
            stage_futures.append(
                leader.stage_zarr_node_span.remote(
                    zarr_path,
                    int(group["node_start"]),
                    int(group["node_end"]),
                    list(group["workers"]),
                )
            )

        try:
            stage_metadata = self.wait_for_worker_futures(
                stage_futures,
                phase="zarr_node_ram_stage",
                timeout_s=self._resolve_phase_timeout_s(
                    "zarr_node_ram_stage",
                    default_s=3600.0,
                ),
            )
        except Exception:
            self._pending_node_ram_stage_leaders = leaders
            self._cleanup_pending_node_ram_stage_leaders(suppress_exceptions=True)
            raise

        shape = (int(n_samples), int(self.n_features))
        async_config = None
        if self.processing_config is not None:
            async_config = getattr(self.processing_config, "async_loading_config", None)
        if async_config is None:
            async_config = AsyncLoadingConfig()

        data_size_gb = int(n_samples) * int(self.n_features) * 4 / (1024**3)
        use_async = async_config.should_use_async(data_size_gb)

        distribution_futures = []
        for group, leader in zip(groups, leaders):
            for worker_spec in group["workers"]:
                worker = self.workers[int(worker_spec["worker_idx"])]
                is_leader_worker = int(worker_spec["worker_idx"]) == int(group["leader_worker_idx"])
                if use_async:
                    if is_leader_worker:
                        future = worker.use_local_node_staged_ram_shard_async.remote(
                            int(worker_spec["worker_idx"]),
                            int(worker_spec["start"]),
                            int(worker_spec["end"]),
                            async_config=async_config,
                        )
                    else:
                        future = worker.use_node_staged_ram_shard_async.remote(
                            leader,
                            int(worker_spec["worker_idx"]),
                            int(worker_spec["start"]),
                            int(worker_spec["end"]),
                            async_config=async_config,
                        )
                else:
                    if is_leader_worker:
                        future = worker.use_local_node_staged_ram_shard.remote(
                            int(worker_spec["worker_idx"]),
                            int(worker_spec["start"]),
                            int(worker_spec["end"]),
                        )
                    else:
                        future = worker.use_node_staged_ram_shard.remote(
                            leader,
                            int(worker_spec["worker_idx"]),
                            int(worker_spec["start"]),
                            int(worker_spec["end"]),
                        )
                distribution_futures.append(future)

        stage_copy_workers: List[int] = []
        for meta in stage_metadata:
            if isinstance(meta, dict):
                copy_workers = meta.get("copy_workers_used")
                if isinstance(copy_workers, (int, float)):
                    stage_copy_workers.append(int(copy_workers))

        details: Dict[str, Any] = {
            "data_staging_strategy": "zarr_to_node_local_ram_shards",
            "data_staging_async": bool(use_async),
            "data_staging_workers": int(n_workers),
            "data_staging_nodes": int(len(groups)),
            "data_staging_node_metadata_count": int(len(stage_metadata)),
        }
        if stage_copy_workers:
            details["data_staging_node_copy_workers_max"] = int(max(stage_copy_workers))

        if use_async:
            self.distribution_futures = distribution_futures
            self.async_loading_active = True
            self.worker_metadata = []
            self.data_distributed = False
            self.total_samples = int(shape[0])
            self.n_features = int(shape[1]) if len(shape) > 1 else 1
            self._pending_node_ram_stage_leaders = leaders
            if async_config.enable_progress_reporting:
                self._start_loading_monitor(async_config.progress_interval)
        else:
            try:
                self.worker_metadata = self.wait_for_worker_futures(
                    distribution_futures,
                    phase="zarr_node_ram_distribution",
                    timeout_s=self._resolve_phase_timeout_s(
                        "zarr_node_ram_distribution",
                        default_s=3600.0,
                    ),
                )
                self.total_samples = int(shape[0])
                self.n_features = int(shape[1]) if len(shape) > 1 else 1
                self.distribution_futures = None
                self.async_loading_active = False
                self.data_distributed = True
                details["data_staging_worker_metadata_count"] = int(len(self.worker_metadata))
            finally:
                self._pending_node_ram_stage_leaders = leaders
                self._cleanup_pending_node_ram_stage_leaders()

        return details
            
    def _distribute_fast_array_once(self, source_info: Optional[Dict]) -> Dict[str, Any]:
        """
        One-time distribution of data as FastArrayStore files to workers.
        Each worker creates a local copy for multi-node support.
        
        Args:
            source_info: Dictionary with source information (type, path/array, format)
        """
        staging_details: Dict[str, Any] = {}
        if isinstance(source_info, dict):
            staging_details["data_staging_source_type"] = source_info.get("type")
            staging_details["data_staging_source_format"] = source_info.get("format")

        # Check if we should use RAM mode
        use_ram_mode = False
        data_ref = None
        
        # Handle array input - check if it fits in RAM first
        if source_info and source_info.get('type') == 'memory':
            data = source_info['array']
            use_ram_mode, reason = self._should_use_ram_mode(data.shape)
            
            if use_ram_mode:
                print("=" * 60, flush=True)
                print("======== RAM MODE ACTIVATED ========", flush=True)
                print("=" * 60, flush=True)
                print(f"Reason: {reason}", flush=True)
                print("Using direct memory transfer (no disk I/O)", flush=True)
                staging_details["data_staging_mode"] = "ram"
                staging_details["data_staging_strategy"] = "array_to_parallel_worker_ram_shards"

                # Distribute one RAM shard object per worker.
                self._distribute_ram_shards(data)
                staging_details["data_staging_async"] = bool(getattr(self, "async_loading_active", False))
                
                # Free the original array
                del data
                source_info['array'] = None
                return staging_details
            else:
                print("=" * 60, flush=True)
                print("======== DISK MODE - LOCAL MEMORY WRITES ========", flush=True)
                print("=" * 60, flush=True)
                print(f"Reason: {reason}", flush=True)
                print("Writing data to local disk as FastArrayStore format...", flush=True)
                staging_details["data_staging_mode"] = "disk"
                staging_details["data_staging_strategy"] = "array_to_fast_array_store"
            
            data = source_info['array']

            # Create FastArrayStore inside node-local temporary directory
            fast_path = self._allocate_temp_dir('fast_array')

            logger.info(f"Writing array data to FastArrayStore: {fast_path}")
            
            # Create FastArrayStore
            store = FastArrayStore(fast_path, mode='w')
            chunk_shape = (min(self.loader_chunk_size, data.shape[0]), data.shape[1] if len(data.shape) > 1 else 1)
            arr = store.create(shape=data.shape, dtype=np.float32, chunks=chunk_shape)
            
            # Write data in chunks
            for i in range(0, data.shape[0], self.loader_chunk_size):
                end = min(i + self.loader_chunk_size, data.shape[0])
                chunk = data[i:end]
                if chunk.dtype != np.float32:
                    chunk = chunk.astype(np.float32, copy=False)
                arr[i:end] = chunk
            
            store.close()
            
            # Free memory
            del data
            source_info['array'] = None
            
            # Update source_info
            source_info = {
                'type': 'file',
                'path': fast_path,
                'format': 'fast_array'
            }
            logger.info(f"Array successfully converted to FastArrayStore at: {fast_path}")
            staging_details["data_staging_intermediate_path"] = fast_path
        
        if not source_info or source_info.get('type') != 'file':
            raise ValueError("Source info with file path required")
        
        file_path = source_info['path']
        file_format = source_info.get('format', 'auto')
        n_workers = len(self.workers)
        
        # Handle existing FastArrayStore
        if file_format == 'fast_array' or os.path.exists(os.path.join(file_path, 'array_metadata.json')):
            store = FastArrayStore(file_path, mode='r')
            n_samples = store.n_samples
            n_features = store.n_features
            self.total_samples = n_samples
            self.n_features = n_features

            if n_workers == 1:
                # Single worker uses existing store
                store.close()
                future = self.workers[0].use_existing_fast_array.remote(file_path)
                self.worker_metadata = self.wait_for_worker_futures(
                    [future],
                    phase="fast_array_single_worker_distribution",
                    timeout_s=self._resolve_phase_timeout_s(
                        "fast_array_single_worker_distribution",
                        default_s=3600.0,
                    ),
                )
                if "data_staging_mode" not in staging_details:
                    staging_details["data_staging_mode"] = "disk"
                staging_details["data_staging_strategy"] = "fast_array_store_single_worker"
                staging_details["data_staging_workers"] = int(n_workers)
                staging_details["data_staging_worker_metadata_count"] = int(len(self.worker_metadata))
                return staging_details

            # Multi-worker: each creates local copy
            samples_per_worker = n_samples // n_workers

            run_id = uuid.uuid4().hex
            logger.info(
                "FastArrayStore distribution started: shape=%s workers=%d path=%s run_id=%s",
                (n_samples, n_features),
                n_workers,
                file_path,
                run_id,
            )
            print(
                f"FastArrayStore distribution started: shape={(n_samples, n_features)} "
                f"workers={n_workers} path={file_path} run_id={run_id}",
                flush=True,
            )

            distribution_futures = []
            for i, worker in enumerate(self.workers):
                start = i * samples_per_worker
                end = start + samples_per_worker if i < n_workers - 1 else n_samples

                # Each worker creates local FastArrayStore copy
                future = worker.create_local_fast_array.remote(
                    file_path, start, end, i, run_id
                )
                distribution_futures.append(future)

            self.worker_metadata = self.wait_for_worker_futures(
                distribution_futures,
                phase="fast_array_local_copy_distribution",
                timeout_s=self._resolve_phase_timeout_s(
                    "fast_array_local_copy_distribution",
                    default_s=3600.0,
                ),
            )
            staging_details["data_staging_workers"] = int(n_workers)
            staging_details["data_staging_worker_metadata_count"] = int(len(self.worker_metadata))

            elapsed_values = []
            throughput_values = []
            copy_workers_values = []
            for meta in self.worker_metadata:
                if not isinstance(meta, dict):
                    continue
                elapsed = meta.get("elapsed_s")
                if elapsed is not None:
                    elapsed_values.append(float(elapsed))
                throughput = meta.get("throughput_gb_s")
                if throughput is not None:
                    throughput_values.append(float(throughput))
                copy_workers = meta.get("copy_workers_used")
                if copy_workers is not None:
                    copy_workers_values.append(int(copy_workers))

            if elapsed_values:
                staging_details["data_staging_worker_elapsed_s_min"] = float(min(elapsed_values))
                staging_details["data_staging_worker_elapsed_s_max"] = float(max(elapsed_values))
                staging_details["data_staging_worker_elapsed_s_mean"] = float(
                    sum(elapsed_values) / max(1, len(elapsed_values))
                )

            if throughput_values:
                staging_details["data_staging_worker_throughput_gb_s_min"] = float(min(throughput_values))
                staging_details["data_staging_worker_throughput_gb_s_max"] = float(max(throughput_values))
                staging_details["data_staging_worker_throughput_gb_s_mean"] = float(
                    sum(throughput_values) / max(1, len(throughput_values))
                )

            if copy_workers_values:
                staging_details["data_staging_worker_copy_workers_min"] = int(min(copy_workers_values))
                staging_details["data_staging_worker_copy_workers_max"] = int(max(copy_workers_values))
                staging_details["data_staging_worker_copy_workers_mean"] = float(
                    sum(copy_workers_values) / max(1, len(copy_workers_values))
                )
            if "data_staging_mode" not in staging_details:
                staging_details["data_staging_mode"] = "disk"
            staging_details["data_staging_strategy"] = "fast_array_store_to_worker_local_shards"
            staging_details["data_staging_workers"] = int(n_workers)
            staging_details["data_staging_worker_metadata_count"] = int(len(self.worker_metadata))
            logger.info(
                "FastArrayStore distribution completed: shape=%s workers=%d path=%s run_id=%s",
                (n_samples, n_features),
                n_workers,
                file_path,
                run_id,
            )
            print(
                f"FastArrayStore distribution completed: shape={(n_samples, n_features)} "
                f"workers={n_workers} path={file_path} run_id={run_id}",
                flush=True,
            )
            store.close()
            return staging_details
        
        # Handle Zarr conversion
        elif file_format == 'zarr' or file_path.endswith('.zarr'):
            z = open_array_read(file_path).array
            shape = z.shape
            n_samples = shape[0] if len(shape) > 0 else 0
            n_features = shape[1] if len(shape) > 1 else 1
            self.total_samples = n_samples
            self.n_features = n_features
            
            # Check if Zarr data fits in RAM
            use_ram_mode, reason = self._should_use_ram_mode(shape)
            
            if use_ram_mode:
                workers_on_node = self._estimate_workers_on_node(n_workers)
                multi_node = workers_on_node < n_workers
                print("=" * 60, flush=True)
                if multi_node:
                    print("======== RAM MODE ACTIVATED (Zarr, node-local) ========", flush=True)
                else:
                    print("======== RAM MODE ACTIVATED (Zarr) ========", flush=True)
                print("=" * 60, flush=True)
                print(f"Reason: {reason}", flush=True)
                staging_details["data_staging_mode"] = "ram"
                if multi_node:
                    print("Loading Zarr spans once per node, then distributing local worker RAM shards", flush=True)
                    staging_details["data_staging_strategy"] = "zarr_to_node_local_ram_shards"
                    staging_details.update(self._distribute_zarr_to_node_ram_shards(file_path, n_samples=n_samples))
                    return staging_details
                else:
                    print("Loading Zarr data into RAM for direct memory transfer", flush=True)
                    staging_details["data_staging_strategy"] = "zarr_to_parallel_worker_ram_shards"

                    data = z[:]
                    logger.info(f"Loaded Zarr data with dtype: {data.dtype}")
                    self._distribute_ram_shards(data)
                    staging_details["data_staging_async"] = bool(getattr(self, "async_loading_active", False))

                    del data
                    staging_details["data_staging_workers"] = int(n_workers)
                    staging_details["data_staging_worker_metadata_count"] = int(
                        len(self.worker_metadata) if self.worker_metadata else 0
                    )
                    return staging_details
            else:
                print("=" * 60, flush=True)
                print("======== DISK MODE - LOCAL MEMORY WRITES (Zarr) ========", flush=True)
                print("=" * 60, flush=True)
                print(f"Reason: {reason}", flush=True)
                print("Distributing Zarr directly to worker-local FastArrayStore shards", flush=True)
                staging_details["data_staging_mode"] = "disk"
                staging_details["data_staging_strategy"] = "zarr_to_worker_local_shards"

            run_id = uuid.uuid4().hex
            logger.info(
                "Zarr direct-to-local distribution started: shape=%s workers=%d path=%s run_id=%s",
                shape,
                n_workers,
                file_path,
                run_id,
            )
            print(
                f"Zarr direct-to-local distribution started: shape={shape} "
                f"workers={n_workers} path={file_path} run_id={run_id}",
                flush=True,
            )

            samples_per_worker = n_samples // n_workers if n_workers else n_samples
            distribution_futures = []
            for i, worker in enumerate(self.workers):
                start = i * samples_per_worker
                end = start + samples_per_worker if i < n_workers - 1 else n_samples
                future = worker.create_local_fast_array_from_zarr.remote(
                    file_path, start, end, i, run_id
                )
                distribution_futures.append(future)

            self.worker_metadata = self.wait_for_worker_futures(
                distribution_futures,
                phase="zarr_local_copy_distribution",
                timeout_s=self._resolve_phase_timeout_s(
                    "zarr_local_copy_distribution",
                    default_s=3600.0,
                ),
            )

            logger.info(
                "Zarr direct-to-local distribution completed: shape=%s workers=%d path=%s run_id=%s",
                shape,
                n_workers,
                file_path,
                run_id,
            )
            print(
                f"Zarr direct-to-local distribution completed: shape={shape} "
                f"workers={n_workers} path={file_path} run_id={run_id}",
                flush=True,
            )
            return staging_details
                    
        # Handle other formats by loading and converting
        else:
            # Load data based on format
            if file_path.endswith('.npy'):
                data = np.load(file_path)
            elif file_path.endswith('.parquet'):
                import pandas as pd
                df = pd.read_parquet(file_path)
                data = df.to_numpy()
            else:
                raise ValueError(f"Unsupported format: {file_path}")
            
            # Convert to FastArrayStore
            source_info = {'type': 'memory', 'array': data, 'format': 'array'}
            return self._distribute_fast_array_once(source_info)
    
    def get_worker_info(self) -> List[Dict]:
        """
        Get information about all workers
        
        Returns:
            List of worker information dictionaries
        """
        if not self.workers:
            return []
        
        info_futures = [worker.get_info.remote() for worker in self.workers]
        return self.wait_for_worker_futures(
            info_futures,
            phase="worker_info",
            timeout_s=self._resolve_phase_timeout_s("worker_info", default_s=30.0),
        )
    
    def get_final_weights(self):
        """
        Get final weights from workers (transfers from GPU to CPU).
        Only call this at the end of training or when weights are needed on CPU.
        
        Returns:
            Tuple of (weights, delta_weights) as numpy arrays on CPU,
            or just weights if momentum is not enabled
        """
        if not self.workers:
            raise RuntimeError("No workers initialized")
        
        # Get weights from first worker (all workers have identical weights after NCCL sync)
        result = ray.get(self.workers[0].get_final_weights.remote())
        
        # Return in the expected format
        return result
    
    def get_gpu_weights(self):
        """
        Get weights that remain on GPU (no CPU transfer).
        Use this during training iterations to avoid unnecessary transfers.
        
        Returns:
            Weights as CuPy array on GPU
        """
        if not self.workers:
            raise RuntimeError("No workers initialized")
        
        # Get GPU weights from first worker without CPU transfer
        # This method needs to be implemented in the worker
        return ray.get(self.workers[0].get_gpu_weights_only.remote())
    
    def finalize(self):
        """
        Finalize processing and return final weights.
        Called at the end of training.

        Returns:
            Final weights (kept as CuPy arrays for GPU operations)
        """
        if self.workers:
            result = self.get_final_weights()
            return result
        return None
    
    def _should_use_ram_mode(self, shape: tuple) -> tuple[bool, str]:
        """
        Determine if we should use RAM mode based on data shape and per-worker memory requirements.

        Args:
            shape: Shape of the data array
            
        Returns:
            Tuple of (use_ram_mode, reason_string)
        """
        force_disk_mode = False
        if isinstance(self.ray_config, dict):
            force_disk_mode = bool(self.ray_config.get("force_disk_mode", False))
        elif self.ray_config is not None:
            force_disk_mode = bool(getattr(self.ray_config, "force_disk_mode", False))

        if force_disk_mode:
            return (
                False,
                "force_disk_mode=True in RayConfig; forcing disk staging",
            )

        # Get data dimensions
        n_samples = shape[0]
        n_features = shape[1] if len(shape) > 1 else 1
        
        # Calculate data size per worker
        n_workers = len(self.workers) if self.workers else self.num_gpus if self.num_gpus else 1
        n_workers = max(1, n_workers)
        if n_samples:
            samples_per_worker = max(1, math.ceil(n_samples / n_workers))
        else:
            samples_per_worker = 0
        data_size_per_worker = samples_per_worker * n_features * 4  # float32 = 4 bytes

        # Get available RAM per worker
        available_ram = psutil.virtual_memory().available

        # Estimate number of workers sharing a single node's RAM (can be < total in multi-node).
        n_workers_on_node = self._estimate_workers_on_node(n_workers)
        multi_node = n_workers_on_node < n_workers

        ram_per_worker = available_ram / n_workers_on_node

        # Apply job-level memory limit if present (PBS/SLURM/cgroup)
        job_limit = self._get_job_memory_limit_bytes()
        if job_limit and job_limit < (1 << 62):  # ignore unrealistically large "no limit" values
            job_ram_per_worker = job_limit / max(1, n_workers_on_node)
            ram_per_worker = min(ram_per_worker, job_ram_per_worker)

        if multi_node:
            n_nodes = max(1, math.ceil(n_workers / max(1, n_workers_on_node)))
            samples_per_node = min(
                int(n_samples),
                int(samples_per_worker) * max(1, int(n_workers_on_node)),
            )
            data_size_per_node = samples_per_node * n_features * 4
            pinned_buffer_per_node = (
                min(self.loader_chunk_size, samples_per_worker) * n_features * 4 * max(1, n_workers_on_node)
            )
            shard_copy_bytes = data_size_per_node
            other_overhead = data_size_per_node * 0.1
            total_needed_per_node = (
                data_size_per_node
                + shard_copy_bytes
                + pinned_buffer_per_node
                + other_overhead
            )

            available_ram_per_node = available_ram
            if job_limit and job_limit < (1 << 62):
                available_ram_per_node = min(available_ram_per_node, job_limit / max(1, n_nodes))

            usable_ram_per_node = available_ram_per_node * 0.8
            headroom_factor = 1.5
            required_ram_per_node = total_needed_per_node * headroom_factor
            use_ram_mode = usable_ram_per_node > required_ram_per_node

            data_gb = data_size_per_node / (1024**3)
            needed_gb = required_ram_per_node / (1024**3)
            available_gb = usable_ram_per_node / (1024**3)
            limit_gb = (job_limit / (1024**3)) if job_limit else None

            if use_ram_mode:
                reason = (
                    f"Node-local RAM staging ({data_gb:.2f} GB per node span) fits in available RAM "
                    f"({available_gb:.2f} GB per node with safety margin)"
                )
            else:
                reason = (
                    f"Node-local RAM staging ({data_gb:.2f} GB per node span, {needed_gb:.2f} GB needed "
                    f"with {headroom_factor:.1f}x headroom) exceeds safe RAM limit "
                    f"({available_gb:.2f} GB per node)"
                )
            if limit_gb:
                reason += f"; job limit: {limit_gb:.2f} GB"

            return use_ram_mode, reason

        # Account for additional memory needs
        pinned_buffer_size = min(self.loader_chunk_size, samples_per_worker) * n_features * 4
        other_overhead = data_size_per_worker * 0.1  # 10% for other allocations

        total_needed_per_worker = data_size_per_worker + pinned_buffer_size + other_overhead
        usable_ram_per_worker = ram_per_worker * 0.8  # 20% safety margin

        # Make decision
        headroom_factor = 2
        required_ram_per_worker = total_needed_per_worker * headroom_factor
        use_ram_mode = usable_ram_per_worker > required_ram_per_worker

        # Create informative message
        data_gb = data_size_per_worker / (1024**3)
        needed_gb = required_ram_per_worker / (1024**3)
        available_gb = usable_ram_per_worker / (1024**3)
        limit_gb = (job_limit / (1024**3)) if job_limit else None

        if use_ram_mode:
            reason = (f"Dataset ({data_gb:.2f} GB per worker) fits in available RAM "
                     f"({available_gb:.2f} GB per worker with safety margin)")
        else:
            reason = (f"Dataset ({data_gb:.2f} GB per worker, {needed_gb:.2f} GB needed "
                     f"with {headroom_factor:.1f}x headroom) exceeds safe RAM limit "
                     f"({available_gb:.2f} GB per worker)")
        if limit_gb:
            reason += f"; job limit: {limit_gb:.2f} GB"

        return use_ram_mode, reason

    def _get_job_memory_limit_bytes(self) -> Optional[int]:
        """Best-effort detection of job-level memory limits (scheduler or cgroup)."""
        from .resource_limits import get_job_memory_limit_bytes

        return get_job_memory_limit_bytes()
	    
    @staticmethod
    def _estimate_workers_on_node(total_workers: int) -> int:
        """
        Best-effort estimate of how many GPU workers will share a single node.

        For multi-node Ray, `ray.cluster_resources()` reports cluster-wide totals, so we derive
        a per-node estimate from `ray.nodes()` instead (e.g., max GPUs on any alive node).
        """
        from .resource_limits import estimate_workers_on_node

        return estimate_workers_on_node(total_workers)

    @staticmethod
    def _compute_worker_spans(n_samples: int, n_workers: int) -> List[tuple[int, int]]:
        """Split rows into contiguous per-worker spans."""
        if n_workers <= 0:
            return []
        if n_samples <= 0:
            return [(0, 0) for _ in range(n_workers)]

        samples_per_worker = n_samples // n_workers
        spans: List[tuple[int, int]] = []
        for worker_idx in range(n_workers):
            start = worker_idx * samples_per_worker
            end = start + samples_per_worker if worker_idx < n_workers - 1 else n_samples
            spans.append((int(start), int(end)))
        return spans

    def _create_parallel_ram_shard_refs(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Create one Ray object-store shard per worker using parallel manager threads."""
        spans = self._compute_worker_spans(int(data.shape[0]), len(self.workers))
        if not spans:
            return []

        max_workers = min(len(spans), max(1, (os.cpu_count() or 1) - 1))
        logger.info(
            "Creating %d worker RAM shard object(s) with %d manager thread(s)",
            len(spans),
            max_workers,
        )

        shard_specs: List[Optional[Dict[str, Any]]] = [None] * len(spans)

        def _put_shard(item: tuple[int, tuple[int, int]]) -> tuple[int, Dict[str, Any]]:
            worker_idx, (start, end) = item
            shard = np.asarray(data[start:end], dtype=np.float32)
            if not shard.flags["C_CONTIGUOUS"]:
                shard = np.ascontiguousarray(shard)
            return worker_idx, {
                "worker_idx": int(worker_idx),
                "start": int(start),
                "end": int(end),
                "n_samples": int(end - start),
                "ref": ray.put(shard),
            }

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ram-shard-put",
        ) as executor:
            futures = [
                executor.submit(_put_shard, (worker_idx, span))
                for worker_idx, span in enumerate(spans)
            ]
            for future in concurrent.futures.as_completed(futures):
                worker_idx, spec = future.result()
                shard_specs[worker_idx] = spec

        return [spec for spec in shard_specs if spec is not None]

    def _distribute_ram_shards(self, data: np.ndarray) -> None:
        """Distribute per-worker RAM shard objects with sync or async worker setup."""
        shape = tuple(int(dim) for dim in data.shape)
        n_samples = shape[0]
        n_features = shape[1] if len(shape) > 1 else 1

        async_config = None
        if self.processing_config is not None:
            async_config = getattr(self.processing_config, "async_loading_config", None)
        if async_config is None:
            async_config = AsyncLoadingConfig()

        data_size_gb = n_samples * n_features * 4 / (1024**3)
        use_async = async_config.should_use_async(data_size_gb)
        shard_specs = self._create_parallel_ram_shard_refs(data)

        if use_async:
            logger.info("Using ASYNC RAM shard setup for %.2f GB dataset", data_size_gb)
            self._distribute_ram_shards_async(shard_specs, shape, async_config)
        else:
            logger.info(
                "Using SYNC RAM shard setup for %.2f GB dataset (<%s GB threshold)",
                data_size_gb,
                async_config.auto_disable_threshold_gb,
            )
            self._distribute_ram_shards_sync(shard_specs, shape)

    def _distribute_ram_shards_sync(self, shard_specs: List[Dict[str, Any]], shape: tuple) -> None:
        """Synchronously configure workers from per-worker RAM shard objects."""
        n_samples = shape[0]
        n_features = shape[1] if len(shape) > 1 else 1

        distribution_futures = []
        for spec in shard_specs:
            worker_idx = int(spec["worker_idx"])
            distribution_futures.append(
                self.workers[worker_idx].use_ram_data.remote(
                    spec["ref"],
                    int(spec["start"]),
                    int(spec["end"]),
                    worker_shard=True,
                )
            )

        self.worker_metadata = self.wait_for_worker_futures(
            distribution_futures,
            phase="ram_shard_sync_distribution",
            timeout_s=self._resolve_phase_timeout_s(
                "ram_shard_sync_distribution",
                default_s=1800.0,
            ),
        )

        self.total_samples = n_samples
        self.n_features = n_features
        self.distribution_futures = None
        self.async_loading_active = False
        self.data_distributed = True

        logger.info(
            "RAM shard distribution complete: %d samples across %d workers",
            n_samples,
            len(self.workers),
        )

    def _distribute_ram_shards_async(
        self,
        shard_specs: List[Dict[str, Any]],
        shape: tuple,
        config: AsyncLoadingConfig,
    ) -> None:
        """Asynchronously configure workers from per-worker RAM shard objects."""
        n_samples = shape[0]
        n_features = shape[1] if len(shape) > 1 else 1

        logger.info(
            "Starting async RAM shard distribution: %d samples across %d workers",
            n_samples,
            len(self.workers),
        )

        self.distribution_futures = [
            self.workers[int(spec["worker_idx"])].use_ram_data_async.remote(
                spec["ref"],
                int(spec["start"]),
                int(spec["end"]),
                async_config=config,
                worker_shard=True,
            )
            for spec in shard_specs
        ]
        self.async_loading_active = True
        self.total_samples = n_samples
        self.n_features = n_features
        self.worker_metadata = []
        self.data_distributed = False

        if config.enable_progress_reporting:
            self._start_loading_monitor(config.progress_interval)

        logger.info(
            "Async RAM shard distribution initiated; waiting for worker setup completion before processing starts"
        )

    def _distribute_ram_data(self, data_ref, shape: tuple):
        """
        Distribute data to workers using RAM mode with optional async loading.
        
        Args:
            data_ref: Ray object reference to the data
            shape: Shape of the data array
        """
        n_workers = len(self.workers)
        n_samples = shape[0]
        n_features = shape[1] if len(shape) > 1 else 1
        
        # Check if async loading should be used
        async_config = None
        if self.processing_config is not None:
            async_config = getattr(self.processing_config, "async_loading_config", None)
        if async_config is None:
            async_config = AsyncLoadingConfig()
        
        data_size_gb = n_samples * n_features * 4 / (1024**3)  # float32
        use_async = async_config.should_use_async(data_size_gb)
        
        if use_async:
            logger.info(f"Using ASYNC loading for {data_size_gb:.2f} GB dataset")
            return self._distribute_ram_data_async(data_ref, shape, async_config)
        else:
            logger.info(f"Using SYNC loading for {data_size_gb:.2f} GB dataset (<{async_config.auto_disable_threshold_gb} GB threshold)")
            return self._distribute_ram_data_sync(data_ref, shape)
    
    def _distribute_ram_data_sync(self, data_ref, shape: tuple):
        """Original synchronous distribution (fallback)"""
        n_workers = len(self.workers)
        n_samples = shape[0]
        n_features = shape[1] if len(shape) > 1 else 1
        
        if n_workers == 1:
            # Single worker gets full data reference
            future = self.workers[0].use_ram_data.remote(data_ref, 0, n_samples)
            self.worker_metadata = self.wait_for_worker_futures(
                [future],
                phase="ram_sync_distribution",
                timeout_s=self._resolve_phase_timeout_s(
                    "ram_sync_distribution",
                    default_s=1800.0,
                ),
            )
        else:
            # Multi-worker: each gets their slice boundaries
            samples_per_worker = n_samples // n_workers
            
            distribution_futures = []
            for i, worker in enumerate(self.workers):
                start = i * samples_per_worker
                end = start + samples_per_worker if i < n_workers - 1 else n_samples
                
                # Worker will extract their slice from the shared data
                future = worker.use_ram_data.remote(data_ref, start, end)
                distribution_futures.append(future)
            
            self.worker_metadata = self.wait_for_worker_futures(
                distribution_futures,
                phase="ram_sync_distribution",
                timeout_s=self._resolve_phase_timeout_s(
                    "ram_sync_distribution",
                    default_s=1800.0,
                ),
            )
        
        # Store metadata
        self.total_samples = n_samples
        self.n_features = n_features
        self.distribution_futures = None
        self.async_loading_active = False
        self.data_distributed = True
        
        logger.info(f"RAM mode distribution complete: {n_samples} samples across {n_workers} workers")
    
    def _distribute_ram_data_async(self, data_ref, shape: tuple, config: AsyncLoadingConfig):
        """
        New asynchronous distribution with progressive loading.
        Processing can begin before all data is loaded.
        """
        n_workers = len(self.workers)
        n_samples = shape[0]
        n_features = shape[1] if len(shape) > 1 else 1
        
        logger.info(f"Starting async RAM distribution: {n_samples} samples across {n_workers} workers")
        
        if n_workers == 1:
            # Single worker - simplified async configuration
            future = self.workers[0].use_ram_data_async.remote(
                data_ref, 0, n_samples,
                async_config=config
            )
            self.distribution_futures = [future]
            self.async_loading_active = True
        else:
            # Multi-worker async distribution
            samples_per_worker = n_samples // n_workers
            
            distribution_futures = []
            
            for i, worker in enumerate(self.workers):
                start = i * samples_per_worker
                end = start + samples_per_worker if i < n_workers - 1 else n_samples
                
                # Configure async loading on each worker
                future = worker.use_ram_data_async.remote(
                    data_ref, start, end,
                    async_config=config
                )
                distribution_futures.append(future)
            self.distribution_futures = distribution_futures
            self.async_loading_active = True
        
        # Store metadata
        self.total_samples = n_samples
        self.n_features = n_features
        self.worker_metadata = []
        self.data_distributed = False  # Not ready until async setup futures complete
        
        # Start monitoring thread for progress reporting
        if config.enable_progress_reporting:
            self._start_loading_monitor(config.progress_interval)
        
        logger.info(
            "Async distribution initiated; waiting for worker setup completion before processing starts"
        )
    
    def _start_loading_monitor(self, interval: float = 5.0):
        """Monitor async loading progress in background"""
        def monitor():
            while self.async_loading_active:
                time.sleep(interval)
                if self.workers:
                    progress_futures = [w.get_loading_progress.remote() for w in self.workers]
                    try:
                        progress_list = ray.get(progress_futures, timeout=1.0)
                        total_ready = sum(p['ready_count'] for p in progress_list)
                        total_chunks = sum(p['total_chunks'] for p in progress_list)
                        pct = (total_ready / total_chunks * 100) if total_chunks > 0 else 0
                        logger.info(f"Loading progress: {total_ready}/{total_chunks} chunks ready ({pct:.1f}%)")
                    except Exception:
                        logger.exception("Loading progress monitor failed")
                        raise
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def wait_for_all_data(self, timeout: float = 300.0) -> bool:
        """
        Wait for all async loading to complete.
        Optional - processing can continue without calling this.
        """
        deadline = time.monotonic() + max(0.1, float(timeout))

        try:
            if self.distribution_futures:
                remaining = max(0.1, deadline - time.monotonic())
                self.ensure_distribution_ready(timeout=remaining)
        except Exception as exc:
            logger.error("Failed waiting for async distribution setup: %s", exc)
            return False

        if not self.async_loading_active:
            return True

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error("Timeout waiting for async loading to complete")
                return False

            progress_futures = [w.get_loading_progress.remote() for w in self.workers]
            try:
                progress_list = self.wait_for_worker_futures(
                    progress_futures,
                    phase="async_loading_progress",
                    timeout_s=min(remaining, 10.0),
                    poll_interval_s=1.0,
                )
            except Exception as exc:
                logger.error("Failed querying async loading progress: %s", exc)
                return False

            total_ready = 0
            total_chunks = 0
            configured_not_started = 0
            for progress in progress_list:
                if not isinstance(progress, dict):
                    continue
                status = str(progress.get("status", "") or "")
                if status == "async_configured_not_started":
                    configured_not_started += 1
                total_ready += int(progress.get("ready_count", 0) or 0)
                total_chunks += int(progress.get("total_chunks", 0) or 0)

            if configured_not_started == len(progress_list) and len(progress_list) > 0:
                logger.info(
                    "Async data loaders are configured but not started; returning before iteration start"
                )
                return True

            if total_chunks > 0 and total_ready >= total_chunks:
                logger.info("All async data loading complete")
                self.async_loading_active = False
                return True

            time.sleep(min(1.0, max(0.1, remaining)))
