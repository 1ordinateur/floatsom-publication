"""
Tests for Ray storage path separation between shared staging and worker-local copies.
"""

from pathlib import Path

import pytest

pytest.importorskip("cupy")
pytest.importorskip("ray")

from floatsom.floatsom_params import RayConfig
from floatsom.processing.ray_ops import ray_pipeline_base as manager_module
from floatsom.processing.ray_ops.workers import ray_pipeline_base_worker as worker_module


def _build_ray_config(shared_path: Path, local_path: Path) -> RayConfig:
    return RayConfig(
        chunk_size=1,
        storage_path=str(shared_path),
        local_storage_path=str(local_path),
        num_gpus=1,
    )


def _make_worker(ray_config: RayConfig, worker_id: int):
    worker = worker_module.RayPipelineBaseWorker.__new__(worker_module.RayPipelineBaseWorker)
    worker.ray_config = ray_config
    worker.worker_id = worker_id
    return worker


def test_ray_config_requires_storage_paths():
    with pytest.raises(ValueError, match="storage_path"):
        RayConfig(
            chunk_size=1,
            storage_path="",
            local_storage_path="local",
            num_gpus=1,
        )

    with pytest.raises(ValueError, match="local_storage_path"):
        RayConfig(
            chunk_size=1,
            storage_path="shared",
            local_storage_path="",
            num_gpus=1,
        )


def test_manager_staging_uses_shared_storage(tmp_path, monkeypatch):
    shared_path = tmp_path / "shared"
    local_path = tmp_path / "local"
    ray_config = _build_ray_config(shared_path, local_path)

    monkeypatch.setattr(manager_module.socket, "gethostname", lambda: "node-a")
    manager = manager_module.RayWorkerManager(num_gpus=1, ray_config=ray_config)

    expected = shared_path / "node-a" / "manager"
    assert Path(manager.manager_temp_root) == expected


def test_worker_local_storage_uses_local_root_and_hostname(tmp_path, monkeypatch):
    shared_path = tmp_path / "shared"
    local_path = tmp_path / "local"
    ray_config = _build_ray_config(shared_path, local_path)
    worker = _make_worker(ray_config, worker_id=3)

    monkeypatch.setattr(worker_module.socket, "gethostname", lambda: "node-a")
    worker_path = Path(worker._get_local_storage_path())

    assert worker_path.name == "ray_worker_3"
    assert worker_path.parent.name == "node-a"
    assert worker_path.parents[1] == local_path


def test_worker_local_storage_is_node_scoped(tmp_path, monkeypatch):
    shared_path = tmp_path / "shared"
    local_path = tmp_path / "local"
    ray_config = _build_ray_config(shared_path, local_path)
    worker = _make_worker(ray_config, worker_id=0)

    monkeypatch.setattr(worker_module.socket, "gethostname", lambda: "node-a")
    path_a = Path(worker._get_local_storage_path())

    monkeypatch.setattr(worker_module.socket, "gethostname", lambda: "node-b")
    path_b = Path(worker._get_local_storage_path())

    assert path_a.parent.name == "node-a"
    assert path_b.parent.name == "node-b"
    assert path_a != path_b
