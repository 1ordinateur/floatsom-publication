from __future__ import annotations

from pathlib import Path

import pytest

from floatsom.processing.ray_ops.local_storage import (
    LocalStorageWipeError,
    wipe_worker_local_storage,
)


def test_wipe_worker_local_storage_clears_only_worker_dir(tmp_path: Path) -> None:
    base = tmp_path / "jobfs"
    hostname = "node-a"
    worker_id = 2

    manager_dir = base / hostname / "manager"
    manager_dir.mkdir(parents=True)
    sentinel = manager_dir / "keep.txt"
    sentinel.write_text("keep")

    worker_dir = base / hostname / f"ray_worker_{worker_id}"
    worker_dir.mkdir(parents=True)
    (worker_dir / "old.bin").write_bytes(b"stale")

    wiped = Path(
        wipe_worker_local_storage(
            base=str(base),
            hostname=hostname,
            worker_id=worker_id,
        )
    )

    assert wiped == worker_dir.resolve()
    assert worker_dir.exists()
    assert list(worker_dir.iterdir()) == []
    assert sentinel.read_text() == "keep"


def test_wipe_worker_local_storage_rejects_empty_base() -> None:
    with pytest.raises(LocalStorageWipeError, match="empty"):
        wipe_worker_local_storage(base="", hostname="node-a", worker_id=0)


def test_wipe_worker_local_storage_rejects_root_base() -> None:
    with pytest.raises(LocalStorageWipeError, match="unsafe base"):
        wipe_worker_local_storage(base="/", hostname="node-a", worker_id=0)


def test_wipe_worker_local_storage_rejects_escape_hostname(tmp_path: Path) -> None:
    base = tmp_path / "jobfs"
    base.mkdir()

    with pytest.raises(LocalStorageWipeError, match="outside base"):
        wipe_worker_local_storage(base=str(base), hostname="..", worker_id=0)

