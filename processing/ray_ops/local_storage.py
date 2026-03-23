"""Worker-local scratch directory management for Ray pipelines."""

from __future__ import annotations

import os
import shutil


class LocalStorageWipeError(RuntimeError):
    """Raised when a worker-local storage path cannot be safely wiped."""


def _real_dir(path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def worker_local_storage_dir(*, base: str, hostname: str, worker_id: int) -> str:
    """Return the FloatSOM-managed worker-local directory under `base`."""
    return os.path.join(base, hostname, f"ray_worker_{int(worker_id)}")


def wipe_worker_local_storage(*, base: str, hostname: str, worker_id: int) -> str:
    """Wipe (delete + recreate) the worker-local directory used by FloatSOM."""
    if not base:
        raise LocalStorageWipeError("local_storage_path is empty")

    base_real = _real_dir(base)
    if base_real in (os.path.sep, ""):
        raise LocalStorageWipeError(
            f"Refusing to wipe worker-local storage under unsafe base path: {base!r}"
        )

    worker_dir = worker_local_storage_dir(
        base=base_real,
        hostname=str(hostname),
        worker_id=int(worker_id),
    )
    worker_real = _real_dir(worker_dir)

    try:
        common = os.path.commonpath([base_real, worker_real])
    except Exception as exc:  # pragma: no cover
        raise LocalStorageWipeError(
            f"Could not validate worker-local storage path {worker_real!r}: {exc}"
        ) from exc

    if common != base_real:
        raise LocalStorageWipeError(
            f"Refusing to wipe worker-local storage outside base path: {worker_real}"
        )

    try:
        if os.path.exists(worker_real) and not os.path.isdir(worker_real):
            os.remove(worker_real)
        if os.path.isdir(worker_real):
            shutil.rmtree(worker_real)
        os.makedirs(worker_real, exist_ok=True)
    except Exception as exc:  # pragma: no cover
        raise LocalStorageWipeError(
            f"Failed to wipe worker-local storage at {worker_real}: {exc}"
        ) from exc

    return worker_real
