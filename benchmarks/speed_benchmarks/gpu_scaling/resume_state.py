#!/usr/bin/env python3
"""
Resume state tracking for GPU scaling benchmarks.

This module provides a lightweight manifest that records the status
of each benchmark configuration so interrupted runs can be resumed
without repeating completed work.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class ResumeManager:
    """Persist and manage benchmark resume state."""

    VALID_STATUSES = {"pending", "running", "done", "failed"}

    def __init__(self, state_path: str, enabled: bool, reset: bool = False):
        self.state_path = state_path
        self.enabled = bool(enabled)
        self._state: Dict[str, Any] = {"version": 1, "tasks": {}}
        self._dirty = False

        if not self.enabled:
            return

        if not state_path:
            raise ValueError("ResumeManager requires a non-empty state_path when enabled")

        if reset and os.path.exists(self.state_path):
            os.remove(self.state_path)

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_task(self, task_key: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ensure a task exists in the manifest. Returns the task record.
        """
        record = self._state["tasks"].get(task_key)
        if record is None:
            record = {
                "status": "pending",
                "metadata": metadata or {},
                "updated_at": self._timestamp(),
                "train_time": None,
                "error": None,
                "attempts": 0,
            }
            self._state["tasks"][task_key] = record
            self._dirty = True
        else:
            if metadata:
                record.setdefault("metadata", {}).update(metadata)
        return record

    def get_status(self, task_key: str) -> Optional[str]:
        """Return the current status for a task."""
        record = self._state["tasks"].get(task_key)
        if record is None:
            return None
        return record["status"]

    def get_record(self, task_key: str) -> Optional[Dict[str, Any]]:
        """Return the full record for a task."""
        return self._state["tasks"].get(task_key)

    def mark_running(self, task_key: str) -> None:
        """Mark a task as currently executing."""
        if not self.enabled:
            return
        record = self.register_task(task_key)
        record["status"] = "running"
        record["attempts"] = record.get("attempts", 0) + 1
        record["updated_at"] = self._timestamp()
        record["error"] = None
        self._dirty = True
        self._persist_if_needed()

    def mark_done(self, task_key: str, train_time: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark a task as completed successfully."""
        if not self.enabled:
            return
        record = self.register_task(task_key)
        record["status"] = "done"
        record["updated_at"] = self._timestamp()
        if train_time is not None:
            record["train_time"] = float(train_time)
        if metadata:
            record.setdefault("metadata", {}).update(metadata)
        record["error"] = None
        self._dirty = True
        self._persist_if_needed()

    def mark_failed(self, task_key: str, error: Optional[str] = None) -> None:
        """Mark a task as failed so it can be retried."""
        if not self.enabled:
            return
        record = self.register_task(task_key)
        record["status"] = "failed"
        record["updated_at"] = self._timestamp()
        if error:
            record["error"] = error
        self._dirty = True
        self._persist_if_needed()

    def update_train_time_if_missing(self, task_key: str, train_time: float) -> None:
        """Populate train time for finished tasks when bootstrapping from logs."""
        if not self.enabled:
            return
        record = self.get_record(task_key)
        if not record:
            return
        if record.get("train_time") is None:
            record["train_time"] = float(train_time)
            record["updated_at"] = self._timestamp()
            self._dirty = True
            self._persist_if_needed()

    def get_train_time(self, task_key: str) -> Optional[float]:
        """Return the recorded train time, if available."""
        record = self.get_record(task_key)
        if record:
            return record.get("train_time")
        return None

    def all_done(self) -> bool:
        """Return True if every registered task is marked as done."""
        if not self.enabled:
            return True
        tasks = self._state.get("tasks", {})
        if not tasks:
            return True
        return all(task.get("status") == "done" for task in tasks.values())

    def flush(self) -> None:
        """Force persistence of the current manifest."""
        if not self.enabled:
            return
        self._persist(force=True)

    def build_task_key(
        self,
        *,
        mode: str,
        topology: str,
        processing_method: str,
        gpu_count: Any,
        repeat_index: int,
        axis_name: str,
        axis_value: Any,
    ) -> str:
        """
        Construct a stable task key from benchmark configuration parameters.
        """
        return "|".join(
            [
                f"mode={mode}",
                f"topology={topology}",
                f"method={processing_method}",
                f"{axis_name}={axis_value}",
                f"gpus={gpu_count}",
                f"repeat={repeat_index}",
            ]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.enabled:
            return

        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict) and "tasks" in data:
                        self._state = data
                        return
            except Exception:
                # Fall back to a clean manifest if the file is unreadable
                pass

        # Ensure directory exists before first write
        state_dir = os.path.dirname(self.state_path)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)

        self._state = {"version": 1, "tasks": {}}
        self._persist(force=True)

    def _persist_if_needed(self) -> None:
        if self._dirty:
            self._persist()

    def _persist(self, force: bool = False) -> None:
        if not self.enabled:
            return
        if not self._dirty and not force:
            return

        state_dir = os.path.dirname(self.state_path)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)

        tmp_path = f"{self.state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, self.state_path)
        self._dirty = False

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

