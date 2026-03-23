"""
Regression tests for shared resource limit helpers.

These tests are CPU-only (no Ray/CuPy dependency) and lock in behaviour used by
Ray pipeline memory guards.
"""

from __future__ import annotations

from floatsom.processing.ray_ops import resource_limits


def test_parse_memory_value_parses_units():
    assert resource_limits.parse_memory_value("1GB") == 1024**3
    assert resource_limits.parse_memory_value("1.5GB") == int(1.5 * 1024**3)
    assert resource_limits.parse_memory_value("2048MB") == 2048 * 1024**2
    assert resource_limits.parse_memory_value("4096") == 4096
    assert resource_limits.parse_memory_value("") is None


def test_job_memory_limit_reads_pbs_resource_list_mem_lowercase():
    env = {"PBS_RESOURCE_LIST_mem": "10GB"}
    assert resource_limits.get_job_memory_limit_bytes(env=env, cgroup_paths=()) == 10 * 1024**3


def test_job_memory_limit_reads_slurm_mem_per_cpu_digits_as_mb():
    env = {"SLURM_MEM_PER_CPU": "4096"}
    assert (
        resource_limits.get_job_memory_limit_bytes(env=env, cgroup_paths=())
        == 4096 * 1024**2
    )


def test_job_memory_limit_returns_none_without_env_or_cgroup():
    assert resource_limits.get_job_memory_limit_bytes(env={}, cgroup_paths=()) is None

