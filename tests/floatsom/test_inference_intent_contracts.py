import importlib
import sys
import types

import numpy as np
import pytest

class _FakeStream:
    def __init__(self, non_blocking=True):
        self.non_blocking = bool(non_blocking)
        self.ptr = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False

    def synchronize(self):
        return None


class _FakeCP:
    float32 = np.float32
    int32 = np.int32

    class _FakeCUDA:
        Stream = _FakeStream

    cuda = _FakeCUDA()

    @staticmethod
    def empty(shape, dtype):
        return np.empty(shape, dtype=dtype)

    @staticmethod
    def asnumpy(array):
        return np.asarray(array)


class _StubDataSource:
    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float32)

    def get_reference(self):
        return self._data


class _FakeLoader:
    instances = []

    def __init__(self, *, ram_data=None, chunk_size, **kwargs):
        del kwargs
        self._data = np.asarray(ram_data, dtype=np.float32)
        self.chunk_size = int(chunk_size)
        self.get_chunk_calls = []
        self.mark_chunk_consumed_calls = []
        self.cleanup_calls = 0
        self.__class__.instances.append(self)

    def get_chunk(self, chunk_index, use_randomized_order=False, *, advance_cursor):
        chunk_idx = int(chunk_index)
        self.get_chunk_calls.append((chunk_idx, bool(use_randomized_order), bool(advance_cursor)))
        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, int(self._data.shape[0]))
        return self._data[start:end], {"chunk_index": chunk_idx}

    def mark_chunk_consumed(self, chunk_index):
        self.mark_chunk_consumed_calls.append(int(chunk_index))

    def cleanup(self):
        self.cleanup_calls += 1


def _make_import_time_fake_cupy_module():
    # Minimal surface needed for import-time annotations in no-CuPy environments.
    fake_cupy = types.ModuleType("cupy")

    class _ImportTimeFakeStream:
        pass

    fake_cupy.ndarray = np.ndarray
    fake_cupy.cuda = types.SimpleNamespace(Stream=_ImportTimeFakeStream)
    return fake_cupy


def _load_inference_modules():
    inference_module = importlib.import_module("floatsom.base.inference")
    loader_module = importlib.import_module("floatsom.data.cpugpu_fast_loader")
    return inference_module, loader_module


@pytest.fixture
def _inference_harness_cls(monkeypatch):
    _FakeLoader.instances = []

    prior_inference_module = sys.modules.get("floatsom.base.inference")
    prior_loader_module = sys.modules.get("floatsom.data.cpugpu_fast_loader")

    try:
        importlib.import_module("cupy")
    except ModuleNotFoundError:
        monkeypatch.setitem(sys.modules, "cupy", _make_import_time_fake_cupy_module())

    inference_module, loader_module = _load_inference_modules()
    monkeypatch.setattr(inference_module, "cp", _FakeCP)
    monkeypatch.setattr(loader_module, "CPUGPUFastLoader", _FakeLoader)

    class _InferenceHarnessImpl(inference_module.InferenceMixin):
        def predict(self, data):
            chunk = np.asarray(data)
            chunk_idx = int(chunk[0, 0]) if chunk.size else -1
            return np.full(chunk.shape[0], chunk_idx, dtype=np.int32)

    yield _InferenceHarnessImpl

    if prior_inference_module is None:
        sys.modules.pop("floatsom.base.inference", None)
    else:
        sys.modules["floatsom.base.inference"] = prior_inference_module
    if prior_loader_module is None:
        sys.modules.pop("floatsom.data.cpugpu_fast_loader", None)
    else:
        sys.modules["floatsom.data.cpugpu_fast_loader"] = prior_loader_module


def test_infer_gpu_streamed_uses_explicit_prefetch_and_consume_intents(_inference_harness_cls):
    model = _inference_harness_cls()
    samples = np.arange(20, dtype=np.float32).reshape(10, 2)
    output = np.empty(10, dtype=np.int32)

    result = model._infer_gpu_streamed(
        _StubDataSource(samples),
        output_array=output,
        zarr_output_array=None,
        zarr_output_path=None,
        chunk_size=4,
        n_samples=10,
        n_features=2,
        verbose=False,
    )

    assert result is output
    loader = _FakeLoader.instances[-1]
    assert loader.get_chunk_calls == [
        (0, False, True),
        (1, False, False),
        (2, False, False),
    ]
    assert loader.mark_chunk_consumed_calls == [1, 2]
    assert loader.cleanup_calls == 1
    np.testing.assert_array_equal(output, np.array([0, 0, 0, 0, 8, 8, 8, 8, 16, 16], dtype=np.int32))


def test_infer_gpu_simple_uses_consume_intent_for_every_chunk(_inference_harness_cls):
    model = _inference_harness_cls()
    samples = np.arange(20, dtype=np.float32).reshape(10, 2)
    output = np.empty(10, dtype=np.int32)

    result = model._infer_gpu_simple(
        _StubDataSource(samples),
        output_array=output,
        zarr_output_array=None,
        zarr_output_path=None,
        chunk_size=4,
        n_samples=10,
        verbose=False,
    )

    assert result is output
    loader = _FakeLoader.instances[-1]
    assert loader.get_chunk_calls == [
        (0, False, True),
        (1, False, True),
        (2, False, True),
    ]
    assert loader.mark_chunk_consumed_calls == []
    assert loader.cleanup_calls == 1
    np.testing.assert_array_equal(output, np.array([0, 0, 0, 0, 8, 8, 8, 8, 16, 16], dtype=np.int32))
