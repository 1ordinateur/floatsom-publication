import numpy as np
import pytest

benchmark = pytest.importorskip("floatsom.benchmarks.run_sklearn_benchmarks")


def _parse_args(monkeypatch, argv):
    monkeypatch.setattr(benchmark.sys, "argv", argv, raising=True)
    return benchmark.parse_args()


def test_parse_args_accepts_rng_topology(monkeypatch):
    args = _parse_args(
        monkeypatch,
        [
            "run_sklearn_benchmarks.py",
            "--topology_type",
            "rng",
        ],
    )

    assert args.topology_type == "rng"


def test_parse_args_accepts_representative_topology_column(monkeypatch):
    args = _parse_args(
        monkeypatch,
        [
            "run_sklearn_benchmarks.py",
            "--representative_topology_column",
            "--representative_output_svg",
            "custom_representative.svg",
        ],
    )

    assert args.representative_topology_column is True
    assert args.representative_output_svg == "custom_representative.svg"


def test_train_floatsom_uses_graph_node_count_for_rng(monkeypatch, capsys):
    args = _parse_args(
        monkeypatch,
        [
            "run_sklearn_benchmarks.py",
            "--topology_type",
            "rng",
            "--mst_nodes",
            "7",
            "--iterations",
            "1",
            "--grid_size",
            "3",
            "--processing_method",
            "batch",
            "--sampling_method",
            "full",
            "--chunk_size",
            "8",
            "--force_cpu",
        ],
    )

    captured = {}

    class DummySOM:
        def train(self, data):
            assert data.shape == (12, 5)
            return {"iterations_completed": 1}

    def fake_create_floatsom(params):
        captured["params"] = params
        return DummySOM()

    monkeypatch.setattr(
        benchmark, "create_floatsom", fake_create_floatsom, raising=True
    )

    data = np.random.rand(12, 5).astype(np.float32)
    som, training_stats, train_time = benchmark.train_floatsom(
        data, args, metadata={"dataset_name": "dummy"}
    )

    assert isinstance(som, DummySOM)
    assert training_stats["iterations_completed"] == 1
    assert train_time >= 0.0

    params = captured["params"]
    assert params.topology_config.topology_type == "rng"
    assert params.topology_config.num_nodes == 7

    output = capsys.readouterr().out
    assert "rng (7 nodes)" in output


class _DummyTopologyConfig:
    topology_type = "rng"
    grid_size = 10


class _DummyParams:
    topology_config = _DummyTopologyConfig()


class _DummyTopology:
    name = "rng"
    adjacency_list = None


class _DummySOMForMetrics:
    def __init__(self, weights: np.ndarray):
        self._weights = weights
        self.params = _DummyParams()
        self.topology = _DummyTopology()

    def get_weights(self):
        return self._weights


def _cuda_available() -> bool:
    try:
        return benchmark.cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def test_distortion_measure_handles_graph_node_count_mismatch_cpu():
    rng = np.random.RandomState(123)
    weights = rng.rand(64, 2).astype(np.float32)
    data = rng.rand(24, 2).astype(np.float32)

    som = _DummySOMForMetrics(weights)
    som_wrapper = benchmark.FloatSOMWrapper(som)
    metric = benchmark.DistortionMeasure(use_gpu=False, batch_size=8)

    value = metric.compute(som_wrapper, data)
    assert np.isfinite(value)


def test_topographic_function_handles_graph_node_count_mismatch_cpu():
    rng = np.random.RandomState(234)
    weights = rng.rand(64, 2).astype(np.float32)
    data = rng.rand(24, 2).astype(np.float32)

    som = _DummySOMForMetrics(weights)
    som_wrapper = benchmark.FloatSOMWrapper(som)
    metric = benchmark.TopographicFunction(use_gpu=False, max_neurons=1000)

    value = metric.compute(som_wrapper, data)
    assert np.isfinite(value)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA device required for GPU trustworthiness metric")
def test_trustworthiness_handles_graph_node_count_mismatch():
    rng = np.random.RandomState(456)
    weights = rng.rand(64, 2).astype(np.float32)
    data = rng.rand(24, 2).astype(np.float32)

    som = _DummySOMForMetrics(weights)
    som_wrapper = benchmark.FloatSOMWrapper(som)
    metric = benchmark.Trustworthiness(k=3)

    value = metric.compute(som_wrapper, data)
    assert np.isfinite(value)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA device required for GPU neighborhood metric")
def test_neighborhood_preservation_handles_graph_node_count_mismatch():
    rng = np.random.RandomState(789)
    weights = rng.rand(64, 2).astype(np.float32)
    data = rng.rand(24, 2).astype(np.float32)

    som = _DummySOMForMetrics(weights)
    som_wrapper = benchmark.FloatSOMWrapper(som)
    metric = benchmark.NeighborhoodPreservation(
        k=3,
        use_approximate=False,
        batch_size=8,
    )

    value = metric.compute(som_wrapper, data)
    assert np.isfinite(value)


class _RepresentativeDummyTopology:
    def __init__(
        self,
        name: str,
        grid_size: int = 2,
        mst_edges=None,
        is_reformed: bool = False,
    ):
        self.name = name
        self.grid_size = grid_size
        self.mst_edges = list(mst_edges or [])
        self.is_reformed = is_reformed


class _RepresentativeDummySOM:
    def __init__(self, weights: np.ndarray, topology: _RepresentativeDummyTopology):
        self._weights = weights
        self.topology = topology

    def get_weights(self):
        return self._weights


def _make_representative_dummy_som(topology_name: str) -> _RepresentativeDummySOM:
    if topology_name == "hexagonal":
        weights = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        topology = _RepresentativeDummyTopology(name="hexagonal", grid_size=2)
        return _RepresentativeDummySOM(weights=weights, topology=topology)

    if topology_name in {"mst", "rng"}:
        weights = np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            dtype=np.float32,
        )
        edges = [(0, 1), (1, 2), (2, 3)]
        topology = _RepresentativeDummyTopology(name=topology_name, mst_edges=edges)
        return _RepresentativeDummySOM(weights=weights, topology=topology)

    raise AssertionError(f"Unexpected topology: {topology_name}")


def test_representative_projection_uses_shared_2d_basis():
    rng = np.random.RandomState(123)
    data = rng.randn(32, 5).astype(np.float32)
    shared_weights = rng.randn(8, 5).astype(np.float32)
    weights_by_topology = {
        "hexagonal": shared_weights.copy(),
        "mst": shared_weights.copy(),
        "rng": rng.randn(8, 5).astype(np.float32),
    }

    data_2d, weights_2d, axis_label_mode = benchmark._project_representative_coordinates(
        data_np=data,
        weights_by_topology=weights_by_topology,
    )

    assert data_2d.shape == (32, 2)
    assert weights_2d["hexagonal"].shape == (8, 2)
    assert weights_2d["mst"].shape == (8, 2)
    assert weights_2d["rng"].shape == (8, 2)
    assert axis_label_mode == "pca"
    assert np.allclose(weights_2d["hexagonal"], weights_2d["mst"])


def test_generate_representative_reuses_pretrained_and_writes_svg(monkeypatch, tmp_path):
    args = _parse_args(
        monkeypatch,
        [
            "run_sklearn_benchmarks.py",
            "--topology_type",
            "hexagonal",
            "--representative_topology_column",
            "--force_cpu",
        ],
    )
    output_svg = tmp_path / "representative.svg"
    args.representative_output_svg = str(output_svg)

    data = np.random.RandomState(456).randn(40, 2).astype(np.float32)
    metadata = {"dataset_name": "unit_test_dataset"}

    pre_trained = {"hexagonal": _make_representative_dummy_som("hexagonal")}
    train_calls = []

    def fake_train_floatsom(train_data, topology_args, train_metadata):
        assert train_data is data
        assert train_metadata is metadata
        topology = str(topology_args.topology_type)
        train_calls.append(topology)
        return _make_representative_dummy_som(topology), {"iterations_completed": 1}, 0.0

    monkeypatch.setattr(benchmark, "train_floatsom", fake_train_floatsom, raising=True)

    output_path = benchmark.generate_representative_topology_column_figure(
        data=data,
        metadata=metadata,
        args=args,
        output_dir=str(tmp_path),
        pre_trained_soms=pre_trained,
    )

    assert train_calls == ["mst", "rng"]
    assert output_path == str(output_svg)
    assert output_svg.exists()

    svg_text = output_svg.read_text(encoding="utf-8")
    assert svg_text.count("Common Legend") == 1
    assert "Hexagonal (nodes + connections)" in svg_text
    assert "MST (nodes + connections)" in svg_text
    assert "RNG (nodes + connections)" in svg_text
    assert "#FF4FA3" in svg_text
    assert "#00E5FF" in svg_text
    assert "#8A2BE2" in svg_text
    assert svg_text.find("Hexagonal") < svg_text.find("MST") < svg_text.find("RNG")
