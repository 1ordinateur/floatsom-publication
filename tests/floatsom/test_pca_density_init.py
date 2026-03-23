import numpy as np

from floatsom.topology import pca_initialization


class _DummyPCA:
    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, data: np.ndarray):
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data[:, : self.n_components]


class _DummyKDE:
    def __init__(self, bandwidth: str):
        self.bandwidth = bandwidth

    def fit(self, data: np.ndarray):
        return self

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] > 100000:
            raise AssertionError("Density scoring should run on a capped working set")
        return np.zeros(data.shape[0], dtype=np.float64)


def test_pca_density_init_subsamples_large_input(monkeypatch):
    monkeypatch.setattr(pca_initialization, "SKLEARN_AVAILABLE", True)
    monkeypatch.setattr(pca_initialization, "PCA", _DummyPCA)
    monkeypatch.setattr(pca_initialization, "KernelDensity", _DummyKDE)

    rng = np.random.default_rng(7)
    data = rng.normal(size=(100500, 4)).astype(np.float32)

    weights = pca_initialization.pca_density_init(
        data=data,
        grid_shape=(20, 20),
        xp=np,
        verbose=False,
    )

    assert weights.shape == (20, 20, 4)
    assert np.isfinite(weights).all()
