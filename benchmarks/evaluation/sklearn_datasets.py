"""
Sklearn toy dataset generation for SOM benchmarking.
"""

import os
from pathlib import Path
from functools import lru_cache
from urllib.error import URLError
import numpy as np
import cupy as cp
from typing import Tuple, Optional, Dict, Any, List
import inspect

try:
    from sklearn.datasets import (
        make_swiss_roll, make_s_curve, make_moons, 
        make_circles, make_blobs,
        load_breast_cancer, load_wine, load_iris, 
        load_digits, load_diabetes, fetch_olivetti_faces,
        fetch_california_housing, fetch_covtype, fetch_kddcup99,
        fetch_lfw_people
    )
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


DEFAULT_OLIVETTI_CACHE_CANDIDATES = (
    "/g/data/eu59/SIFEAN/sklearn_data",
    "/g/data/eu59/SIFEAN/sfa/sklearn_data",
)


def _resolve_sklearn_cache_candidates() -> List[str]:
    candidates: List[str] = []
    env_home = os.environ.get("SCIKIT_LEARN_DATA")
    if env_home:
        candidates.append(env_home)
    for candidate in DEFAULT_OLIVETTI_CACHE_CANDIDATES:
        path = Path(candidate)
        if path.exists() or path.parent.exists():
            candidates.append(str(path))
    candidates.append(str(Path.home() / "scikit_learn_data"))
    seen = set()
    ordered: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


@lru_cache(maxsize=1)
def _load_olivetti_faces(seed: Optional[int]):
    data_home = os.environ.get("SCIKIT_LEARN_DATA")
    try:
        return fetch_olivetti_faces(
            data_home=data_home,
            shuffle=False,
            random_state=seed,
            download_if_missing=True,
            return_X_y=False,
        )
    except (URLError, OSError) as exc:
        last_exc: Exception = exc
        for candidate in _resolve_sklearn_cache_candidates():
            try:
                return fetch_olivetti_faces(
                    data_home=candidate,
                    shuffle=False,
                    random_state=seed,
                    download_if_missing=False,
                    return_X_y=False,
                )
            except (URLError, OSError) as fallback_exc:
                last_exc = fallback_exc
        raise RuntimeError(
            "Failed to load olivetti_faces. Network download failed and no local cache was found. "
            "Set SCIKIT_LEARN_DATA or populate one of the default cache locations."
        ) from last_exc


def _filter_fetch_kwargs(fetcher: object, kwargs: Dict[str, object]) -> Dict[str, object]:
    """Drop kwargs not supported by the sklearn fetcher signature."""
    try:
        parameters = set(inspect.signature(fetcher).parameters)
    except (TypeError, ValueError):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def _load_sklearn_fetched_dataset(fetcher: object, *, seed: Optional[int] = None, **kwargs: object):
    """
    Load a sklearn fetch_* dataset with the same offline-cache probing logic as olivetti_faces.

    Policy:
    - Try `SCIKIT_LEARN_DATA` first, allowing network download if supported.
    - If download fails, probe known cache candidates with download disabled.
    - If no cache exists, raise a clear error telling the user where to populate it.
    """
    data_home = os.environ.get("SCIKIT_LEARN_DATA")
    primary_kwargs: Dict[str, object] = {"data_home": data_home, **kwargs}
    if seed is not None:
        primary_kwargs.setdefault("random_state", seed)

    # First attempt: allow download when the fetcher supports it.
    try:
        call_kwargs = dict(primary_kwargs)
        call_kwargs.setdefault("download_if_missing", True)
        call_kwargs = _filter_fetch_kwargs(fetcher, call_kwargs)
        return fetcher(**call_kwargs)
    except (URLError, OSError) as exc:
        last_exc: Exception = exc
        for candidate in _resolve_sklearn_cache_candidates():
            try:
                call_kwargs = dict(primary_kwargs)
                call_kwargs["data_home"] = candidate
                call_kwargs.setdefault("download_if_missing", False)
                call_kwargs = _filter_fetch_kwargs(fetcher, call_kwargs)
                return fetcher(**call_kwargs)
            except (URLError, OSError) as fallback_exc:
                last_exc = fallback_exc
        raise RuntimeError(
            "Failed to load sklearn fetched dataset. Network download failed and no local cache was found. "
            "Set SCIKIT_LEARN_DATA or populate one of the default cache locations."
        ) from last_exc


def _encode_categorical_vector(values: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    classes, encoded = np.unique(values, return_inverse=True)
    return encoded.astype(np.int64), [str(item) for item in classes]


def _encode_categorical_columns(values: np.ndarray) -> Tuple[np.ndarray, Dict[int, List[str]]]:
    if values.ndim != 2:
        return values.astype(np.float32), {}
    encoded = values.astype(object, copy=True)
    mappings: Dict[int, List[str]] = {}
    for col_idx in range(encoded.shape[1]):
        column = encoded[:, col_idx]
        try:
            encoded[:, col_idx] = column.astype(np.float32)
        except (ValueError, TypeError):
            column_encoded, column_labels = _encode_categorical_vector(column)
            encoded[:, col_idx] = column_encoded
            mappings[col_idx] = column_labels
    return encoded.astype(np.float32), mappings


# Dataset configuration with difficulty levels
DATASET_CONFIGS = {
    'swiss_roll': {
        'easy': {'n_samples': 1000, 'noise': 0.05, 'hole': False},
        'medium': {'n_samples': 3000, 'noise': 0.1, 'hole': False},
        'hard': {'n_samples': 30000, 'noise': 0.2, 'hole': True}
    },
    'moons': {
        'easy': {'n_samples': 1000, 'noise': 0.05},
        'medium': {'n_samples': 3000, 'noise': 0.1},
        'hard': {'n_samples': 30000, 'noise': 0.2}
    },
    'circles': {
        'easy': {'n_samples': 1000, 'noise': 0.05, 'factor': 0.6},
        'medium': {'n_samples': 3000, 'noise': 0.1, 'factor': 0.4},
        'hard': {'n_samples': 30000, 'noise': 0.2, 'factor': 0.2}
    },
    'blobs': {
        'easy': {'n_samples': 1000, 'centers': 3, 'cluster_std': 0.5},
        'medium': {'n_samples': 3000, 'centers': 5, 'cluster_std': 1.0},
        'hard': {'n_samples': 30000, 'centers': 8, 'cluster_std': 2.0}
    },
    's_curve': {
        'easy': {'n_samples': 1000, 'noise': 0.05},
        'medium': {'n_samples': 3000, 'noise': 0.1},
        'hard': {'n_samples': 30000, 'noise': 0.2}
    },
    # Real-world datasets - no configuration needed, just difficulty levels
    'breast_cancer': {
        'easy': {},
        'medium': {},
        'hard': {}
    },
    'wine': {
        'easy': {},
        'medium': {},
        'hard': {}
    },
    'iris': {
        'easy': {},
        'medium': {},
        'hard': {}
    },
    'digits': {
        'easy': {},
        'medium': {},
        'hard': {}
    },
    'olivetti_faces': {
        'easy': {},
        'medium': {},
        'hard': {}
    },
    'diabetes': {
        'easy': {},
        'medium': {},
        'hard': {}
    },
    'california_housing': {
        'easy': {},
        'medium': {},
        'hard': {}
    },
    'covertype': {
        'easy': {},
        'medium': {},
        'hard': {}
    },
    'kddcup99': {
        'easy': {},
        'medium': {},
        'hard': {}
    },
    'lfw_people': {
        'easy': {},
        'medium': {},
        'hard': {}
    }
}


def generate_sklearn_dataset(
    dataset_name: str,
    difficulty: str = 'medium',
    seed: Optional[int] = None,
    n_features: Optional[int] = None,
    normalize: bool = True,
    centers: Optional[int] = None,
) -> Tuple[cp.ndarray, Dict[str, Any]]:
    """
    Generate sklearn toy dataset for SOM benchmarking.
    
    Args:
        dataset_name: Name of dataset ('swiss_roll', 'moons', 'circles', 'blobs', 's_curve', 
                     'breast_cancer', 'wine', 'iris', 'digits', 'diabetes', 'olivetti_faces',
                     'california_housing', 'covertype', 'kddcup99', 'lfw_people')
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        seed: Random seed for reproducibility
        n_features: For 'blobs' dataset, number of features (overrides default)
        normalize: Whether to normalize the data
        
    Returns:
        Tuple of (data, metadata) where data is cupy array and metadata contains info
        
    Raises:
        ImportError: If scikit-learn is not available
        ValueError: If dataset_name or difficulty is invalid
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for toy datasets")
    
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    if difficulty not in DATASET_CONFIGS[dataset_name]:
        raise ValueError(f"Unknown difficulty: {difficulty}. Available: {list(DATASET_CONFIGS[dataset_name].keys())}")
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # Get configuration
    config = DATASET_CONFIGS[dataset_name][difficulty].copy()
    
    # Overrides for 'blobs'
    if dataset_name == 'blobs':
        if n_features is not None:
            config['n_features'] = n_features
        if centers is not None:
            config['centers'] = centers
    
    # Generate dataset
    metadata = {
        'dataset_name': dataset_name,
        'difficulty': difficulty,
        'config': config.copy(),
        'seed': seed
    }
    
    if dataset_name == 'swiss_roll':
        X, color = make_swiss_roll(random_state=seed, **config)
        metadata['intrinsic_dim'] = 2
        metadata['embedding_dim'] = 3
        metadata['color'] = color
        
    elif dataset_name == 's_curve':
        X, color = make_s_curve(random_state=seed, **config)
        metadata['intrinsic_dim'] = 1
        metadata['embedding_dim'] = 3
        metadata['color'] = color
        
    elif dataset_name == 'moons':
        X, y = make_moons(random_state=seed, **config)
        metadata['intrinsic_dim'] = 2
        metadata['embedding_dim'] = 2
        metadata['labels'] = y
        
    elif dataset_name == 'circles':
        X, y = make_circles(random_state=seed, **config)
        metadata['intrinsic_dim'] = 2
        metadata['embedding_dim'] = 2
        metadata['labels'] = y
        
    elif dataset_name == 'blobs':
        # Default to 2D for visualization if not specified
        if 'n_features' not in config:
            config['n_features'] = 2
        X, y = make_blobs(random_state=seed, **config)
        metadata['intrinsic_dim'] = config['n_features']
        metadata['embedding_dim'] = config['n_features']
        metadata['labels'] = y
        metadata['n_clusters'] = config['centers']
        
    # Real-world datasets - use standard sklearn parameters
    elif dataset_name == 'breast_cancer':
        dataset = load_breast_cancer(return_X_y=False, as_frame=False)
        X, y = dataset.data, dataset.target
        metadata['intrinsic_dim'] = X.shape[1]
        metadata['embedding_dim'] = X.shape[1]
        metadata['labels'] = y
        metadata['n_classes'] = len(np.unique(y))
        metadata['feature_names'] = list(dataset.feature_names) if dataset.feature_names is not None else []
        metadata['target_names'] = list(dataset.target_names) if dataset.target_names is not None else []
        metadata['description'] = dataset.DESCR
        metadata['filename'] = dataset.filename if hasattr(dataset, 'filename') else None
        
    elif dataset_name == 'wine':
        dataset = load_wine(return_X_y=False, as_frame=False)
        X, y = dataset.data, dataset.target
        metadata['intrinsic_dim'] = X.shape[1]
        metadata['embedding_dim'] = X.shape[1]
        metadata['labels'] = y
        metadata['n_classes'] = len(np.unique(y))
        metadata['feature_names'] = list(dataset.feature_names) if dataset.feature_names is not None else []
        metadata['target_names'] = list(dataset.target_names) if dataset.target_names is not None else []
        metadata['description'] = dataset.DESCR
        metadata['filename'] = dataset.filename if hasattr(dataset, 'filename') else None
        
    elif dataset_name == 'iris':
        dataset = load_iris(return_X_y=False, as_frame=False)
        X, y = dataset.data, dataset.target
        metadata['intrinsic_dim'] = X.shape[1]
        metadata['embedding_dim'] = X.shape[1]
        metadata['labels'] = y
        metadata['n_classes'] = len(np.unique(y))
        metadata['feature_names'] = list(dataset.feature_names) if dataset.feature_names is not None else []
        metadata['target_names'] = list(dataset.target_names) if dataset.target_names is not None else []
        metadata['description'] = dataset.DESCR
        metadata['filename'] = dataset.filename if hasattr(dataset, 'filename') else None
        
    elif dataset_name == 'digits':
        dataset = load_digits(return_X_y=False, as_frame=False)
        X, y = dataset.data, dataset.target
        metadata['intrinsic_dim'] = X.shape[1]
        metadata['embedding_dim'] = X.shape[1]
        metadata['labels'] = y
        metadata['n_classes'] = len(np.unique(y))
        metadata['target_names'] = [f'digit_{i}' for i in range(10)]
        metadata['description'] = dataset.DESCR
        metadata['images'] = dataset.images  # Keep original 8x8 image format
        
    elif dataset_name == 'diabetes':
        dataset = load_diabetes(return_X_y=False, as_frame=False, scaled=True)
        X, y = dataset.data, dataset.target
        metadata['intrinsic_dim'] = X.shape[1]
        metadata['embedding_dim'] = X.shape[1]
        metadata['labels'] = y
        metadata['regression_task'] = True  # This is a regression dataset
        metadata['feature_names'] = list(dataset.feature_names) if dataset.feature_names is not None else []
        metadata['description'] = dataset.DESCR
        metadata['filename'] = dataset.filename if hasattr(dataset, 'filename') else None

    elif dataset_name == 'california_housing':
        dataset = _load_sklearn_fetched_dataset(fetch_california_housing, seed=seed)
        X, y = dataset.data, dataset.target
        metadata['intrinsic_dim'] = X.shape[1]
        metadata['embedding_dim'] = X.shape[1]
        metadata['labels'] = y
        metadata['regression_task'] = True
        metadata['feature_names'] = list(dataset.feature_names) if dataset.feature_names is not None else []
        metadata['description'] = dataset.DESCR
        metadata['filename'] = dataset.filename if hasattr(dataset, 'filename') else None

    elif dataset_name == 'covertype':
        dataset = _load_sklearn_fetched_dataset(fetch_covtype, seed=seed)
        X, y = dataset.data, dataset.target
        metadata['intrinsic_dim'] = X.shape[1]
        metadata['embedding_dim'] = X.shape[1]
        metadata['labels'] = y
        metadata['n_classes'] = len(np.unique(y))
        metadata['feature_names'] = list(dataset.feature_names) if getattr(dataset, 'feature_names', None) is not None else []
        metadata['target_names'] = list(dataset.target_names) if getattr(dataset, 'target_names', None) is not None else []
        metadata['description'] = dataset.DESCR

    elif dataset_name == 'kddcup99':
        dataset = _load_sklearn_fetched_dataset(fetch_kddcup99, seed=seed)
        X, y = dataset.data, dataset.target
        categorical_mappings: Dict[int, List[str]] = {}
        if X.dtype.kind in {'O', 'S', 'U'}:
            X, categorical_mappings = _encode_categorical_columns(X)
            metadata['categorical_mappings'] = categorical_mappings
        if y.dtype.kind in {'O', 'S', 'U'}:
            y, label_names = _encode_categorical_vector(y)
            metadata['label_names'] = label_names
        metadata['intrinsic_dim'] = X.shape[1]
        metadata['embedding_dim'] = X.shape[1]
        metadata['labels'] = y
        metadata['n_classes'] = len(np.unique(y))
        metadata['feature_names'] = list(dataset.feature_names) if getattr(dataset, 'feature_names', None) is not None else []
        metadata['description'] = dataset.DESCR

    elif dataset_name == 'lfw_people':
        dataset = _load_sklearn_fetched_dataset(fetch_lfw_people, seed=seed)
        X, y = dataset.data, dataset.target
        metadata['intrinsic_dim'] = X.shape[1]
        metadata['embedding_dim'] = X.shape[1]
        metadata['labels'] = y
        metadata['n_classes'] = len(np.unique(y))
        metadata['target_names'] = list(dataset.target_names) if dataset.target_names is not None else []
        metadata['description'] = dataset.DESCR
        metadata['images'] = dataset.images

    elif dataset_name == 'olivetti_faces':
        dataset = _load_olivetti_faces(seed)
        X, y = dataset.data, dataset.target
        metadata['intrinsic_dim'] = X.shape[1]
        metadata['embedding_dim'] = X.shape[1]
        metadata['labels'] = y
        metadata['n_classes'] = len(np.unique(y))
        metadata['description'] = dataset.DESCR
        metadata['images'] = dataset.images  # Keep original 64x64 image format
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        metadata['normalized'] = True
        metadata['scaler_mean'] = scaler.mean_
        metadata['scaler_scale'] = scaler.scale_
    else:
        metadata['normalized'] = False
    
    # Convert to cupy and ensure float32
    X_gpu = cp.array(X, dtype=cp.float32)
    
    metadata['shape'] = X_gpu.shape
    metadata['dtype'] = str(X_gpu.dtype)
    
    return X_gpu, metadata


def get_available_datasets() -> Dict[str, list]:
    """
    Get list of available sklearn datasets and their difficulty levels.
    
    Returns:
        Dictionary mapping dataset names to available difficulty levels
    """
    if not SKLEARN_AVAILABLE:
        return {}
    
    return {name: list(configs.keys()) for name, configs in DATASET_CONFIGS.items()}


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with dataset information
        
    Raises:
        ValueError: If dataset_name is invalid
    """
    if not SKLEARN_AVAILABLE:
        return {}
    
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    info = {
        'name': dataset_name,
        'difficulties': list(DATASET_CONFIGS[dataset_name].keys()),
        'configs': DATASET_CONFIGS[dataset_name]
    }
    
    # Add dataset-specific information
    if dataset_name == 'swiss_roll':
        info['description'] = '3D Swiss roll manifold dataset'
        info['intrinsic_dim'] = 2
        info['embedding_dim'] = 3
        info['has_color'] = True
        
    elif dataset_name == 's_curve':
        info['description'] = '3D S-shaped curve manifold dataset'
        info['intrinsic_dim'] = 1
        info['embedding_dim'] = 3
        info['has_color'] = True
        
    elif dataset_name == 'moons':
        info['description'] = '2D binary classification with crescent shapes'
        info['intrinsic_dim'] = 2
        info['embedding_dim'] = 2
        info['has_labels'] = True
        
    elif dataset_name == 'circles':
        info['description'] = '2D binary classification with concentric circles'
        info['intrinsic_dim'] = 2
        info['embedding_dim'] = 2
        info['has_labels'] = True
        
    elif dataset_name == 'blobs':
        info['description'] = 'Multi-dimensional Gaussian clusters'
        info['intrinsic_dim'] = 'configurable'
        info['embedding_dim'] = 'configurable'
        info['has_labels'] = True
        
    # For real-world datasets, load them to get actual metadata
    elif dataset_name in [
        'breast_cancer', 'wine', 'iris', 'digits', 'diabetes', 'olivetti_faces',
        'california_housing', 'covertype', 'kddcup99', 'lfw_people'
    ]:
        try:
            if dataset_name == 'breast_cancer':
                dataset = load_breast_cancer()
            elif dataset_name == 'wine':
                dataset = load_wine()
            elif dataset_name == 'iris':
                dataset = load_iris()
            elif dataset_name == 'digits':
                dataset = load_digits()
            elif dataset_name == 'diabetes':
                dataset = load_diabetes()
            elif dataset_name == 'olivetti_faces':
                dataset = fetch_olivetti_faces()
            elif dataset_name == 'california_housing':
                dataset = _load_sklearn_fetched_dataset(fetch_california_housing)
            elif dataset_name == 'covertype':
                dataset = _load_sklearn_fetched_dataset(fetch_covtype)
            elif dataset_name == 'kddcup99':
                dataset = _load_sklearn_fetched_dataset(fetch_kddcup99)
            elif dataset_name == 'lfw_people':
                dataset = _load_sklearn_fetched_dataset(fetch_lfw_people)
            
            X, y = dataset.data, dataset.target
            info['description'] = dataset.DESCR.split('\n')[0]  # First line of description
            info['intrinsic_dim'] = X.shape[1]
            info['embedding_dim'] = X.shape[1]
            info['has_labels'] = True
            
            # Handle regression vs classification datasets
            if dataset_name in {'diabetes', 'california_housing'}:
                info['regression_task'] = True
                info['target_range'] = (float(np.min(y)), float(np.max(y)))
            else:
                info['n_classes'] = len(np.unique(y))
                info['regression_task'] = False
                
            info['n_samples'] = X.shape[0]
            info['has_feature_names'] = hasattr(dataset, 'feature_names') and dataset.feature_names is not None
            info['has_target_names'] = hasattr(dataset, 'target_names') and dataset.target_names is not None
        except Exception as e:
            info['description'] = f'Real-world dataset (error loading: {e})'
            info['has_labels'] = True
        
    return info
