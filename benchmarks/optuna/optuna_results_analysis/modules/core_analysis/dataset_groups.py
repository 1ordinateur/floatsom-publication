"""
Dataset grouping utilities for Optuna analysis outputs.

Defines canonical synthetic vs real dataset groupings and provides
helpers for ordering and labeling in reports/figures.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


SYNTHETIC_DATASETS: tuple[str, ...] = (
    "blobs",
    "circles",
    "moons",
    "s_curve",
    "swiss_roll",
)

REAL_DATASET_ORDER: tuple[str, ...] = (
    "breast_cancer",
    "wine",
    "iris",
    "digits",
    "olivetti_faces",
    "diabetes",
    "california_housing",
    "covertype",
    "kddcup99",
    "lfw_people",
)

GLOBAL_SYNTHETIC_LABEL = "GLOBAL_SYNTHETIC"
GLOBAL_REAL_LABEL = "GLOBAL_REAL"
GLOBAL_OVERALL_LABEL = "GLOBAL_OVERALL"

CANONICAL_GLOBAL_LABELS: tuple[str, ...] = (
    GLOBAL_SYNTHETIC_LABEL,
    GLOBAL_REAL_LABEL,
    GLOBAL_OVERALL_LABEL,
)
GLOBAL_LABELS: tuple[str, ...] = CANONICAL_GLOBAL_LABELS
GLOBAL_LABEL_ORDER: tuple[str, ...] = CANONICAL_GLOBAL_LABELS

DATASET_GROUP_ORDER: tuple[str, ...] = ("synthetic", "real", "global")
SUMMARY_GROUPS: tuple[str, ...] = ("synthetic", "real", "overall")
SUMMARY_LABELS: Dict[str, str] = {
    "synthetic": GLOBAL_SYNTHETIC_LABEL,
    "real": GLOBAL_REAL_LABEL,
    "overall": GLOBAL_OVERALL_LABEL,
}


def _normalize_key(value: object) -> str:
    return str(value).strip().lower()


def is_global_dataset_label(value: object) -> bool:
    key = _normalize_key(value)
    return key in {label.lower() for label in GLOBAL_LABELS}


def dataset_group(value: object) -> str:
    """Return 'synthetic' or 'real' for known datasets; global labels return 'global'."""
    key = _normalize_key(value)
    if key in {label.lower() for label in GLOBAL_LABELS}:
        return "global"
    if key in {name.lower() for name in SYNTHETIC_DATASETS}:
        return "synthetic"
    return "real"


def ordered_dataset_labels(dataset_values: Iterable[object]) -> List[str]:
    """Return ordered dataset labels with synthetic then real, followed by globals."""
    mapping: Dict[str, str] = {}
    for value in dataset_values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        mapping.setdefault(key, text)

    if not mapping:
        return []

    synthetic_keys = [name.lower() for name in SYNTHETIC_DATASETS if name.lower() in mapping]
    real_keys = [name.lower() for name in REAL_DATASET_ORDER if name.lower() in mapping]

    known_keys = set(synthetic_keys) | set(real_keys)
    extra_keys = sorted(
        key for key in mapping.keys() if key not in known_keys and key not in {label.lower() for label in GLOBAL_LABELS}
    )

    synthetic_labels = [mapping[key] for key in synthetic_keys]
    real_labels = [mapping[key] for key in real_keys]
    extra_labels = [mapping[key] for key in extra_keys]

    global_labels: List[str] = []
    for label in GLOBAL_LABEL_ORDER:
        key = label.lower()
        if key in mapping:
            global_labels.append(mapping[key])

    return synthetic_labels + real_labels + extra_labels + global_labels
