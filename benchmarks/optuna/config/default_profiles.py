"""Resolve named library profiles, the comparison baseline, or custom JSON files."""
from pathlib import Path
from floatsom.defaults import PROFILE_NAMES, profile_path

LEGACY_NAMES = {
    "floatsom_min1000_tuned_defaults.json": "publication",
    "floatsom_min1000_tuned_defaults_rng_random.json": "publication-rng-random",
    "floatsom_min1000_tuned_defaults_rng_mst_config.json": "publication-rng-mst-config",
    "xpysom_untuned_defaults.json": "xpysom-untuned",
}


def resolve_defaults_path(value: str) -> Path:
    """Explicit files win; named profiles are independent of working directory."""
    path = Path(value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path(__file__).resolve().parents[3] / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    name = LEGACY_NAMES.get(value, value)
    if name in PROFILE_NAMES:
        return profile_path(name)
    if name == "xpysom-untuned":
        return Path(__file__).with_name("baselines") / "xpysom-untuned.json"
    return path.resolve()
