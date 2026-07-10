#!/usr/bin/env python3
"""Analyze matched radius-sweep runs and render topology response lines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.machinery
import json
import math
from pathlib import Path
import sys
import types
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in {None, ""} and "floatsom" not in sys.modules:
    _repo_root = Path(__file__).resolve().parents[2]
    _pkg = types.ModuleType("floatsom")
    _pkg.__file__ = str(_repo_root / "__init__.py")
    _pkg.__path__ = [str(_repo_root)]
    _pkg.__package__ = "floatsom"
    _pkg.__spec__ = importlib.machinery.ModuleSpec("floatsom", loader=None, is_package=True)
    _pkg.__spec__.submodule_search_locations = _pkg.__path__
    sys.modules["floatsom"] = _pkg

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PAIR_KEY_COLUMNS: Tuple[str, ...] = ("dataset", "seed", "architecture", "sampling_method")
RUN_KEY_COLUMNS: Tuple[str, ...] = (*PAIR_KEY_COLUMNS, "initial_radius")
TOPOLOGY_ORDER: Tuple[str, ...] = ("hexagonal", "mst", "rng")
TOPOLOGY_STYLES: Dict[str, Dict[str, str]] = {
    "hexagonal": {"label": "Hexagonal", "color": "#4C78A8", "marker": "o"},
    "mst": {"label": "MST", "color": "#F58518", "marker": "s"},
    "rng": {"label": "RNG", "color": "#54A24B", "marker": "^"},
}
METRIC_SPECS: Tuple[Tuple[str, str, str, bool], ...] = (
    ("balanced_qe_raw", "Balanced QE", "balanced_qe", False),
    ("balanced_mean_tied_rank_raw", "Balanced MTR", "balanced_mtr", False),
    ("balanced_node_utilization_raw", "Balanced node utilisation", "balanced_node_utilisation", True),
)


def _load_manifest(path_value: str | Path) -> Tuple[Dict[str, object], Path]:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Radius-sweep manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Radius-sweep manifest must contain a JSON object: {path}")
    runs_value = payload.get("runs_csv")
    if not runs_value:
        raise ValueError("Radius-sweep manifest does not define runs_csv.")
    runs_path = Path(str(runs_value)).expanduser()
    if not runs_path.exists():
        local_candidate = path.parent / runs_path.name
        if local_candidate.exists():
            runs_path = local_candidate
        else:
            raise FileNotFoundError(f"Runs CSV referenced by manifest does not exist: {runs_value}")
    return payload, runs_path.resolve()


def _canonicalize_runs(df: pd.DataFrame) -> pd.DataFrame:
    required = set(RUN_KEY_COLUMNS).union(metric for metric, _, _, _ in METRIC_SPECS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Radius-sweep runs CSV is missing required columns: {missing}")

    working = df.copy()
    for column in ("dataset", "architecture", "sampling_method"):
        working[column] = working[column].astype(str).str.strip().str.lower()
    working["seed"] = pd.to_numeric(working["seed"], errors="raise").astype(np.int64)
    working["initial_radius"] = pd.to_numeric(working["initial_radius"], errors="raise").astype(float)
    for metric, _, _, _ in METRIC_SPECS:
        working[metric] = pd.to_numeric(working[metric], errors="coerce")
        if not np.isfinite(working[metric].to_numpy(dtype=float)).all():
            raise ValueError(f"Metric column contains non-finite values: {metric}")
    duplicates = working.duplicated(list(RUN_KEY_COLUMNS), keep=False)
    if bool(duplicates.any()):
        duplicate_keys = working.loc[duplicates, list(RUN_KEY_COLUMNS)].head(5).to_dict(orient="records")
        raise ValueError(f"Duplicate radius-sweep run keys found: {duplicate_keys}")
    return working.sort_values(list(RUN_KEY_COLUMNS), kind="mergesort").reset_index(drop=True)


def _validate_coverage(df: pd.DataFrame, manifest: Dict[str, object]) -> None:
    dimensions = {
        "dataset": [str(value).strip().lower() for value in manifest.get("datasets", [])],
        "seed": [int(value) for value in manifest.get("seeds", [])],
        "architecture": [str(value).strip().lower() for value in manifest.get("topologies", [])],
        "sampling_method": [str(value).strip().lower() for value in manifest.get("sampling_methods", ["full"])],
        "initial_radius": [float(value) for value in manifest.get("initial_radii", [])],
    }
    if any(not values for values in dimensions.values()):
        raise ValueError(f"Manifest is missing one or more coverage dimensions: {dimensions}")
    expected = math.prod(len(values) for values in dimensions.values())
    if len(df) != expected:
        raise ValueError(f"Incomplete radius-sweep coverage: expected {expected} rows, found {len(df)}.")
    for column, expected_values in dimensions.items():
        observed = df[column].drop_duplicates().tolist()
        if column == "initial_radius":
            expected_set = {round(float(value), 12) for value in expected_values}
            observed_set = {round(float(value), 12) for value in observed}
        else:
            expected_set = set(expected_values)
            observed_set = set(observed)
        if observed_set != expected_set:
            raise ValueError(
                f"Coverage mismatch for {column}: expected={sorted(expected_set)}, observed={sorted(observed_set)}"
            )


def _validate_manifest_configuration(df: pd.DataFrame, manifest: Dict[str, object]) -> None:
    fixed_params = manifest.get("fixed_params")
    if not isinstance(fixed_params, dict) or not fixed_params:
        return
    required_constants = {
        "processing_type": "batch",
        "sampling_method": "full",
        "batch_mode": "full_batch",
        "evaluation_split": "both",
    }
    for column, expected in required_constants.items():
        if column not in df.columns:
            raise ValueError(f"Runs CSV is missing manifest invariant column: {column}")
        observed = set(df[column].astype(str).str.strip().str.lower())
        if observed != {expected}:
            raise ValueError(f"Manifest configuration mismatch in {column}: {sorted(observed)}")
    if "param_initial_radius" not in df.columns:
        raise ValueError("Runs CSV is missing param_initial_radius required for configuration validation.")
    if not np.allclose(
        df["initial_radius"].to_numpy(dtype=float),
        pd.to_numeric(df["param_initial_radius"], errors="raise").to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("initial_radius and param_initial_radius disagree.")
    for param_name, expected in fixed_params.items():
        column = f"param_{param_name}"
        if column not in df.columns:
            raise ValueError(f"Runs CSV is missing fixed parameter column declared by manifest: {column}")
        series = df[column]
        if isinstance(expected, bool):
            normalized = series.astype(str).str.strip().str.lower().map(
                {"true": True, "1": True, "false": False, "0": False}
            )
            valid = not normalized.isna().any() and set(normalized) == {expected}
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
            valid = bool(np.allclose(values, float(expected), rtol=0.0, atol=1e-12))
        else:
            valid = set(series.astype(str).str.strip().str.lower()) == {str(expected).strip().lower()}
        if not valid:
            raise ValueError(f"Runs CSV does not match manifest fixed parameter: {param_name}")


def _radius_mask(series: pd.Series, radius: float) -> np.ndarray:
    return np.isclose(series.to_numpy(dtype=float), float(radius), rtol=0.0, atol=1e-12)


def _paired_percent_change(candidate: np.ndarray, anchor: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    denominator = np.abs(anchor)
    if np.any(denominator <= 0):
        raise ValueError("Cannot calculate paired percentage change when an anchor metric is zero.")
    if higher_is_better:
        return (candidate - anchor) / denominator * 100.0
    return (anchor - candidate) / denominator * 100.0


def build_pair_table(
    df: pd.DataFrame,
    *,
    anchor_radius: float,
    radii: Sequence[float],
    topologies: Sequence[str],
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for metric, metric_label, metric_slug, higher_is_better in METRIC_SPECS:
        for topology in topologies:
            topology_df = df[df["architecture"] == topology].copy()
            anchor_df = topology_df[_radius_mask(topology_df["initial_radius"], anchor_radius)].copy()
            anchor_df = anchor_df[list(PAIR_KEY_COLUMNS) + [metric]].rename(columns={metric: "anchor_value"})
            if anchor_df.empty:
                raise ValueError(f"No anchor rows for topology={topology} radius={anchor_radius}")

            for radius in radii:
                candidate_df = topology_df[_radius_mask(topology_df["initial_radius"], radius)].copy()
                candidate_df = candidate_df[list(PAIR_KEY_COLUMNS) + [metric]].rename(
                    columns={metric: "candidate_value"}
                )
                pairs = anchor_df.merge(candidate_df, on=list(PAIR_KEY_COLUMNS), how="inner", validate="one_to_one")
                if len(pairs) != len(anchor_df) or len(pairs) != len(candidate_df):
                    raise ValueError(
                        f"Incomplete pairing for metric={metric} topology={topology} radius={radius}: "
                        f"anchor={len(anchor_df)} candidate={len(candidate_df)} paired={len(pairs)}"
                    )
                pairs["metric"] = metric
                pairs["metric_label"] = metric_label
                pairs["metric_slug"] = metric_slug
                pairs["higher_is_better"] = higher_is_better
                pairs["initial_radius"] = float(radius)
                pairs["anchor_initial_radius"] = float(anchor_radius)
                pairs["pct_change_favoring_candidate"] = _paired_percent_change(
                    pairs["candidate_value"].to_numpy(dtype=float),
                    pairs["anchor_value"].to_numpy(dtype=float),
                    higher_is_better=higher_is_better,
                )
                rows.append(pairs)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _mean_ci(values: np.ndarray) -> Tuple[float, float, float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(finite))
    median = float(np.median(finite))
    if finite.size == 1:
        return mean, median, mean, mean, np.nan
    std = float(np.std(finite, ddof=1))
    if std == 0:
        p_value = 1.0 if math.isclose(mean, 0.0, rel_tol=0.0, abs_tol=1e-15) else 0.0
        return mean, median, mean, mean, p_value
    margin = float(stats.t.ppf(0.975, finite.size - 1) * std / np.sqrt(finite.size))
    _, p_value = stats.ttest_1samp(finite, popmean=0.0, nan_policy="omit")
    return mean, median, mean - margin, mean + margin, float(p_value)


def build_summaries(pair_df: pd.DataFrame, *, anchor_radius: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_records: List[Dict[str, object]] = []
    for keys, subset in pair_df.groupby(["metric", "metric_label", "metric_slug", "architecture", "initial_radius", "dataset"]):
        metric, label, slug, topology, radius, dataset = keys
        effects = subset["pct_change_favoring_candidate"].to_numpy(dtype=float)
        dataset_records.append(
            {
                "metric": metric,
                "metric_label": label,
                "metric_slug": slug,
                "architecture": topology,
                "initial_radius": float(radius),
                "dataset": dataset,
                "n_pairs": int(len(effects)),
                "mean_pct_change": float(np.mean(effects)),
                "median_pct_change": float(np.median(effects)),
            }
        )
    dataset_summary = pd.DataFrame(dataset_records)

    pooled_records: List[Dict[str, object]] = []
    group_columns = ["metric", "metric_label", "metric_slug", "architecture", "initial_radius"]
    for keys, subset in pair_df.groupby(group_columns):
        metric, label, slug, topology, radius = keys
        effects = subset["pct_change_favoring_candidate"].to_numpy(dtype=float)
        is_anchor = math.isclose(float(radius), float(anchor_radius), rel_tol=0.0, abs_tol=1e-12)
        if is_anchor:
            mean, median, ci_low, ci_high, p_value = 0.0, 0.0, 0.0, 0.0, np.nan
        else:
            mean, median, ci_low, ci_high, p_value = _mean_ci(effects)
        tolerance = 1e-12
        pooled_records.append(
            {
                "metric": metric,
                "metric_label": label,
                "metric_slug": slug,
                "architecture": topology,
                "initial_radius": float(radius),
                "anchor_initial_radius": float(anchor_radius),
                "n_pairs": int(len(effects)),
                "mean_anchor_value": float(subset["anchor_value"].mean()),
                "mean_candidate_value": float(subset["candidate_value"].mean()),
                "mean_pct_change": mean,
                "median_pct_change": median,
                "ci_low_pct": ci_low,
                "ci_high_pct": ci_high,
                "raw_p_value": p_value,
                "candidate_wins": int(np.sum(effects > tolerance)),
                "anchor_wins": int(np.sum(effects < -tolerance)),
                "ties": int(np.sum(np.abs(effects) <= tolerance)),
                "is_anchor": bool(is_anchor),
            }
        )
    pooled_summary = pd.DataFrame(pooled_records)
    pooled_summary["bh_q_value"] = np.nan
    test_mask = (~pooled_summary["is_anchor"]) & pooled_summary["raw_p_value"].notna()
    pooled_summary.loc[test_mask, "bh_q_value"] = benjamini_hochberg(
        pooled_summary.loc[test_mask, "raw_p_value"].to_numpy(dtype=float)
    )
    return dataset_summary, pooled_summary


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("BH adjustment requires a finite one-dimensional p-value array in [0, 1].")
    count = len(values)
    if count == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted_ranked = ranked * count / np.arange(1, count + 1, dtype=float)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def _q_stars(q_value: float) -> str:
    if not np.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def render_response_plot(
    summary_df: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
    anchor_radius: float,
    output_path: Path,
    dpi: int,
) -> None:
    plot_df = summary_df[summary_df["metric"] == metric].copy()
    if plot_df.empty:
        raise ValueError(f"No pooled radius-response rows for metric: {metric}")

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "legend.title_fontsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(9.4, 7.8))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("white")
    all_y = plot_df[["ci_low_pct", "ci_high_pct", "mean_pct_change"]].to_numpy(dtype=float).ravel()
    finite_y = all_y[np.isfinite(all_y)]
    y_span = max(1.0, float(np.max(finite_y) - np.min(finite_y))) if finite_y.size else 1.0

    present_topologies = [topology for topology in TOPOLOGY_ORDER if topology in set(plot_df["architecture"])]
    for topology_index, topology in enumerate(present_topologies):
        topology_df = plot_df[plot_df["architecture"] == topology].sort_values("initial_radius")
        style = TOPOLOGY_STYLES[topology]
        x = topology_df["initial_radius"].to_numpy(dtype=float)
        y = topology_df["mean_pct_change"].to_numpy(dtype=float)
        lower = y - topology_df["ci_low_pct"].to_numpy(dtype=float)
        upper = topology_df["ci_high_pct"].to_numpy(dtype=float) - y
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([lower, upper]),
            color=style["color"],
            marker=style["marker"],
            markersize=8.5,
            linewidth=2.5,
            capsize=4.0,
            elinewidth=1.7,
            label=style["label"],
            zorder=2 + topology_index,
        )
        for _, row in topology_df.iterrows():
            stars = _q_stars(float(row["bh_q_value"]))
            if not stars:
                continue
            offset = 7.0 + topology_index * 5.0
            ax.annotate(
                stars,
                (float(row["initial_radius"]), float(row["mean_pct_change"])),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color=style["color"],
                clip_on=False,
            )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8, zorder=0)
    ax.axvline(float(anchor_radius), color="#555555", linestyle=":", linewidth=1.1, alpha=0.75, zorder=0)
    radii = sorted(plot_df["initial_radius"].unique().tolist())
    ax.set_xticks(radii)
    tick_labels = [
        f"{radius:g}" if not math.isclose(radius, anchor_radius, abs_tol=1e-12) else f"{radius:.3f}"
        for radius in radii
    ]
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.set_xlabel("Initial radius")
    ax.set_ylabel(f"{metric_label} change vs r={anchor_radius:.3f} (%)\n(positive is better)")
    ax.grid(axis="both", color="black", linewidth=0.7, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.1)
    ax.legend(title="Topology", loc="best", frameon=True)
    ax.margins(x=0.035, y=max(0.10, 8.0 / (100.0 + y_span)))
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.12, top=0.985)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.16, transparent=True)
    plt.close(fig)


def analyze(
    *,
    manifest_path: str | Path,
    output_dir: Optional[str | Path] = None,
    dpi: int = 300,
    strict_coverage: bool = True,
) -> Dict[str, object]:
    manifest, runs_path = _load_manifest(manifest_path)
    runs_df = _canonicalize_runs(pd.read_csv(runs_path))
    if strict_coverage:
        _validate_coverage(runs_df, manifest)
        _validate_manifest_configuration(runs_df, manifest)

    anchor_radius = float(manifest["anchor_initial_radius"])
    radii = [float(value) for value in manifest["initial_radii"]]
    topologies = [str(value).strip().lower() for value in manifest["topologies"]]
    pair_df = build_pair_table(
        runs_df,
        anchor_radius=anchor_radius,
        radii=radii,
        topologies=topologies,
    )
    dataset_summary, pooled_summary = build_summaries(pair_df, anchor_radius=anchor_radius)

    expected_tests = (len(radii) - 1) * len(topologies) * len(METRIC_SPECS)
    observed_tests = int(pooled_summary["bh_q_value"].notna().sum())
    if strict_coverage and observed_tests != expected_tests:
        raise ValueError(f"Expected {expected_tests} BH-adjusted tests, found {observed_tests}.")

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path(manifest_path).expanduser().resolve().parent / "radius_response_analysis"
    )
    figures_dir = destination / "figures"
    tables_dir = destination / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    pair_path = tables_dir / "radius_response_pairs.csv"
    dataset_path = tables_dir / "radius_response_by_dataset.csv"
    pooled_path = tables_dir / "radius_response_pooled_summary.csv"
    pair_df.to_csv(pair_path, index=False)
    dataset_summary.to_csv(dataset_path, index=False)
    pooled_summary.to_csv(pooled_path, index=False)

    figures: Dict[str, str] = {}
    for metric, label, slug, _ in METRIC_SPECS:
        figure_path = figures_dir / f"radius_response_{slug}.svg"
        render_response_plot(
            pooled_summary,
            metric=metric,
            metric_label=label,
            anchor_radius=anchor_radius,
            output_path=figure_path,
            dpi=int(dpi),
        )
        figures[slug] = str(figure_path)

    metadata: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_manifest": str(Path(manifest_path).expanduser().resolve()),
        "source_runs_csv": str(runs_path),
        "output_dir": str(destination),
        "anchor_initial_radius": anchor_radius,
        "initial_radii": radii,
        "topologies": topologies,
        "n_runs": int(len(runs_df)),
        "n_pairs": int(len(pair_df)),
        "bh_family_size": observed_tests,
        "bh_family_definition": "all non-anchor radius x topology x balanced-metric pooled tests",
        "pair_table": str(pair_path),
        "dataset_summary": str(dataset_path),
        "pooled_summary": str(pooled_path),
        "figures": figures,
    }
    metadata_path = destination / "RADIUS_RESPONSE_ANALYSIS_METADATA.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_json"] = str(metadata_path)
    print(f"Saved radius-response analysis to {destination}")
    print(f"BH family size: {observed_tests}")
    return metadata


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--strict-coverage", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    analyze(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        dpi=args.dpi,
        strict_coverage=bool(args.strict_coverage),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
