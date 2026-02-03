#!/usr/bin/env python3
"""Generate composer clouds from a cached full-feature CSV.

This script is meant to make "instant" composer clouds possible once you have
cached feature vectors (e.g., full PDMX) in a single CSV.

It supports two scientifically interesting views:

1) Canonical axes ("old PCA axes")
   - Project the chosen subset into the PCA space fit on the canonical dataset
     (typically Bach/Mozart/Chopin/Debussy).
   - This enables direct visual comparison across arbitrary subsets.

2) Refit axes ("custom PCA axes")
   - Fit PCA on the chosen subset itself.
   - This can reveal local structure, but axes change with the subset.

Filtering can be done via:
- A JSON config file containing named groups (recommended), or
- CLI include/exclude regex patterns.

Outputs are Plotly HTML files (plus .json sidecars when possible).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA  # type: ignore[import]
from sklearn.preprocessing import StandardScaler  # type: ignore[import]

from embedding_cache import load_meta  # type: ignore[attr-defined]
from feature_embedding import (  # type: ignore[attr-defined]
    EXCLUDED_FEATURES,
    _axis_titles,
    _plot_composer_clouds,
    _write_plotly_figure,
)


DEFAULT_CANONICAL_MODEL_CACHE = Path("data/embeddings/pca_embedding_cache.csv")
DEFAULT_OUTDIR = Path("figures/embeddings/subsets")


@dataclass(frozen=True)
class FilterSpec:
    include_composer: Tuple[str, ...] = ()
    exclude_composer: Tuple[str, ...] = ()
    include_title: Tuple[str, ...] = ()
    exclude_title: Tuple[str, ...] = ()

    @staticmethod
    def from_group_config(group: Dict[str, Any]) -> "FilterSpec":
        return FilterSpec(
            include_composer=tuple(group.get("include_composer", []) or []),
            exclude_composer=tuple(group.get("exclude_composer", []) or []),
            include_title=tuple(group.get("include_title", []) or []),
            exclude_title=tuple(group.get("exclude_title", []) or []),
        )


def _compile_patterns(patterns: Sequence[str]) -> List[re.Pattern[str]]:
    compiled: List[re.Pattern[str]] = []
    for pat in patterns:
        text = str(pat).strip()
        if not text:
            continue
        compiled.append(re.compile(text, flags=re.IGNORECASE))
    return compiled


def _matches_any(patterns: List[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) is not None for p in patterns)


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _filter_rows(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    if df.empty:
        return df

    composer_text = df.get("composer_label", pd.Series([""] * len(df))).apply(_safe_text)
    title_text = df.get("title", pd.Series([""] * len(df))).apply(_safe_text)

    inc_comp = _compile_patterns(spec.include_composer)
    exc_comp = _compile_patterns(spec.exclude_composer)
    inc_title = _compile_patterns(spec.include_title)
    exc_title = _compile_patterns(spec.exclude_title)

    mask = pd.Series(True, index=df.index)

    if inc_comp:
        mask &= composer_text.apply(lambda t: _matches_any(inc_comp, t))
    if exc_comp:
        mask &= ~composer_text.apply(lambda t: _matches_any(exc_comp, t))

    if inc_title:
        mask &= title_text.apply(lambda t: _matches_any(inc_title, t))
    if exc_title:
        mask &= ~title_text.apply(lambda t: _matches_any(exc_title, t))

    return df.loc[mask].copy()


def _balance_per_composer(df: pd.DataFrame, max_per_composer: Optional[int]) -> pd.DataFrame:
    if max_per_composer is None or max_per_composer <= 0 or df.empty:
        return df
    if "composer_label" not in df.columns:
        return df

    # Deterministic sampling: sort by path/title, then take first N per composer.
    sort_cols = [c for c in ("composer_label", "mxl_abs_path", "mxl_path", "title") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    balanced = df.groupby("composer_label", group_keys=False).head(max_per_composer).reset_index(drop=True)
    return balanced


def _numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    # Drop any accidental ID-like columns.
    numeric_cols = [col for col in numeric_cols if col not in {"dim1", "dim2", "dim3"}]
    numeric_cols = [col for col in numeric_cols if col not in EXCLUDED_FEATURES]
    return numeric_cols


def _prepare_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    if not feature_cols:
        raise ValueError("No feature columns available to build a PCA matrix.")
    filled = df[list(feature_cols)].copy()
    filled = filled.apply(pd.to_numeric, errors="coerce")
    filled = filled.fillna(filled.mean())
    scaler = StandardScaler()
    return scaler.fit_transform(filled.values)


def _project_into_canonical_axes(df: pd.DataFrame, model_cache_csv: Path) -> np.ndarray:
    meta = load_meta(model_cache_csv)
    pca = meta.get("pca")
    if not isinstance(pca, dict):
        raise ValueError("Model cache meta is missing PCA artifacts (meta['pca']).")

    feature_columns = pca.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("Model cache meta PCA artifacts missing feature_columns.")

    impute_means = pca.get("impute_means")
    if not isinstance(impute_means, dict):
        raise ValueError("Model cache meta PCA artifacts missing impute_means.")

    scaler_mean = np.array(pca.get("scaler_mean", []), dtype=float)
    scaler_scale = np.array(pca.get("scaler_scale", []), dtype=float)
    pca_mean = np.array(pca.get("pca_mean", []), dtype=float)
    components = np.array(pca.get("pca_components", []), dtype=float)

    if scaler_mean.shape[0] != len(feature_columns) or scaler_scale.shape[0] != len(feature_columns):
        raise ValueError("Scaler parameters in model cache do not match feature vector length.")
    if pca_mean.shape[0] != len(feature_columns):
        raise ValueError("PCA mean in model cache does not match feature vector length.")
    if components.shape != (3, len(feature_columns)):
        raise ValueError("PCA components in model cache have an unexpected shape.")

    # Build an aligned feature matrix for this subset.
    aligned = pd.DataFrame(index=df.index)
    for col in feature_columns:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
        else:
            series = pd.Series(np.nan, index=df.index)
        mean_value = impute_means.get(col)
        try:
            mean_num = float(mean_value) if mean_value is not None else np.nan
        except Exception:
            mean_num = np.nan
        if not np.isfinite(mean_num):
            mean_num = float(series.mean()) if np.isfinite(series.mean()) else 0.0
        aligned[col] = series.fillna(mean_num)

    x = aligned.values.astype(float)
    x_scaled = (x - scaler_mean) / scaler_scale
    coords = (x_scaled - pca_mean) @ components.T
    return coords


def _refit_pca(df: pd.DataFrame, seed: int) -> Tuple[np.ndarray, PCA, List[str]]:
    feature_cols = _numeric_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found for PCA refit.")
    matrix = _prepare_matrix(df, feature_cols)
    model = PCA(n_components=3, random_state=seed)
    coords = model.fit_transform(matrix)
    return coords, model, feature_cols


def _write_subset_csv(df: pd.DataFrame, coords: np.ndarray, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out[["dim1", "dim2", "dim3"]] = coords
    out.to_csv(output_csv, index=False)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate composer clouds from a cached full-feature CSV, either on canonical PCA axes or on subset-refit axes."
        )
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        required=True,
        help="CSV containing identifiers + full feature columns (e.g., data/features/full_pdmx_feature_cache.csv).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config containing named composer/title filter groups.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Group name in --config to use for filtering.",
    )

    parser.add_argument(
        "--include-composer",
        action="append",
        default=None,
        help="Regex include filter applied to composer_label (can be repeated).",
    )
    parser.add_argument(
        "--exclude-composer",
        action="append",
        default=None,
        help="Regex exclude filter applied to composer_label (can be repeated).",
    )
    parser.add_argument(
        "--include-title",
        action="append",
        default=None,
        help="Regex include filter applied to title (can be repeated).",
    )
    parser.add_argument(
        "--exclude-title",
        action="append",
        default=None,
        help="Regex exclude filter applied to title (can be repeated).",
    )

    parser.add_argument(
        "--max-per-composer",
        type=int,
        default=None,
        help="Optional cap per composer_label (deterministic selection).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional global cap for quick smoke tests.",
    )

    parser.add_argument(
        "--axes",
        choices=["canonical", "refit", "both"],
        default="both",
        help="Which axes to use: canonical (project), refit (subset PCA), or both.",
    )
    parser.add_argument(
        "--canonical-model-cache",
        type=Path,
        default=DEFAULT_CANONICAL_MODEL_CACHE,
        help="Canonical PCA model cache CSV (used for canonical projection).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset PCA refit.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for HTML figures and optional CSV exports.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for output filenames (defaults to --group or 'subset').",
    )
    parser.add_argument(
        "--write-subset-csv",
        action="store_true",
        help="Also write filtered subset rows to CSV with dim1..dim3 columns.",
    )
    return parser.parse_args(argv)


def _load_group_spec(config_path: Path, group_name: str) -> Tuple[FilterSpec, Dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    groups = config.get("groups") if isinstance(config, dict) else None
    if not isinstance(groups, dict):
        raise ValueError("Config JSON must be an object with a 'groups' object.")
    group = groups.get(group_name)
    if not isinstance(group, dict):
        raise ValueError(f"Group not found or invalid: {group_name}")
    return FilterSpec.from_group_config(group), group


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.feature_cache.exists():
        raise FileNotFoundError(f"Feature cache not found: {args.feature_cache}")

    group_meta: Dict[str, Any] = {}
    spec = FilterSpec(
        include_composer=tuple(args.include_composer or []),
        exclude_composer=tuple(args.exclude_composer or []),
        include_title=tuple(args.include_title or []),
        exclude_title=tuple(args.exclude_title or []),
    )

    if args.config is not None and args.group is not None:
        group_spec, group_meta = _load_group_spec(args.config, args.group)
        # CLI patterns augment group patterns.
        spec = FilterSpec(
            include_composer=tuple(group_spec.include_composer) + tuple(spec.include_composer),
            exclude_composer=tuple(group_spec.exclude_composer) + tuple(spec.exclude_composer),
            include_title=tuple(group_spec.include_title) + tuple(spec.include_title),
            exclude_title=tuple(group_spec.exclude_title) + tuple(spec.exclude_title),
        )

    label = args.label or args.group or "subset"
    safe_label = re.sub(r"[^a-zA-Z0-9_\-]+", "_", label).strip("_") or "subset"

    df = pd.read_csv(args.feature_cache)
    filtered = _filter_rows(df, spec)
    if args.limit is not None and args.limit > 0:
        filtered = filtered.head(args.limit).copy()

    filtered = _balance_per_composer(filtered, args.max_per_composer)

    if filtered.empty:
        print("No rows matched the requested filters.")
        return 2

    if "composer_label" in filtered.columns:
        counts = filtered["composer_label"].value_counts().sort_index()
        print(f"Subset pieces: {len(filtered)}")
        print("Pieces per composer:")
        for composer, count in counts.items():
            print(f"  - {composer}: {int(count)}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    axis_titles = _axis_titles("pca")

    if args.axes in {"canonical", "both"}:
        coords = _project_into_canonical_axes(filtered, args.canonical_model_cache)
        clouds_path = args.outdir / f"{safe_label}__canonical_axes_clouds.html"
        _plot_composer_clouds(
            filtered,
            coords,
            clouds_path,
            axis_titles,
            f"Komponistenwolken (Canonical PCA Axes) — {label}",
            write_html_also=True,
        )
        print(f"Canonical-axes clouds written to {clouds_path}")
        if args.write_subset_csv:
            csv_path = args.outdir / f"{safe_label}__canonical_axes_subset.csv"
            _write_subset_csv(filtered, coords, csv_path)
            print(f"Canonical-axes subset CSV written to {csv_path}")

    if args.axes in {"refit", "both"}:
        coords_refit, model, feature_cols = _refit_pca(filtered, seed=args.seed)
        clouds_path = args.outdir / f"{safe_label}__refit_axes_clouds.html"
        _plot_composer_clouds(
            filtered,
            coords_refit,
            clouds_path,
            axis_titles,
            f"Komponistenwolken (Refit PCA Axes) — {label}",
            write_html_also=True,
        )
        print(f"Refit-axes clouds written to {clouds_path}")
        print(
            "Refit explained variance (PC1-3): "
            + ", ".join(f"{v * 100.0:.1f}%" for v in model.explained_variance_ratio_)
        )
        if args.write_subset_csv:
            csv_path = args.outdir / f"{safe_label}__refit_axes_subset.csv"
            _write_subset_csv(filtered, coords_refit, csv_path)
            print(f"Refit-axes subset CSV written to {csv_path}")

    # Optional: write a tiny run manifest for reproducibility.
    manifest = {
        "feature_cache": str(args.feature_cache),
        "config": str(args.config) if args.config else None,
        "group": args.group,
        "group_meta": group_meta,
        "filter": {
            "include_composer": list(spec.include_composer),
            "exclude_composer": list(spec.exclude_composer),
            "include_title": list(spec.include_title),
            "exclude_title": list(spec.exclude_title),
        },
        "max_per_composer": args.max_per_composer,
        "axes": args.axes,
        "canonical_model_cache": str(args.canonical_model_cache),
        "seed": args.seed,
        "subset_piece_count": int(len(filtered)),
    }
    manifest_path = args.outdir / f"{safe_label}__run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
