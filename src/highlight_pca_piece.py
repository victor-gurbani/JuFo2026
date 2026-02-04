#!/usr/bin/env python3
"""Highlight one or more MusicXML pieces within the PCA composer clouds."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # type: ignore[import]

from embedding_cache import (  # type: ignore[attr-defined]
    DEFAULT_CACHE_CSV,
    cache_is_compatible,
    load_cache,
    load_meta,
    lookup_cached_embedding,
    project_features_into_cached_pca,
)
from feature_embedding import (  # type: ignore[attr-defined]
    CLOUD_GRID_SIZE,
    CLOUD_ISO_FRACTION,
    DEFAULT_HARMONIC,
    DEFAULT_MELODIC,
    DEFAULT_RHYTHMIC,
    _compute_projection,
    _gaussian_pdf,
    _load_features,
    _merge_feature_tables,
    _prepare_feature_matrix,
)
from harmonic_features import compute_harmonic_features  # type: ignore[attr-defined]
from melodic_features import compute_melodic_features  # type: ignore[attr-defined]
from rhythmic_features import compute_rhythmic_features  # type: ignore[attr-defined]
from score_parser import parse_score  # type: ignore[attr-defined]

DEFAULT_OUTPUT_DIR = Path("tmp/pca_highlights")
HIGHLIGHT_MARKER_COLORS: tuple[str, ...] = (
    "#0d1b2a",
    "#1b263b",
    "#191919",
    "#1d3557",
    "#2c3e50",
    "#3a0ca3",
    "#4a148c",
    "#311b92",
    "#1a237e",
    "#0b3d91",
    "#004d40",
    "#1b5e20",
    "#3e2723",
    "#4e342e",
    "#5d4037",
    "#3c1f0b",
    "#370617",
    "#5f0f40",
    "#342ead",
    "#2d132c",
    "#1f2421",
    "#143601",
    "#2b2d42",
    "#2f3e46",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a composer-cloud PCA HTML and highlight one or more MusicXML pieces."
        )
    )
    parser.add_argument(
        "musicxml",
        nargs="+",
        type=Path,
        help=("One or more MusicXML (.mxl or .musicxml) files to highlight."),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Optional explicit output HTML path. Overrides the temporary directory location."),
    )
    parser.add_argument(
        "--harmonic",
        type=Path,
        default=DEFAULT_HARMONIC,
        help=(
            "Harmonic features CSV supplying the baseline dataset for PCA."),
    )
    parser.add_argument(
        "--melodic",
        type=Path,
        default=DEFAULT_MELODIC,
        help=(
            "Melodic features CSV supplying the baseline dataset for PCA."),
    )
    parser.add_argument(
        "--rhythmic",
        type=Path,
        default=DEFAULT_RHYTHMIC,
        help=(
            "Rhythmic features CSV supplying the baseline dataset for PCA."),
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=DEFAULT_CACHE_CSV,
        help=(
            "Optional PCA embedding cache CSV (created by src/embedding_cache.py). "
            "When present and compatible with the supplied feature CSVs, cached coordinates are reused."
        ),
    )
    parser.add_argument(
        "--lookup-cache",
        type=Path,
        default=None,
        help=(
            "Optional secondary cache CSV used only to look up the highlighted piece coordinates (dim1..dim3). "
            "Useful when you have a very large projected cache (e.g., full PDMX) but still want to draw clouds "
            "from the smaller canonical cache."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable using the embedding cache even if present.",
    )
    parser.add_argument(
        "--composer",
        type=str,
        action="append",
        default=None,
        help=(
            "Composer labels for highlighted pieces. Provide multiple times for multiple inputs; "
            "defaults to 'External' when fewer values than pieces are supplied."),
    )
    parser.add_argument(
        "--title",
        type=str,
        action="append",
        default=None,
        help=(
            "Custom titles for highlighted pieces. Provide multiple times matching the inputs; "
            "defaults to each file's stem when omitted."),
    )
    return parser.parse_args()


def _normalize_path(value: Path | str) -> str:
    text = str(value).strip()
    return str(Path(text).expanduser().resolve())


def _normalize_path_or_none(value: object) -> str | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text:
        return None
    try:
        return str(Path(text).expanduser().resolve())
    except Exception:
        return None


def _lookup_embedding_in_csv(
    cache_csv: Path,
    mxl_abs_path: str,
    *,
    chunksize: int = 50_000,
) -> dict[str, object] | None:
    """Lookup one embedding row in a potentially huge cache CSV without loading it fully."""

    if not cache_csv.exists():
        return None

    target = str(mxl_abs_path)
    usecols = {"mxl_abs_path", "mxl_path", "dim1", "dim2", "dim3", "composer_label", "title"}

    try:
        reader = pd.read_csv(
            cache_csv,
            usecols=lambda c: c in usecols,
            dtype=str,
            chunksize=int(chunksize),
        )
        for chunk in reader:
            if chunk.empty:
                continue
            candidate_col = "mxl_abs_path" if "mxl_abs_path" in chunk.columns else ("mxl_path" if "mxl_path" in chunk.columns else None)
            if candidate_col is None:
                return None
            matches = chunk.index[chunk[candidate_col].astype(str) == target]
            if len(matches) == 0:
                continue
            row = chunk.loc[matches[0]].to_dict()
            for dim in ("dim1", "dim2", "dim3"):
                try:
                    row[dim] = float(str(row.get(dim, "nan")))
                except Exception:
                    row[dim] = float("nan")
            if not (
                np.isfinite(row["dim1"]) and np.isfinite(row["dim2"]) and np.isfinite(row["dim3"])
            ):
                return None
            return row
    except Exception:
        return None

    return None


def _drop_existing_piece(df: pd.DataFrame, normalized_path: str) -> pd.DataFrame:
    candidate_cols = [col for col in ("mxl_path", "mxl_abs_path") if col in df.columns]
    if not candidate_cols or df.empty:
        return df
    mask = pd.Series(False, index=df.index)
    for col in candidate_cols:
        normalized_col = df[col].apply(_normalize_path_or_none)
        mask = mask | (normalized_col == normalized_path)
    return df.loc[~mask].copy()


def _sanitize_metric_value(value: object) -> object:
    if value is None:
        return np.nan
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (str, bytes)):
        text = str(value).strip()
        if not text:
            return np.nan
        try:
            return float(text)
        except ValueError:
            return text
    return value


def _base_row(df: pd.DataFrame, normalized_path: str, composer: str, title: str) -> dict[str, object]:
    base: dict[str, object] = {}
    if "composer_label" in df.columns:
        base["composer_label"] = composer
    if "title" in df.columns:
        base["title"] = title
    if "mxl_path" in df.columns:
        base["mxl_path"] = normalized_path
    if "mxl_abs_path" in df.columns:
        base["mxl_abs_path"] = normalized_path
    return base


def _append_metrics(df: pd.DataFrame, base: dict[str, object], metrics: dict[str, object]) -> pd.DataFrame:
    row = {**base}
    for key, value in metrics.items():
        row[key] = _sanitize_metric_value(value)
    row_df = pd.DataFrame([row])
    combined = pd.concat([df, row_df], ignore_index=True, sort=False)
    return combined


def _compute_piece_metrics(path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    score = parse_score(path)
    harmonic_metrics = compute_harmonic_features(score)
    melodic_metrics = compute_melodic_features(score)
    rhythmic_metrics = compute_rhythmic_features(score)
    return harmonic_metrics, melodic_metrics, rhythmic_metrics


def build_cloud_figure(df: pd.DataFrame, highlight_df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    highlight_df = highlight_df.copy()

    if "composer_label" not in df.columns:
        df["composer_label"] = "Unknown"
    if "title" not in df.columns:
        df["title"] = "Untitled"

    if "composer_label" not in highlight_df.columns:
        highlight_df["composer_label"] = "External"
    if "title" not in highlight_df.columns:
        highlight_df["title"] = "Untitled"

    # Pandas may parse missing values as NaN floats, which breaks sorting/masking.
    df["composer_label"] = df["composer_label"].fillna("Unknown").astype(str)
    df["title"] = df["title"].fillna("Untitled").astype(str)
    highlight_df["composer_label"] = highlight_df["composer_label"].fillna("External").astype(str)
    highlight_df["title"] = highlight_df["title"].fillna("Untitled").astype(str)

    coords = df[["dim1", "dim2", "dim3"]].to_numpy()
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    padding = (maxs - mins) * 0.1
    lower = mins - padding
    upper = maxs + padding

    grid_axes = [
        np.linspace(lower[idx], upper[idx], CLOUD_GRID_SIZE)
        for idx in range(3)
    ]
    X, Y, Z = np.meshgrid(*grid_axes, indexing="ij")
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    palette = px.colors.qualitative.Plotly

    fig = go.Figure()

    highlight_indices = set(highlight_df.index)
    highlight_mask = df.index.to_series().isin(highlight_indices)

    composers = sorted(df["composer_label"].unique())

    for idx, composer in enumerate(composers):
        composer_mask = df["composer_label"] == composer
        composer_coords = coords[composer_mask.to_numpy()]
        if composer_coords.shape[0] < 4:
            continue
        cov = np.cov(composer_coords, rowvar=False)
        if cov.shape != (3, 3):
            continue
        cov = cov + np.eye(3) * 1e-6
        mean = composer_coords.mean(axis=0)
        pdf = _gaussian_pdf(grid_points, mean, cov)
        pdf = pdf.reshape(X.shape)
        vmax = float(np.nanmax(pdf))
        if not np.isfinite(vmax) or vmax <= 0:
            continue
        iso = float(vmax * CLOUD_ISO_FRACTION)
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Isosurface(
                x=X.ravel(),
                y=Y.ravel(),
                z=Z.ravel(),
                value=pdf.ravel(),
                isomin=iso,
                isomax=vmax,
                surface_count=1,
                colorscale=[[0.0, color], [1.0, color]],
                opacity=0.35,
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=False,
                lighting=dict(
                    ambient=0.45,
                    diffuse=0.6,
                    specular=0.25,
                    roughness=0.4,
                    fresnel=0.1,
                ),
                lightposition=dict(x=80, y=100, z=60),
                name=str(composer),
                hovertemplate=f"Composer: {composer}<extra></extra>",
            )
        )

    for idx, composer in enumerate(composers):
        composer_mask = df["composer_label"] == composer
        scatter_mask = composer_mask & (~highlight_mask)
        if not scatter_mask.any():
            continue
        color = palette[idx % len(palette)]
        composer_data = df.loc[scatter_mask, ["composer_label", "title"]]
        fig.add_trace(
            go.Scatter3d(
                x=df.loc[scatter_mask, "dim1"],
                y=df.loc[scatter_mask, "dim2"],
                z=df.loc[scatter_mask, "dim3"],
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.8),
                name=f"{composer} pieces",
                customdata=composer_data.to_numpy(),
                hovertemplate=(
                    "Composer: %{customdata[0]}<br>Title: %{customdata[1]}<br>"
                    "dim1=%{x:.2f}<br>dim2=%{y:.2f}<br>dim3=%{z:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    for idx, (_, row) in enumerate(highlight_df.iterrows()):
        color = HIGHLIGHT_MARKER_COLORS[idx % len(HIGHLIGHT_MARKER_COLORS)]
        label = f"{row['composer_label']} - {row['title']}"
        fig.add_trace(
            go.Scatter3d(
                x=[row["dim1"]],
                y=[row["dim2"]],
                z=[row["dim3"]],
                mode="markers",
                marker=dict(
                    size=12,
                    color=color,
                    symbol="diamond",
                    line=dict(width=3, color="#FFFFFF"),
                ),
                name=label,
                text=[label],
                hovertemplate=(
                    "<b>%{text}</b><br>dim1=%{x:.2f}<br>dim2=%{y:.2f}<br>dim3=%{z:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Composer Clouds (PCA) with Highlighted Pieces",
        scene=dict(xaxis_title="dim1", yaxis_title="dim2", zaxis_title="dim3"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=26, b=0),
    )
    return fig


def determine_output_path(output: Path | None, musicxml_paths: list[Path]) -> Path:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    first = musicxml_paths[0]
    slug = Path(first).stem.replace(" ", "_")
    tmp_dir = Path(tempfile.mkdtemp(dir=DEFAULT_OUTPUT_DIR, prefix="highlight_"))
    return tmp_dir / f"{slug}_pca_cloud.html"


def main() -> int:
    args = parse_arguments()
    input_paths = args.musicxml
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        missing_str = "\n".join(str(path) for path in missing)
        print(f"[error] The following MusicXML files were not found:\n{missing_str}")
        return 1

    composer_inputs = args.composer or []
    title_inputs = args.title or []

    composers: list[str] = []
    titles: list[str] = []
    normalized_paths: list[str] = []

    for idx, path in enumerate(input_paths):
        composer_value = composer_inputs[idx] if idx < len(composer_inputs) else "External"
        title_value = title_inputs[idx] if idx < len(title_inputs) else path.stem
        composers.append(composer_value)
        titles.append(title_value)
        normalized_paths.append(_normalize_path(path))

    # Fast path: use precomputed embedding cache if available and compatible.
    if not args.no_cache:
        meta_json = args.embedding_cache.with_suffix(args.embedding_cache.suffix + ".meta.json")
        compatible, reason = cache_is_compatible(
            args.embedding_cache,
            meta_json,
            args.harmonic,
            args.melodic,
            args.rhythmic,
            method="pca",
            seed=42,
            perplexity=30.0,
            early_exaggeration=12.0,
            learning_rate=200.0,
            metric="euclidean",
        )
        if compatible:
            try:
                cache_df = load_cache(args.embedding_cache).copy()
                meta = load_meta(args.embedding_cache)
            except Exception as exc:
                print(f"[warn] Failed to load embedding cache; falling back to recomputation: {exc}")
            else:
                combined = cache_df.copy()
                highlight_indices: list[int] = []
                appended_rows: list[dict[str, object]] = []

                for idx, normalized_path in enumerate(normalized_paths):
                    # First: optional secondary lookup cache (e.g. a huge full-PDMX projected cache).
                    if args.lookup_cache is not None:
                        found = _lookup_embedding_in_csv(args.lookup_cache, normalized_path)
                        if found is not None:
                            appended_rows.append(
                                {
                                    "composer_label": composers[idx],
                                    "title": titles[idx],
                                    "mxl_path": normalized_path,
                                    "mxl_abs_path": normalized_path,
                                    "dim1": float(str(found.get("dim1", "nan"))),
                                    "dim2": float(str(found.get("dim2", "nan"))),
                                    "dim3": float(str(found.get("dim3", "nan"))),
                                }
                            )
                            continue

                    cached = lookup_cached_embedding(combined, normalized_path)
                    if cached is not None:
                        continue

                    # Not present in cache: compute features and project into cached PCA space.
                    try:
                        harmonic_metrics, melodic_metrics, rhythmic_metrics = _compute_piece_metrics(input_paths[idx])
                        feature_mapping = {
                            **harmonic_metrics,
                            **melodic_metrics,
                            **rhythmic_metrics,
                        }
                        coords = project_features_into_cached_pca(feature_mapping, meta)
                    except Exception as exc:
                        print(
                            f"[warn] Could not project {input_paths[idx]} into cached PCA space ({exc}); "
                            "falling back to full recomputation."
                        )
                        break

                    appended_rows.append(
                        {
                            "composer_label": composers[idx],
                            "title": titles[idx],
                            "mxl_path": normalized_path,
                            "mxl_abs_path": normalized_path,
                            "dim1": float(coords[0]),
                            "dim2": float(coords[1]),
                            "dim3": float(coords[2]),
                        }
                    )
                else:
                    if appended_rows:
                        append_df = pd.DataFrame(appended_rows)
                        combined = pd.concat([combined, append_df], ignore_index=True, sort=False)

                    # Resolve highlight indices after potential appends.
                    normalized_series = combined["mxl_abs_path"].astype(str)
                    highlight_indices = []
                    for normalized_path in normalized_paths:
                        matches = combined.index[normalized_series == normalized_path]
                        if len(matches) == 0:
                            print("[error] Highlighted piece could not be located in embedding cache.")
                            return 1
                        highlight_indices.append(int(matches[0]))

                    highlight_mask_series = combined.index.to_series().isin(highlight_indices)
                    combined["is_highlight"] = highlight_mask_series

                    highlight_df = combined.loc[highlight_indices].copy()
                    # Apply user-provided labels for display only.
                    for idx, row_index in enumerate(highlight_df.index.tolist()):
                        highlight_df.loc[row_index, "composer_label"] = composers[idx]
                        highlight_df.loc[row_index, "title"] = titles[idx]

                    figure = build_cloud_figure(combined, highlight_df)
                    output_path = determine_output_path(args.output, input_paths)
                    figure.write_html(str(output_path), include_plotlyjs="cdn")
                    figure.write_json(str(output_path.with_suffix(".json")))
                    print(f"Created PCA composer cloud with highlight at {output_path} (used embedding cache)")
                    return 0
        else:
            # Only informational; falling back keeps behavior identical.
            if args.embedding_cache.exists():
                print(f"[info] Embedding cache not compatible; recomputing PCA ({reason}).")

    metrics_per_piece: list[tuple[dict[str, object], dict[str, object], dict[str, object]]] = []
    for idx, path in enumerate(input_paths):
        try:
            metrics_per_piece.append(_compute_piece_metrics(path))
        except Exception as exc:  # pragma: no cover - defensive catch for unforeseen parsing errors
            print(f"[error] Failed to compute features for {path}: {exc}")
            return 1

    try:
        harmonic_df = _load_features(args.harmonic)
        melodic_df = _load_features(args.melodic)
        rhythmic_df = _load_features(args.rhythmic)
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1

    for idx, normalized_path in enumerate(normalized_paths):
        harmonic_df = _drop_existing_piece(harmonic_df, normalized_path)
        melodic_df = _drop_existing_piece(melodic_df, normalized_path)
        rhythmic_df = _drop_existing_piece(rhythmic_df, normalized_path)

        harmonic_base = _base_row(harmonic_df, normalized_path, composers[idx], titles[idx])
        melodic_base = _base_row(melodic_df, normalized_path, composers[idx], titles[idx])
        rhythmic_base = _base_row(rhythmic_df, normalized_path, composers[idx], titles[idx])

        harmonic_metrics, melodic_metrics, rhythmic_metrics = metrics_per_piece[idx]
        harmonic_df = _append_metrics(harmonic_df, harmonic_base, harmonic_metrics)
        melodic_df = _append_metrics(melodic_df, melodic_base, melodic_metrics)
        rhythmic_df = _append_metrics(rhythmic_df, rhythmic_base, rhythmic_metrics)

    combined = _merge_feature_tables(harmonic_df, melodic_df, rhythmic_df)
    matrix, _, _ = _prepare_feature_matrix(combined)
    coords, _ = _compute_projection(
        matrix,
        method="pca",
        seed=42,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate=200.0,
        metric="euclidean",
    )

    combined = combined.copy()
    combined["dim1"] = coords[:, 0]
    combined["dim2"] = coords[:, 1]
    combined["dim3"] = coords[:, 2]

    normalized_series = combined["mxl_path"].apply(_normalize_path_or_none)
    highlight_indices: list[int] = []
    for path in normalized_paths:
        matches = combined.index[normalized_series == path]
        if len(matches) == 0:
            print("[error] Highlighted piece could not be located after feature merge.")
            return 1
        highlight_indices.append(int(matches[0]))

    highlight_mask_series = combined.index.to_series().isin(highlight_indices)
    combined["is_highlight"] = highlight_mask_series

    highlight_frames = [combined.loc[[idx]] for idx in highlight_indices]
    highlight_df = pd.concat(highlight_frames, ignore_index=False).copy()

    figure = build_cloud_figure(combined, highlight_df)
    output_path = determine_output_path(args.output, input_paths)
    figure.write_html(str(output_path), include_plotlyjs="cdn")
    figure.write_json(str(output_path.with_suffix(".json")))
    print(f"Created PCA composer cloud with highlight at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
