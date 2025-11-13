#!/usr/bin/env python3
"""Highlight a single MusicXML piece within the PCA composer clouds."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # type: ignore[import]

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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a composer-cloud PCA HTML and highlight a given MusicXML piece."
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
        "--composer",
        type=str,
        default="External",
        help="Composer label to use for the highlighted piece (default: External).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional override title for the highlighted piece (defaults to file stem).",
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


def build_cloud_figure(df: pd.DataFrame, highlight_row: pd.Series) -> go.Figure:
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

    highlight_mask = df.get("is_highlight", pd.Series(False, index=df.index))
    highlight_mask = highlight_mask.astype(bool)

    for idx, composer in enumerate(sorted(df["composer_label"].unique())):
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

    for idx, composer in enumerate(sorted(df["composer_label"].unique())):
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

    fig.add_trace(
        go.Scatter3d(
            x=[highlight_row["dim1"]],
            y=[highlight_row["dim2"]],
            z=[highlight_row["dim3"]],
            mode="markers",
            marker=dict(
                size=11,
                color="#111111",
                symbol="diamond",
                line=dict(width=3, color="#FFFFFF"),
            ),
            name="Highlighted piece",
            text=[f"{highlight_row['composer_label']} - {highlight_row['title']}"],
            hovertemplate=(
                "<b>%{text}</b><br>dim1=%{x:.2f}<br>dim2=%{y:.2f}<br>dim3=%{z:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Composer Clouds (PCA) with Highlighted Piece",
        scene=dict(xaxis_title="dim1", yaxis_title="dim2", zaxis_title="dim3"),
        legend=dict(itemsizing="constant"),
    )
    return fig


def determine_output_path(output: Path | None, musicxml: Path) -> Path:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = Path(musicxml).stem.replace(" ", "_")
    tmp_dir = Path(tempfile.mkdtemp(dir=DEFAULT_OUTPUT_DIR, prefix="highlight_"))
    return tmp_dir / f"{slug}_pca_cloud.html"


def main() -> int:
    args = parse_arguments()
    if not args.musicxml.exists():
        print(f"MusicXML file not found: {args.musicxml}")
        return 1

    normalized_path = _normalize_path(args.musicxml)
    title = args.title if args.title else args.musicxml.stem

    try:
        harmonic_metrics, melodic_metrics, rhythmic_metrics = _compute_piece_metrics(args.musicxml)
    except Exception as exc:  # pragma: no cover - defensive catch for unforeseen parsing errors
        print(f"[error] Failed to compute features for {args.musicxml}: {exc}")
        return 1

    try:
        harmonic_df = _load_features(args.harmonic)
        melodic_df = _load_features(args.melodic)
        rhythmic_df = _load_features(args.rhythmic)
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1

    harmonic_df = _drop_existing_piece(harmonic_df, normalized_path)
    melodic_df = _drop_existing_piece(melodic_df, normalized_path)
    rhythmic_df = _drop_existing_piece(rhythmic_df, normalized_path)

    harmonic_base = _base_row(harmonic_df, normalized_path, args.composer, title)
    melodic_base = _base_row(melodic_df, normalized_path, args.composer, title)
    rhythmic_base = _base_row(rhythmic_df, normalized_path, args.composer, title)

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
    highlight_mask = normalized_series == normalized_path
    if not highlight_mask.any():
        print("[error] Highlighted piece could not be located after feature merge.")
        return 1
    combined["is_highlight"] = highlight_mask
    highlight_row = combined.loc[highlight_mask].iloc[0]

    figure = build_cloud_figure(combined, highlight_row)
    output_path = determine_output_path(args.output, args.musicxml)
    figure.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Created PCA composer cloud with highlight at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
