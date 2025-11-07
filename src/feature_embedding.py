"""Generate interactive embeddings of feature tables using dimensionality reduction."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # type: ignore[import]
from sklearn.decomposition import PCA  # type: ignore[import]
from sklearn.manifold import TSNE  # type: ignore[import]
from sklearn.preprocessing import StandardScaler  # type: ignore[import]

DEFAULT_HARMONIC = Path("data/features/harmonic_features.csv")
DEFAULT_MELODIC = Path("data/features/melodic_features.csv")
DEFAULT_RHYTHMIC = Path("data/features/rhythmic_features.csv")
DEFAULT_OUTDIR = Path("figures/embeddings")
EXCLUDED_FEATURES = {
    "note_count",
    "note_event_count",
    "chord_event_count",
    "chord_quality_total",
    "roman_chord_count",
    "dissonant_note_count",
}
CLOUD_GRID_SIZE = 22
CLOUD_ISO_FRACTION = 0.22


def _load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    return pd.read_csv(path)


def _merge_feature_tables(harm: pd.DataFrame, mel: pd.DataFrame, rhy: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["composer_label", "title", "mxl_path"]
    for df in (harm, mel, rhy):
        if not set(key_cols).issubset(df.columns):
            raise ValueError("Missing identifying columns in feature tables.")

    combined = (
        harm.set_index(key_cols)
        .join(mel.set_index(key_cols), how="inner", rsuffix="_mel")
        .join(rhy.set_index(key_cols), how="inner", rsuffix="_rhy")
        .reset_index()
    )
    return combined


def _prepare_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, List[str], StandardScaler]:
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    numeric_cols = [col for col in numeric_cols if col not in {"mxl_path"}]
    numeric_cols = [col for col in numeric_cols if col not in EXCLUDED_FEATURES]
    if not numeric_cols:
        raise ValueError("No numeric feature columns available for projection.")
    filled = df[numeric_cols].copy()
    filled = filled.fillna(filled.mean())
    scaler = StandardScaler()
    matrix = scaler.fit_transform(filled.values)
    return matrix, numeric_cols, scaler


def _compute_projection(
    matrix: np.ndarray,
    method: str,
    seed: int,
    perplexity: float,
    early_exaggeration: float,
    learning_rate: float,
    metric: str,
) -> tuple[np.ndarray, Optional[PCA]]:
    if method == "pca":
        model = PCA(n_components=3, random_state=seed)
        coords = model.fit_transform(matrix)
        return coords, model
    if method == "tsne":
        model = TSNE(
            n_components=3,
            init="pca",
            random_state=seed,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            metric=metric,
        )
        coords = model.fit_transform(matrix)
        return coords, None
    raise ValueError(f"Unsupported projection method: {method}")


def _plot_embedding(df: pd.DataFrame, coords: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plot_df[["dim1", "dim2", "dim3"]] = coords

    fig = px.scatter_3d(
        plot_df,
        x="dim1",
        y="dim2",
        z="dim3",
        color="composer_label",
        hover_data={
            "title": True,
            "composer_label": True,
            "mxl_path": True,
            "dim1": False,
            "dim2": False,
            "dim3": False,
        },
        title="Feature Embedding (3D)",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def _plot_embedding_2d(df: pd.DataFrame, coords: np.ndarray, output_path: Path) -> None:
    if coords.shape[1] < 2:
        raise ValueError("Projection returned fewer than two dimensions; cannot create 2D scatter.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plot_df[["dim1", "dim2"]] = coords[:, :2]

    fig = px.scatter(
        plot_df,
        x="dim1",
        y="dim2",
        color="composer_label",
        hover_data={
            "title": True,
            "composer_label": True,
            "mxl_path": True,
            "dim1": False,
            "dim2": False,
        },
        title="Feature Embedding (2D)",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(xaxis_title="dim1", yaxis_title="dim2")
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def _gaussian_pdf(points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    diff = points - mean
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
    det_cov = float(np.linalg.det(cov))
    if not np.isfinite(det_cov) or det_cov <= 0:
        det_cov = 1e-6
    exponent = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    dim = cov.shape[0]
    norm_const = np.sqrt(((2 * np.pi) ** dim) * det_cov)
    norm_const = norm_const if norm_const > 1e-12 else 1e-12
    return np.exp(-0.5 * exponent) / norm_const


def _plot_composer_clouds(df: pd.DataFrame, coords: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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

    for idx, composer in enumerate(df["composer_label"].unique()):
        mask = df["composer_label"] == composer
        comp_coords = coords[mask]
        if comp_coords.shape[0] < 4:
            continue
        cov = np.cov(comp_coords, rowvar=False)
        if cov.shape != (3, 3):
            continue
        cov = cov + np.eye(3) * 1e-6
        mean = comp_coords.mean(axis=0)
        pdf = _gaussian_pdf(grid_points, mean, cov)
        pdf = pdf.reshape(X.shape)
        vmax = float(np.max(pdf))
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
                lighting=dict(ambient=0.45, diffuse=0.6, specular=0.25, roughness=0.4, fresnel=0.1),
                lightposition=dict(x=80, y=100, z=60),
                name=str(composer),
                hovertemplate=f"Composer: {composer}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Composer Clouds (3D)",
        scene=dict(xaxis_title="dim1", yaxis_title="dim2", zaxis_title="dim3"),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def _plot_composer_clouds_2d(df: pd.DataFrame, coords: np.ndarray, output_path: Path) -> None:
    if coords.shape[1] < 2:
        raise ValueError("Projection returned fewer than two dimensions; cannot create 2D clouds.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coords_2d = coords[:, :2]
    mins = coords_2d.min(axis=0)
    maxs = coords_2d.max(axis=0)
    padding = (maxs - mins) * 0.1
    lower = mins - padding
    upper = maxs + padding

    grid_axes = [
        np.linspace(lower[idx], upper[idx], CLOUD_GRID_SIZE)
        for idx in range(2)
    ]
    X, Y = np.meshgrid(*grid_axes, indexing="xy")
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    palette = px.colors.qualitative.Plotly
    fig = go.Figure()

    for idx, composer in enumerate(df["composer_label"].unique()):
        mask = df["composer_label"] == composer
        comp_coords = coords_2d[mask]
        if comp_coords.shape[0] < 3:
            continue
        cov = np.cov(comp_coords, rowvar=False)
        if cov.shape != (2, 2):
            continue
        cov = cov + np.eye(2) * 1e-6
        mean = comp_coords.mean(axis=0)
        pdf = _gaussian_pdf(grid_points, mean, cov).reshape(X.shape)
        vmax = float(np.max(pdf))
        if not np.isfinite(vmax) or vmax <= 0:
            continue
        iso = float(vmax * CLOUD_ISO_FRACTION)
        color = palette[idx % len(palette)]
        z_mask = np.where(pdf < iso, np.nan, pdf)
        fig.add_trace(
            go.Contour(
                x=grid_axes[0],
                y=grid_axes[1],
                z=z_mask,
                showscale=False,
                contours=dict(coloring="heatmap"),
                colorscale=[[0.0, color], [1.0, color]],
                opacity=0.38,
                line=dict(width=0),
                name=str(composer),
                hovertemplate=f"Composer: {composer}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Composer Clouds (2D)",
        xaxis_title="dim1",
        yaxis_title="dim2",
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create interactive embeddings of composer features.")
    parser.add_argument("--harmonic", type=Path, default=DEFAULT_HARMONIC, help="Path to harmonic features CSV.")
    parser.add_argument("--melodic", type=Path, default=DEFAULT_MELODIC, help="Path to melodic features CSV.")
    parser.add_argument("--rhythmic", type=Path, default=DEFAULT_RHYTHMIC, help="Path to rhythmic features CSV.")
    parser.add_argument(
        "--method",
        choices=["pca", "tsne"],
        default="pca",
        help="Projection algorithm (default: pca).",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (only used when --method tsne).",
    )
    parser.add_argument(
        "--tsne-early-exaggeration",
        type=float,
        default=12.0,
        help="t-SNE early exaggeration (only used when --method tsne).",
    )
    parser.add_argument(
        "--tsne-learning-rate",
        type=float,
        default=200.0,
        help="t-SNE learning rate (only used when --method tsne).",
    )
    parser.add_argument(
        "--tsne-metric",
        type=str,
        default="euclidean",
        help="Distance metric for t-SNE (only used when --method tsne).",
    )
    parser.add_argument(
        "--tsne-composer-weight",
        type=float,
        default=0.0,
        help=(
            "Additional weight applied to composer one-hot indicators before t-SNE. "
            "Values >0 encourage pieces from the same composer to cluster together."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for projection algorithms.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTDIR / "embedding_3d.html",
        help="Destination HTML file for the interactive scatter.",
    )
    parser.add_argument(
        "--output-2d",
        type=Path,
        default=None,
        help="Optional HTML file for a 2D scatter version of the embedding.",
    )
    parser.add_argument(
        "--loadings-csv",
        type=Path,
        default=None,
        help="Optional CSV path to store PCA loadings (only when --method pca).",
    )
    parser.add_argument(
        "--clouds-output",
        type=Path,
        default=None,
        help=(
            "Optional HTML file for smoothed composer clouds. Useful for PCA embeddings "
            "to compare overall footprint per composer."
        ),
    )
    parser.add_argument(
        "--clouds-output-2d",
        type=Path,
        default=None,
        help="Optional HTML file for a 2D composer cloud view mirroring the point scatter.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    harmonic = _load_features(args.harmonic)
    melodic = _load_features(args.melodic)
    rhythmic = _load_features(args.rhythmic)

    combined = _merge_feature_tables(harmonic, melodic, rhythmic)
    matrix, feature_cols, scaler = _prepare_feature_matrix(combined)
    tsne_matrix = matrix
    if args.method == "tsne" and args.tsne_composer_weight > 0.0:
        composer_dummies = pd.get_dummies(combined["composer_label"], dtype=float)
        composer_dummies -= composer_dummies.mean(axis=0)
        tsne_matrix = np.hstack(
            [
                matrix,
                composer_dummies.values * args.tsne_composer_weight,
            ]
        )
    coords, pca_model = _compute_projection(
        tsne_matrix,
        args.method,
        args.seed,
        args.perplexity,
        args.tsne_early_exaggeration,
        args.tsne_learning_rate,
        args.tsne_metric,
    )
    if args.method == "tsne" and args.tsne_composer_weight > 0.0:
        print(
            f"Applied composer weight {args.tsne_composer_weight:.2f} to t-SNE input to encourage per-composer clustering."
        )
    _plot_embedding(combined, coords, args.output)

    if args.output_2d is not None:
        _plot_embedding_2d(combined, coords, args.output_2d)
        print(f"2D embedding written to {args.output_2d}")

    print(f"Embedding generated for {len(combined)} pieces using {len(feature_cols)} features -> {args.output}")
    if args.method == "pca" and pca_model is not None:
        explained = pca_model.explained_variance_ratio_ * 100.0
        print(
            "Explained variance (components 1-3): "
            + ", ".join(f"{value:.1f}%" for value in explained)
        )
        if args.loadings_csv is not None:
            loadings = pd.DataFrame(
                pca_model.components_.T,
                index=feature_cols,
                columns=["PC1", "PC2", "PC3"],
            )
            loadings_path = args.loadings_csv
            loadings_path.parent.mkdir(parents=True, exist_ok=True)
            loadings.to_csv(loadings_path)
            print(f"Saved PCA loadings to {loadings_path}")
    elif args.method == "tsne":
        print("t-SNE axes are non-linear embeddings; absolute directions are not individually interpretable.")
    if args.clouds_output is not None:
        _plot_composer_clouds(combined, coords, args.clouds_output)
        print(f"Composer cloud view written to {args.clouds_output}")
    if args.clouds_output_2d is not None:
        _plot_composer_clouds_2d(combined, coords, args.clouds_output_2d)
        print(f"2D composer cloud view written to {args.clouds_output_2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
