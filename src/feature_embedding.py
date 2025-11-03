"""Generate interactive embeddings of feature tables using dimensionality reduction."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

DEFAULT_HARMONIC = Path("data/features/harmonic_features.csv")
DEFAULT_MELODIC = Path("data/features/melodic_features.csv")
DEFAULT_RHYTHMIC = Path("data/features/rhythmic_features.csv")
DEFAULT_OUTDIR = Path("figures/embeddings")


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
    if not numeric_cols:
        raise ValueError("No numeric feature columns available for projection.")
    filled = df[numeric_cols].copy()
    filled = filled.fillna(filled.mean())
    scaler = StandardScaler()
    matrix = scaler.fit_transform(filled.values)
    return matrix, numeric_cols, scaler


def _compute_projection(matrix: np.ndarray, method: str, seed: int, perplexity: float) -> tuple[np.ndarray, Optional[PCA]]:
    if method == "pca":
        model = PCA(n_components=3, random_state=seed)
        coords = model.fit_transform(matrix)
        return coords, model
    if method == "tsne":
        model = TSNE(n_components=3, init="pca", random_state=seed, perplexity=perplexity)
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for projection algorithms.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTDIR / "embedding_3d.html",
        help="Destination HTML file for the interactive scatter.",
    )
    parser.add_argument(
        "--loadings-csv",
        type=Path,
        default=None,
        help="Optional CSV path to store PCA loadings (only when --method pca).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    harmonic = _load_features(args.harmonic)
    melodic = _load_features(args.melodic)
    rhythmic = _load_features(args.rhythmic)

    combined = _merge_feature_tables(harmonic, melodic, rhythmic)
    matrix, feature_cols, scaler = _prepare_feature_matrix(combined)
    coords, pca_model = _compute_projection(matrix, args.method, args.seed, args.perplexity)
    _plot_embedding(combined, coords, args.output)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
