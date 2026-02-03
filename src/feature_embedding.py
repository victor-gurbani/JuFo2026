"""Generate interactive embeddings of feature tables using dimensionality reduction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
DEFAULT_EMBEDDING_CACHE = Path("data/embeddings/pca_embedding_cache.csv")
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _meta_path_for(cache_csv: Path) -> Path:
    # Matches embedding_cache.py naming convention: <cache>.csv.meta.json
    return cache_csv.with_suffix(cache_csv.suffix + ".meta.json")


def _load_cache_meta(cache_csv: Path) -> Dict[str, Any]:
    meta_path = _meta_path_for(cache_csv)
    if not meta_path.exists():
        raise FileNotFoundError(f"Embedding cache metadata not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _cache_is_compatible_for_pca(
    *,
    cache_csv: Path,
    harmonic_path: Path,
    melodic_path: Path,
    rhythmic_path: Path,
    seed: int,
    excluded_features: set[str],
) -> Tuple[bool, str]:
    if not cache_csv.exists():
        return False, f"cache not found: {cache_csv}"

    try:
        meta = _load_cache_meta(cache_csv)
    except Exception as exc:
        return False, f"failed to load cache meta: {exc}"

    inputs = meta.get("inputs")
    if not isinstance(inputs, dict):
        return False, "cache meta missing inputs"

    # Require PCA with matching seed.
    if inputs.get("method") != "pca":
        return False, "cache method is not pca"
    if int(inputs.get("seed", -1)) != int(seed):
        return False, "seed mismatch"

    expected_excluded = list(sorted(excluded_features))
    if inputs.get("excluded_features") != expected_excluded:
        return False, "excluded feature list mismatch"

    # Hash-based validation lets users pass equivalent paths.
    for label, path, key in (
        ("harmonic", harmonic_path, "harmonic_sha256"),
        ("melodic", melodic_path, "melodic_sha256"),
        ("rhythmic", rhythmic_path, "rhythmic_sha256"),
    ):
        if not path.exists():
            return False, f"{label} features file missing: {path}"
        expected_hash = inputs.get(key)
        if not isinstance(expected_hash, str) or not expected_hash:
            return False, f"cache meta missing {key}"
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            return False, f"{label} features hash mismatch"

    pca = meta.get("pca")
    if not isinstance(pca, dict):
        return False, "cache meta missing PCA artifacts"
    if not pca.get("pca_components"):
        return False, "cache meta missing PCA components"
    if not pca.get("feature_columns"):
        return False, "cache meta missing feature column list"

    # Lightweight CSV schema check.
    try:
        header = pd.read_csv(cache_csv, nrows=0)
    except Exception as exc:
        return False, f"failed to read cache header: {exc}"
    for col in ("mxl_path", "dim1", "dim2", "dim3"):
        if col not in header.columns:
            return False, f"cache missing required column: {col}"

    return True, "ok"


def _load_coords_from_cache(combined: pd.DataFrame, cache_csv: Path) -> Optional[np.ndarray]:
    if combined.empty:
        return None
    if "mxl_path" not in combined.columns:
        return None
    if combined["mxl_path"].duplicated().any():
        return None

    cache_df = pd.read_csv(cache_csv)
    if cache_df.empty or "mxl_path" not in cache_df.columns:
        return None
    if cache_df["mxl_path"].duplicated().any():
        return None
    for col in ("dim1", "dim2", "dim3"):
        if col not in cache_df.columns:
            return None

    joined = combined.set_index("mxl_path").join(
        cache_df.set_index("mxl_path")[["dim1", "dim2", "dim3"]],
        how="left",
    )
    if joined[["dim1", "dim2", "dim3"]].isna().any().any():
        return None
    coords = joined[["dim1", "dim2", "dim3"]].to_numpy(dtype=float)
    return coords


def _axis_titles(method: str) -> tuple[str, str, str]:
    if method == "pca":
        return (
            "PC1 (Chromatik/Dissonanz)",
            "PC2 (Dichte/Klarheit)",
            "PC3 (Registral/Textur)",
        )
    return ("dim1", "dim2", "dim3")


def _write_plotly_figure(fig: go.Figure, output_path: Path, write_html_also: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    fig.update_layout(margin=dict(l=0, r=0, t=26, b=0))
    try:
        json_path = output_path.with_suffix(".json")
        fig.write_json(str(json_path))
    except Exception:
        pass
    if suffix == ".html":
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        return
    if suffix in {".png", ".pdf", ".svg"}:
        try:
            fig.write_image(str(output_path))
        except ValueError as exc:
            raise ValueError(
                "Static image export requires the 'kaleido' package. "
                "Install it (e.g., pip install kaleido) or write to .html instead."
            ) from exc

        # Backwards compatible behavior: historically this script always produced HTML.
        # If the user requests a static image, also write a companion HTML next to it.
        if write_html_also:
            html_path = output_path.with_suffix(".html")
            fig.write_html(str(html_path), include_plotlyjs="cdn")
        return
    raise ValueError(f"Unsupported output format: {output_path} (use .html or .png)")


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


def _plot_embedding(
    df: pd.DataFrame,
    coords: np.ndarray,
    output_path: Path,
    axis_titles: tuple[str, str, str],
    title: str,
    write_html_also: bool,
) -> None:
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
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=axis_titles[0], yaxis_title=axis_titles[1], zaxis_title=axis_titles[2]),
    )
    _write_plotly_figure(fig, output_path, write_html_also=write_html_also)


def _plot_embedding_2d(
    df: pd.DataFrame,
    coords: np.ndarray,
    output_path: Path,
    axis_titles: tuple[str, str],
    title: str,
    write_html_also: bool,
) -> None:
    if coords.shape[1] < 2:
        raise ValueError("Projection returned fewer than two dimensions; cannot create 2D scatter.")
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
    fig.update_layout(title=title, xaxis_title=axis_titles[0], yaxis_title=axis_titles[1])
    _write_plotly_figure(fig, output_path, write_html_also=write_html_also)


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


def _plot_composer_clouds(
    df: pd.DataFrame,
    coords: np.ndarray,
    output_path: Path,
    axis_titles: tuple[str, str, str],
    title: str,
    write_html_also: bool,
) -> None:
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
            go.Scatter3d(
                x=comp_coords[:, 0],
                y=comp_coords[:, 1],
                z=comp_coords[:, 2],
                mode="markers",
                marker=dict(size=2, color=color, opacity=0.35),
                name=str(composer),
                showlegend=False,
                hovertemplate=(
                    "Composer: %{customdata[0]}<br>Title: %{customdata[1]}<extra></extra>"
                ),
                customdata=df.loc[mask, ["composer_label", "title"]].values,
            )
        )
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
        title=title,
        scene=dict(xaxis_title=axis_titles[0], yaxis_title=axis_titles[1], zaxis_title=axis_titles[2]),
        legend=dict(itemsizing="constant"),
    )
    _write_plotly_figure(fig, output_path, write_html_also=write_html_also)


def _plot_composer_clouds_2d(
    df: pd.DataFrame,
    coords: np.ndarray,
    output_path: Path,
    axis_titles: tuple[str, str],
    title: str,
    write_html_also: bool,
) -> None:
    if coords.shape[1] < 2:
        raise ValueError("Projection returned fewer than two dimensions; cannot create 2D clouds.")
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
        title=title,
        xaxis_title=axis_titles[0],
        yaxis_title=axis_titles[1],
        legend=dict(itemsizing="constant"),
    )
    _write_plotly_figure(fig, output_path, write_html_also=write_html_also)


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
        "--embedding-cache",
        type=Path,
        default=DEFAULT_EMBEDDING_CACHE,
        help=(
            "Optional PCA embedding cache CSV (created by src/embedding_cache.py). "
            "When --method pca and the cache is compatible with the input feature CSVs, "
            "coordinates are reused instead of refitting PCA."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable using the PCA embedding cache even if present.",
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
        help="Destination .html or .png for the 3D scatter (when using a static format, an .html is also written).",
    )
    parser.add_argument(
        "--output-2d",
        type=Path,
        default=None,
        help="Optional .html or .png for a 2D scatter version of the embedding (static formats also write .html).",
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
            "Optional .html or .png for smoothed composer clouds. Useful for PCA embeddings "
            "to compare overall footprint per composer (static formats also write .html)."
        ),
    )
    parser.add_argument(
        "--clouds-output-2d",
        type=Path,
        default=None,
        help="Optional .html or .png for a 2D composer cloud view mirroring the point scatter (static formats also write .html).",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Do not emit companion .html files when writing static images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    harmonic = _load_features(args.harmonic)
    melodic = _load_features(args.melodic)
    rhythmic = _load_features(args.rhythmic)

    combined = _merge_feature_tables(harmonic, melodic, rhythmic)

    used_cache = False
    coords: Optional[np.ndarray] = None
    pca_model: Optional[PCA] = None
    cached_meta: Optional[Dict[str, Any]] = None

    if args.method == "pca" and not args.no_cache:
        compatible, reason = _cache_is_compatible_for_pca(
            cache_csv=args.embedding_cache,
            harmonic_path=args.harmonic,
            melodic_path=args.melodic,
            rhythmic_path=args.rhythmic,
            seed=args.seed,
            excluded_features=EXCLUDED_FEATURES,
        )
        if compatible:
            coords_from_cache = _load_coords_from_cache(combined, args.embedding_cache)
            if coords_from_cache is not None:
                coords = coords_from_cache
                used_cache = True
                try:
                    cached_meta = _load_cache_meta(args.embedding_cache)
                except Exception:
                    cached_meta = None
            else:
                print(f"[info] Cache is compatible but did not contain all rows; refitting PCA.")
        else:
            print(f"[info] Embedding cache not used ({reason}); computing from scratch.")

    if not used_cache:
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

    if coords is None:
        raise RuntimeError("Failed to compute or load embedding coordinates.")
    if args.method == "tsne" and args.tsne_composer_weight > 0.0:
        print(
            f"Applied composer weight {args.tsne_composer_weight:.2f} to t-SNE input to encourage per-composer clustering."
        )

    axis_titles = _axis_titles(args.method)
    base_title = "PCA-Projektion" if args.method == "pca" else "t-SNE-Projektion"
    write_html_also = not args.no_html
    _plot_embedding(
        combined,
        coords,
        args.output,
        axis_titles,
        f"{base_title} (3D)",
        write_html_also,
    )

    if args.output_2d is not None:
        _plot_embedding_2d(
            combined,
            coords,
            args.output_2d,
            axis_titles[:2],
            f"{base_title} (2D)",
            write_html_also,
        )
        print(f"2D embedding written to {args.output_2d}")

    cache_note = " (used embedding cache)" if used_cache else ""
    feature_count_note = "?"
    try:
        numeric_cols = [col for col in combined.columns if pd.api.types.is_numeric_dtype(combined[col])]
        numeric_cols = [col for col in numeric_cols if col not in {"mxl_path"}]
        numeric_cols = [col for col in numeric_cols if col not in EXCLUDED_FEATURES]
        feature_count_note = str(len(numeric_cols))
    except Exception:
        pass
    print(f"Embedding generated for {len(combined)} pieces using {feature_count_note} features -> {args.output}{cache_note}")

    if args.method == "pca":
        if used_cache and cached_meta is not None:
            try:
                pca = cached_meta.get("pca")
                explained = np.array(pca.get("explained_variance_ratio", []), dtype=float) * 100.0  # type: ignore[union-attr]
                if explained.size >= 3:
                    print(
                        "Explained variance (components 1-3): "
                        + ", ".join(f"{value:.1f}%" for value in explained[:3])
                    )
                if args.loadings_csv is not None:
                    feature_cols = list(pca.get("feature_columns") or [])  # type: ignore[union-attr]
                    components = np.array(pca.get("pca_components", []), dtype=float)  # type: ignore[union-attr]
                    if components.shape[0] == 3 and components.shape[1] == len(feature_cols):
                        loadings = pd.DataFrame(
                            components.T,
                            index=feature_cols,
                            columns=["PC1", "PC2", "PC3"],
                        )
                        loadings_path = args.loadings_csv
                        loadings_path.parent.mkdir(parents=True, exist_ok=True)
                        loadings.to_csv(loadings_path)
                        print(f"Saved PCA loadings to {loadings_path}")
            except Exception:
                pass
        elif pca_model is not None:
            explained = pca_model.explained_variance_ratio_ * 100.0
            print(
                "Explained variance (components 1-3): "
                + ", ".join(f"{value:.1f}%" for value in explained)
            )
            if args.loadings_csv is not None:
                matrix, feature_cols, _ = _prepare_feature_matrix(combined)
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
        _plot_composer_clouds(
            combined,
            coords,
            args.clouds_output,
            axis_titles,
            "Komponistenwolken (3D)",
            write_html_also,
        )
        print(f"Composer cloud view written to {args.clouds_output}")
    if args.clouds_output_2d is not None:
        _plot_composer_clouds_2d(
            combined,
            coords,
            args.clouds_output_2d,
            axis_titles[:2],
            "Komponistenwolken (2D)",
            write_html_also,
        )
        print(f"2D composer cloud view written to {args.clouds_output_2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
