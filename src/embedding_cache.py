"""Build and consume a persistent embedding cache for all pieces.

This project computes a 3D stylistic embedding (typically PCA) over the full
feature matrix. Some workflows (e.g., highlighting a specific piece) previously
recomputed the PCA projection from scratch each time.

This module provides:
- A CLI to compute and store embeddings for all pieces in a CSV.
- A sidecar JSON metadata file containing a reproducible signature and, for PCA,
  the fitted imputation/scaling/projection parameters.
- Helpers to load and validate the cache.

The cache is written atomically so the process can be interrupted safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from harmonic_features import compute_harmonic_features  # type: ignore[attr-defined]
from melodic_features import compute_melodic_features  # type: ignore[attr-defined]
from rhythmic_features import compute_rhythmic_features  # type: ignore[attr-defined]
from score_parser import parse_score  # type: ignore[attr-defined]

from feature_embedding import (  # type: ignore[attr-defined]
    DEFAULT_HARMONIC,
    DEFAULT_MELODIC,
    DEFAULT_RHYTHMIC,
    EXCLUDED_FEATURES,
    _compute_projection,
    _load_features,
    _merge_feature_tables,
    _prepare_feature_matrix,
)

CACHE_VERSION = 1
DEFAULT_CACHE_CSV = Path("data/embeddings/pca_embedding_cache.csv")
DEFAULT_DONE_SUFFIX = ".done.txt"

REQUIRED_CACHE_COLUMNS = (
    "composer_label",
    "title",
    "mxl_path",
    "mxl_abs_path",
    "dim1",
    "dim2",
    "dim3",
)


def load_meta(cache_csv: Path) -> Dict[str, Any]:
    meta_json = meta_path_for(cache_csv)
    if not meta_json.exists():
        raise FileNotFoundError(f"Cache metadata not found: {meta_json}")
    return _load_meta(meta_json)


def lookup_cached_embedding(cache_df: pd.DataFrame, mxl_abs_path: str) -> Optional[pd.Series]:
    if cache_df.empty:
        return None
    if "mxl_abs_path" not in cache_df.columns:
        return None
    matches = cache_df.index[cache_df["mxl_abs_path"].astype(str) == str(mxl_abs_path)]
    if len(matches) == 0:
        return None
    return cache_df.loc[matches[0]]


def project_features_into_cached_pca(feature_mapping: Dict[str, object], meta: Dict[str, Any]) -> np.ndarray:
    """Project a single piece into the cached PCA space.

    Requires that `meta` contains a `pca` section produced by this module.
    """

    pca = meta.get("pca")
    if not isinstance(pca, dict):
        raise ValueError("Cache meta does not contain PCA artifacts.")
    feature_columns = pca.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("Cache meta PCA artifacts missing feature_columns.")

    impute_means = pca.get("impute_means")
    if not isinstance(impute_means, dict):
        impute_means = {}

    def _to_float(value: object) -> float:
        if value is None:
            return float("nan")
        if isinstance(value, (bool,)):
            return float(int(value))
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, (float, np.floating)):
            return float(value)
        try:
            text = str(value).strip()
            if not text:
                return float("nan")
            return float(text)
        except Exception:
            return float("nan")

    x = np.array([_to_float(feature_mapping.get(col)) for col in feature_columns], dtype=float)
    for idx, col in enumerate(feature_columns):
        if np.isfinite(x[idx]):
            continue
        mean_val = impute_means.get(col, None)
        if mean_val is None:
            raise ValueError(f"Missing value for feature '{col}' and no imputation mean available.")
        try:
            x[idx] = float(mean_val)
        except Exception as exc:
            raise ValueError(
                f"Imputation mean for feature '{col}' is not numeric: {mean_val!r}"
            ) from exc

    scaler_mean = np.array(pca.get("scaler_mean", []), dtype=float)
    scaler_scale = np.array(pca.get("scaler_scale", []), dtype=float)
    if scaler_mean.shape[0] != x.shape[0] or scaler_scale.shape[0] != x.shape[0]:
        raise ValueError("Scaler parameters in cache meta do not match feature vector length.")
    safe_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
    x_scaled = (x - scaler_mean) / safe_scale

    pca_mean = np.array(pca.get("pca_mean", []), dtype=float)
    if pca_mean.shape[0] != x_scaled.shape[0]:
        raise ValueError("PCA mean in cache meta does not match feature vector length.")

    components = np.array(pca.get("pca_components", []), dtype=float)
    if components.ndim != 2 or components.shape[1] != x_scaled.shape[0] or components.shape[0] < 3:
        raise ValueError("PCA components in cache meta have an unexpected shape.")

    coords = (x_scaled - pca_mean) @ components.T
    return coords[:3]


def normalize_path(value: Path | str) -> str:
    text = str(value).strip()
    return str(Path(text).expanduser().resolve())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _piece_id_digest(piece_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for piece_id in sorted(set(piece_ids)):
        digest.update(piece_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def meta_path_for(cache_csv: Path) -> Path:
    return cache_csv.with_suffix(cache_csv.suffix + ".meta.json")


def done_path_for(cache_csv: Path) -> Path:
    return cache_csv.with_suffix(cache_csv.suffix + DEFAULT_DONE_SUFFIX)


@dataclass(frozen=True)
class CacheInputSpec:
    method: str
    seed: int
    perplexity: float
    early_exaggeration: float
    learning_rate: float
    metric: str

    harmonic_path: str
    melodic_path: str
    rhythmic_path: str
    harmonic_sha256: str
    melodic_sha256: str
    rhythmic_sha256: str

    excluded_features: Tuple[str, ...]


@dataclass(frozen=True)
class PcaArtifacts:
    feature_columns: Tuple[str, ...]
    impute_means: Dict[str, float]
    scaler_mean: Tuple[float, ...]
    scaler_scale: Tuple[float, ...]
    pca_components: Tuple[Tuple[float, ...], ...]
    pca_mean: Tuple[float, ...]
    explained_variance_ratio: Tuple[float, ...]


def build_cache(
    harmonic_path: Path,
    melodic_path: Path,
    rhythmic_path: Path,
    method: str,
    seed: int,
    perplexity: float,
    early_exaggeration: float,
    learning_rate: float,
    metric: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    harmonic = _load_features(harmonic_path)
    melodic = _load_features(melodic_path)
    rhythmic = _load_features(rhythmic_path)

    combined = _merge_feature_tables(harmonic, melodic, rhythmic)
    if "mxl_path" not in combined.columns:
        raise ValueError("Merged features are missing 'mxl_path'.")

    combined = combined.copy()
    combined["mxl_abs_path"] = combined["mxl_path"].apply(normalize_path)

    if combined["mxl_abs_path"].duplicated().any():
        duplicates = combined.loc[combined["mxl_abs_path"].duplicated(), "mxl_abs_path"].head(10).tolist()
        raise ValueError(
            "Duplicate mxl_abs_path entries detected in merged features; cannot build a stable cache. "
            f"Examples: {duplicates}"
        )

    matrix, feature_cols, scaler = _prepare_feature_matrix(combined)

    coords, pca_model = _compute_projection(
        matrix,
        method=method,
        seed=seed,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        metric=metric,
    )

    cache_df = combined[["composer_label", "title", "mxl_path", "mxl_abs_path"]].copy()
    cache_df[["dim1", "dim2", "dim3"]] = coords[:, :3]

    input_spec = CacheInputSpec(
        method=method,
        seed=seed,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        metric=metric,
        harmonic_path=str(harmonic_path),
        melodic_path=str(melodic_path),
        rhythmic_path=str(rhythmic_path),
        harmonic_sha256=_sha256_file(harmonic_path),
        melodic_sha256=_sha256_file(melodic_path),
        rhythmic_sha256=_sha256_file(rhythmic_path),
        excluded_features=tuple(sorted(EXCLUDED_FEATURES)),
    )

    meta: Dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": asdict(input_spec),
        "piece_id_column": "mxl_abs_path",
        "piece_count": int(len(cache_df)),
        "piece_id_digest": _piece_id_digest(cache_df["mxl_abs_path"].astype(str).tolist()),
    }

    if method == "pca":
        if pca_model is None:
            raise RuntimeError("Expected PCA model artifacts but projection returned None.")
        impute_means_series = combined[feature_cols].mean(numeric_only=True)
        impute_means = {str(k): float(v) for k, v in impute_means_series.items() if np.isfinite(v)}
        artifacts = PcaArtifacts(
            feature_columns=tuple(feature_cols),
            impute_means=impute_means,
            scaler_mean=tuple(float(x) for x in getattr(scaler, "mean_", [])),
            scaler_scale=tuple(float(x) for x in getattr(scaler, "scale_", [])),
            pca_components=tuple(tuple(float(v) for v in row) for row in pca_model.components_),
            pca_mean=tuple(float(v) for v in getattr(pca_model, "mean_", [])),
            explained_variance_ratio=tuple(float(v) for v in pca_model.explained_variance_ratio_),
        )
        meta["pca"] = asdict(artifacts)

    return cache_df, meta


def _resolve_musicxml_path(raw_path: str, repo_root: Path, dataset_root: Optional[Path]) -> Optional[Path]:
    """Resolve a raw mxl path from various CSVs into an absolute existing path."""
    raw = str(raw_path).strip()
    if not raw:
        return None

    candidates: List[Path] = []
    raw_path_obj = Path(raw)
    if raw_path_obj.is_absolute():
        candidates.append(raw_path_obj)
    else:
        candidates.append(repo_root / raw)
        if dataset_root is not None:
            candidates.append(dataset_root / raw)
        # Common PDMX-style entries may include "15571083/..." already.
        candidates.append(repo_root / "15571083" / raw)
        if "/mxl/" in raw:
            mxl_index = raw.index("/mxl/")
            tail = raw[mxl_index + 1 :]
            candidates.append(repo_root / "15571083" / tail)
            candidates.append(repo_root / "15571083" / raw[mxl_index + 5 :])

    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if resolved.exists():
            return resolved
    return None


def _iter_corpus_records(
    corpus_csv: Optional[Path],
    paths_file: Optional[Path],
    repo_root: Path,
    dataset_root: Optional[Path],
) -> Iterator[Tuple[str, str, Path]]:
    """Yield (composer_label, title, abs_path) records."""

    if corpus_csv is not None:
        if not corpus_csv.exists():
            raise FileNotFoundError(f"Corpus CSV not found: {corpus_csv}")
        df = pd.read_csv(corpus_csv)
        if df.empty:
            return iter(())

        composer_col_candidates = ["composer_label", "composer", "composer_name"]
        title_col_candidates = ["title", "song_name"]
        path_col_candidates = [
            "mxl_abs_path",
            "mxl_path",
            "mxl_rel_path",
            "mxl_rel",
            "mxl",
        ]

        composer_col = next((c for c in composer_col_candidates if c in df.columns), None)
        title_col = next((c for c in title_col_candidates if c in df.columns), None)
        path_col = next((c for c in path_col_candidates if c in df.columns), None)
        if path_col is None:
            raise ValueError(
                f"Corpus CSV {corpus_csv} is missing a MusicXML path column. Tried: {path_col_candidates}"
            )

        for _, row in df.iterrows():
            raw_path = row.get(path_col)
            if not isinstance(raw_path, str):
                continue
            resolved = _resolve_musicxml_path(raw_path, repo_root=repo_root, dataset_root=dataset_root)
            if resolved is None:
                continue
            composer = str(row.get(composer_col) or "Unknown").strip() if composer_col else "Unknown"
            title = str(row.get(title_col) or resolved.stem).strip() if title_col else resolved.stem
            yield composer or "Unknown", title or resolved.stem, resolved
        return

    if paths_file is not None:
        if not paths_file.exists():
            raise FileNotFoundError(f"Paths file not found: {paths_file}")
        for line in paths_file.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            resolved = _resolve_musicxml_path(raw, repo_root=repo_root, dataset_root=dataset_root)
            if resolved is None:
                continue
            yield "Unknown", resolved.stem, resolved
        return

    raise ValueError("Provide either --corpus-csv or --paths to define which pieces to cache.")


def _load_done_set(done_path: Path) -> set[str]:
    if not done_path.exists():
        return set()
    done: set[str] = set()
    for line in done_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            done.add(text)
    return done


def _scan_existing_cache_for_done(cache_csv: Path) -> set[str]:
    if not cache_csv.exists():
        return set()
    done: set[str] = set()
    with cache_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = (row.get("mxl_abs_path") or "").strip()
            if value:
                done.add(value)
    return done


def _append_embedding_row(
    cache_csv: Path,
    row: Dict[str, object],
) -> None:
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = cache_csv.exists() and cache_csv.stat().st_size > 0
    with cache_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_CACHE_COLUMNS))
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in REQUIRED_CACHE_COLUMNS})
        handle.flush()


def project_corpus_into_cached_pca(
    *,
    model_cache_csv: Path,
    output_cache_csv: Path,
    corpus_csv: Optional[Path],
    paths_file: Optional[Path],
    dataset_root: Optional[Path],
    resume: bool,
    force: bool,
    limit: Optional[int],
    skip_errors: bool,
    checkpoint_every: int,
) -> int:
    """Project an arbitrary corpus into an existing PCA cache space.

    This is meant for long-running caching (days) across large corpora.
    It appends to output_cache_csv and maintains a sidecar done-file for resume.
    """

    if not model_cache_csv.exists():
        raise FileNotFoundError(f"Model cache CSV not found: {model_cache_csv}")

    model_meta = load_meta(model_cache_csv)

    repo_root = Path(__file__).resolve().parent.parent
    dataset_root_resolved = dataset_root.resolve() if dataset_root is not None else None

    if output_cache_csv.exists() and not resume and not force:
        raise FileExistsError(
            f"Output cache already exists: {output_cache_csv}. Use --resume to continue or --force to overwrite."
        )
    if force and output_cache_csv.exists() and not resume:
        output_cache_csv.unlink()
        meta_path_for(output_cache_csv).unlink(missing_ok=True)  # type: ignore[arg-type]
        done_path_for(output_cache_csv).unlink(missing_ok=True)  # type: ignore[arg-type]

    done_path = done_path_for(output_cache_csv)
    done = _load_done_set(done_path)
    if not done and output_cache_csv.exists():
        # Fallback for older runs: reconstruct done set from CSV.
        done = _scan_existing_cache_for_done(output_cache_csv)
        if done:
            done_path.parent.mkdir(parents=True, exist_ok=True)
            done_path.write_text("\n".join(sorted(done)) + "\n", encoding="utf-8")

    # Ensure meta exists early so other consumers know which PCA space this cache is in.
    meta_out = dict(model_meta)
    meta_out["cache_version"] = CACHE_VERSION
    meta_out["created_at"] = datetime.now(timezone.utc).isoformat()
    meta_out["derived_from"] = {
        "model_cache_csv": str(model_cache_csv),
        "model_cache_meta": str(meta_path_for(model_cache_csv)),
    }
    meta_out["in_progress"] = True
    meta_out["piece_count"] = int(len(done))
    _atomic_write_text(meta_path_for(output_cache_csv), json.dumps(meta_out, indent=2, sort_keys=True))

    processed = 0
    written_since_checkpoint = 0
    with done_path.open("a", encoding="utf-8") as done_handle:
        for composer, title, abs_path in _iter_corpus_records(
            corpus_csv=corpus_csv,
            paths_file=paths_file,
            repo_root=repo_root,
            dataset_root=dataset_root_resolved,
        ):
            if limit is not None and processed >= limit:
                break
            processed += 1

            mxl_abs = normalize_path(abs_path)
            if mxl_abs in done:
                continue

            try:
                score = parse_score(abs_path)
                harmonic_metrics = compute_harmonic_features(score)
                melodic_metrics = compute_melodic_features(score)
                rhythmic_metrics = compute_rhythmic_features(score)
                feature_mapping: Dict[str, object] = {
                    **harmonic_metrics,
                    **melodic_metrics,
                    **rhythmic_metrics,
                }
                coords = project_features_into_cached_pca(feature_mapping, model_meta)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if skip_errors:
                    print(f"[warn] Skipping {abs_path} due to error: {exc}")
                    continue
                raise

            row = {
                "composer_label": composer,
                "title": title,
                "mxl_path": mxl_abs,
                "mxl_abs_path": mxl_abs,
                "dim1": float(coords[0]),
                "dim2": float(coords[1]),
                "dim3": float(coords[2]),
            }
            _append_embedding_row(output_cache_csv, row)
            done.add(mxl_abs)
            done_handle.write(mxl_abs + "\n")
            done_handle.flush()
            written_since_checkpoint += 1

            if checkpoint_every > 0 and written_since_checkpoint >= checkpoint_every:
                meta_out["piece_count"] = int(len(done))
                meta_out["last_checkpoint_at"] = datetime.now(timezone.utc).isoformat()
                _atomic_write_text(
                    meta_path_for(output_cache_csv),
                    json.dumps(meta_out, indent=2, sort_keys=True),
                )
                written_since_checkpoint = 0

    meta_out["in_progress"] = False
    meta_out["piece_count"] = int(len(done))
    try:
        meta_out["piece_id_digest"] = _piece_id_digest(done)
    except Exception:
        pass
    _atomic_write_text(meta_path_for(output_cache_csv), json.dumps(meta_out, indent=2, sort_keys=True))
    return int(len(done))


def _load_meta(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cache_is_compatible(
    cache_csv: Path,
    meta_json: Path,
    harmonic_path: Path,
    melodic_path: Path,
    rhythmic_path: Path,
    method: str,
    seed: int,
    perplexity: float,
    early_exaggeration: float,
    learning_rate: float,
    metric: str,
) -> Tuple[bool, str]:
    if not cache_csv.exists():
        return False, f"cache csv not found: {cache_csv}"
    if not meta_json.exists():
        return False, f"cache meta not found: {meta_json}"

    try:
        meta = _load_meta(meta_json)
    except Exception as exc:
        return False, f"failed to read meta json: {exc}"

    inputs = meta.get("inputs")
    if not isinstance(inputs, dict):
        return False, "meta missing inputs spec"

    expected = {
        "method": method,
        "seed": seed,
        "perplexity": perplexity,
        "early_exaggeration": early_exaggeration,
        "learning_rate": learning_rate,
        "metric": metric,
        "harmonic_path": str(harmonic_path),
        "melodic_path": str(melodic_path),
        "rhythmic_path": str(rhythmic_path),
        "harmonic_sha256": _sha256_file(harmonic_path) if harmonic_path.exists() else None,
        "melodic_sha256": _sha256_file(melodic_path) if melodic_path.exists() else None,
        "rhythmic_sha256": _sha256_file(rhythmic_path) if rhythmic_path.exists() else None,
        "excluded_features": list(sorted(EXCLUDED_FEATURES)),
    }

    for key, value in expected.items():
        if value is None:
            return False, f"input file missing for {key}"
        if inputs.get(key) != value:
            return False, f"meta mismatch for {key}"

    # Lightweight sanity-check of the CSV schema.
    try:
        header = pd.read_csv(cache_csv, nrows=0)
    except Exception as exc:
        return False, f"failed to read cache csv header: {exc}"

    missing_cols = [col for col in REQUIRED_CACHE_COLUMNS if col not in header.columns]
    if missing_cols:
        return False, f"cache csv missing columns: {missing_cols}"

    return True, "ok"


def load_cache(cache_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cache_csv)
    missing_cols = [col for col in REQUIRED_CACHE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Embedding cache missing required columns: {missing_cols}")
    return df


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a persistent 3D embedding (PCA/t-SNE) for all pieces and store it in a CSV cache. "
            "Writes a sidecar .meta.json with reproducible parameters and, for PCA, the projection artifacts."
        )
    )
    parser.add_argument("--harmonic", type=Path, default=DEFAULT_HARMONIC, help="Path to harmonic features CSV.")
    parser.add_argument("--melodic", type=Path, default=DEFAULT_MELODIC, help="Path to melodic features CSV.")
    parser.add_argument("--rhythmic", type=Path, default=DEFAULT_RHYTHMIC, help="Path to rhythmic features CSV.")
    parser.add_argument(
        "--method",
        choices=["pca", "tsne"],
        default="pca",
        help="Projection method (default: pca).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (PCA/t-SNE).")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity (tsne only).")
    parser.add_argument("--tsne-early-exaggeration", type=float, default=12.0, help="t-SNE early exaggeration.")
    parser.add_argument("--tsne-learning-rate", type=float, default=200.0, help="t-SNE learning rate.")
    parser.add_argument("--tsne-metric", type=str, default="euclidean", help="t-SNE distance metric.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CACHE_CSV,
        help="Destination CSV for the embedding cache.",
    )
    parser.add_argument(
        "--project-only",
        action="store_true",
        help=(
            "Do not fit PCA from the feature CSVs. Instead, load PCA artifacts from --model-cache and "
            "project a corpus (from --corpus-csv or --paths) into that cached PCA space, appending results to --output-csv."
        ),
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=DEFAULT_CACHE_CSV,
        help=(
            "PCA cache CSV that contains the PCA artifacts in its .meta.json (used with --project-only). "
            "Default: the canonical cache path."
        ),
    )
    parser.add_argument(
        "--corpus-csv",
        type=Path,
        default=None,
        help=(
            "Optional corpus CSV defining which pieces to cache (supports curated or full datasets). "
            "Must contain a MusicXML path column such as mxl_abs_path, mxl_path, or mxl."
        ),
    )
    parser.add_argument(
        "--paths",
        type=Path,
        default=None,
        help="Optional newline-delimited text file of MusicXML paths to cache (alternative to --corpus-csv).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Optional dataset root used to resolve relative mxl paths (e.g., 15571083). "
            "Useful when caching from full PDMX.csv."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Resume a partially built projection cache.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of corpus records to attempt (smoke tests).",
    )
    parser.add_argument(
        "--no-skip-errors",
        action="store_true",
        help="Abort on the first projection error instead of skipping failing files.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Update meta + flush progress every N new cached pieces (projection mode).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild cache even if an apparently compatible cache already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    cache_csv: Path = args.output_csv
    meta_json = meta_path_for(cache_csv)

    if args.project_only:
        try:
            count = project_corpus_into_cached_pca(
                model_cache_csv=args.model_cache,
                output_cache_csv=cache_csv,
                corpus_csv=args.corpus_csv,
                paths_file=args.paths,
                dataset_root=args.dataset_root,
                resume=args.resume,
                force=args.force,
                limit=args.limit,
                skip_errors=not args.no_skip_errors,
                checkpoint_every=int(args.checkpoint_every),
            )
        except KeyboardInterrupt:
            print("[warn] Interrupted; progress preserved. Re-run with --resume to continue.")
            return 130
        print(f"Projection cache updated: {cache_csv} ({count} pieces)")
        return 0

    compatible, reason = cache_is_compatible(
        cache_csv,
        meta_json,
        args.harmonic,
        args.melodic,
        args.rhythmic,
        args.method,
        args.seed,
        args.perplexity,
        args.tsne_early_exaggeration,
        args.tsne_learning_rate,
        args.tsne_metric,
    )

    if compatible and not args.force:
        print(f"Embedding cache already up-to-date: {cache_csv}")
        return 0
    if not compatible:
        print(f"[info] Rebuilding embedding cache ({reason}).")
    else:
        print("[info] --force specified; rebuilding embedding cache.")

    try:
        cache_df, meta = build_cache(
            args.harmonic,
            args.melodic,
            args.rhythmic,
            args.method,
            args.seed,
            args.perplexity,
            args.tsne_early_exaggeration,
            args.tsne_learning_rate,
            args.tsne_metric,
        )
    except KeyboardInterrupt:
        print("[warn] Interrupted; no cache written.")
        return 130

    _atomic_write_csv(cache_csv, cache_df)
    _atomic_write_text(meta_json, json.dumps(meta, indent=2, sort_keys=True))

    print(f"Wrote embedding cache: {cache_csv} ({len(cache_df)} pieces)")
    print(f"Wrote cache metadata: {meta_json}")
    if args.method == "pca" and "pca" in meta:
        explained = meta["pca"].get("explained_variance_ratio")
        if isinstance(explained, list) and explained:
            pct = [float(v) * 100.0 for v in explained[:3]]
            print("Explained variance (PC1-3): " + ", ".join(f"{v:.1f}%" for v in pct))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
