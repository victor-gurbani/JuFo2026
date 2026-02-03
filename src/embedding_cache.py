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
import multiprocessing as mp
from multiprocessing.pool import Pool as MpPool
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

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

FEATURE_CACHE_ID_COLUMNS = (
    "composer_label",
    "title",
    "mxl_path",
    "mxl_abs_path",
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


def _coerce_label(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text if text else fallback


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _format_progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + ("#" * width) + "]"
    frac = min(max(done / total, 0.0), 1.0)
    filled = int(round(frac * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


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


def _sanitize_metric_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip()
    if not text:
        return ""
    try:
        return float(text)
    except Exception:
        return text


def _derive_feature_schema_from_csvs(harmonic_path: Path, melodic_path: Path, rhythmic_path: Path) -> List[str]:
    """Return the ordered list of feature columns (excluding identifiers) used by this project.

    This uses the headers of the existing feature CSVs, so the full-corpus feature cache
    matches the curated pipeline's feature naming.
    """

    def _feature_cols(path: Path) -> List[str]:
        if not path.exists():
            raise FileNotFoundError(f"Schema source features file not found: {path}")
        header = pd.read_csv(path, nrows=0)
        cols = [c for c in header.columns if c not in {"composer_label", "title", "mxl_path", "mxl_abs_path"}]
        return cols

    schema: List[str] = []
    for part_cols in (_feature_cols(harmonic_path), _feature_cols(melodic_path), _feature_cols(rhythmic_path)):
        for col in part_cols:
            if col not in schema:
                schema.append(col)
    if not schema:
        raise ValueError("Derived empty feature schema from feature CSV headers.")
    return schema


def _append_feature_row(
    features_csv: Path,
    fieldnames: List[str],
    row: Dict[str, object],
) -> None:
    features_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = features_csv.exists() and features_csv.stat().st_size > 0
    with features_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        handle.flush()


def _worker_compute_features(
    task: Tuple[str, str, str],
) -> Tuple[str, str, str, Optional[Dict[str, object]], Optional[str]]:
    """Worker: compute full feature mapping for one piece.

    Must be top-level so it is picklable for multiprocessing on macOS (spawn).
    Returns (composer, title, abs_path_str, feature_mapping, error).
    """

    composer, title, abs_path_str = task
    try:
        score = parse_score(Path(abs_path_str))
        harmonic_metrics = compute_harmonic_features(score)
        melodic_metrics = compute_melodic_features(score)
        rhythmic_metrics = compute_rhythmic_features(score)
        feature_mapping: Dict[str, object] = {
            **harmonic_metrics,
            **melodic_metrics,
            **rhythmic_metrics,
        }
        return composer, title, abs_path_str, feature_mapping, None
    except Exception as exc:
        return composer, title, abs_path_str, None, str(exc)


def cache_corpus_long_running(
    *,
    model_cache_csv: Optional[Path],
    output_embedding_csv: Optional[Path],
    output_features_csv: Optional[Path],
    corpus_csv: Optional[Path],
    paths_file: Optional[Path],
    dataset_root: Optional[Path],
    schema_harmonic: Path,
    schema_melodic: Path,
    schema_rhythmic: Path,
    resume: bool,
    force: bool,
    limit: Optional[int],
    skip_errors: bool,
    checkpoint_every: int,
    progress_every: int,
    workers: int,
    chunksize: int,
) -> int:
    """Long-running caching over arbitrary corpora.

    Can write:
    - A projection cache (dim1..dim3) in an existing PCA space (requires model_cache_csv).
    - A full feature cache (all ~36 features) suitable for refitting PCA on any subset.

    Designed to run for days: appends row-by-row and uses done-files to resume.
    """

    if output_embedding_csv is None and output_features_csv is None:
        raise ValueError("Nothing to write: provide --output-csv and/or --output-features-csv.")

    model_meta: Optional[Dict[str, Any]] = None
    if output_embedding_csv is not None:
        if model_cache_csv is None:
            raise ValueError("--model-cache is required when writing an embedding/projection cache.")
        if not model_cache_csv.exists():
            raise FileNotFoundError(f"Model cache CSV not found: {model_cache_csv}")
        model_meta = load_meta(model_cache_csv)

    repo_root = Path(__file__).resolve().parent.parent
    dataset_root_resolved = dataset_root.resolve() if dataset_root is not None else None

    # Prepare feature schema early (used for full-feature caching).
    feature_schema: Optional[List[str]] = None
    if output_features_csv is not None:
        feature_schema = _derive_feature_schema_from_csvs(schema_harmonic, schema_melodic, schema_rhythmic)

    # Output init / overwrite behavior.
    if output_embedding_csv is not None and output_embedding_csv.exists() and not resume and not force:
        raise FileExistsError(
            f"Output embedding cache already exists: {output_embedding_csv}. Use --resume to continue or --force to overwrite."
        )
    if output_features_csv is not None and output_features_csv.exists() and not resume and not force:
        raise FileExistsError(
            f"Output features cache already exists: {output_features_csv}. Use --resume to continue or --force to overwrite."
        )

    if force and not resume:
        if output_embedding_csv is not None and output_embedding_csv.exists():
            output_embedding_csv.unlink()
            meta_path_for(output_embedding_csv).unlink(missing_ok=True)  # type: ignore[arg-type]
            done_path_for(output_embedding_csv).unlink(missing_ok=True)  # type: ignore[arg-type]
        if output_features_csv is not None and output_features_csv.exists():
            output_features_csv.unlink()
            meta_path_for(output_features_csv).unlink(missing_ok=True)  # type: ignore[arg-type]
            done_path_for(output_features_csv).unlink(missing_ok=True)  # type: ignore[arg-type]

    done_embeddings: set[str] = set()
    done_features: set[str] = set()

    if output_embedding_csv is not None:
        done_path_embed = done_path_for(output_embedding_csv)
        done_embeddings = _load_done_set(done_path_embed)
        if not done_embeddings and output_embedding_csv.exists():
            done_embeddings = _scan_existing_cache_for_done(output_embedding_csv)
            if done_embeddings:
                done_path_embed.parent.mkdir(parents=True, exist_ok=True)
                done_path_embed.write_text("\n".join(sorted(done_embeddings)) + "\n", encoding="utf-8")

        # Ensure meta exists early so other consumers know which PCA space this cache is in.
        meta_out = dict(model_meta or {})
        meta_out["cache_version"] = CACHE_VERSION
        meta_out["created_at"] = datetime.now(timezone.utc).isoformat()
        meta_out["derived_from"] = {
            "model_cache_csv": str(model_cache_csv) if model_cache_csv else None,
            "model_cache_meta": str(meta_path_for(model_cache_csv)) if model_cache_csv else None,
        }
        meta_out["in_progress"] = True
        meta_out["piece_count"] = int(len(done_embeddings))
        _atomic_write_text(meta_path_for(output_embedding_csv), json.dumps(meta_out, indent=2, sort_keys=True))

    if output_features_csv is not None:
        done_path_feat = done_path_for(output_features_csv)
        done_features = _load_done_set(done_path_feat)
        if not done_features and output_features_csv.exists():
            # For features cache, reconstruct using mxl_abs_path too.
            done_features = _scan_existing_cache_for_done(output_features_csv)
            if done_features:
                done_path_feat.parent.mkdir(parents=True, exist_ok=True)
                done_path_feat.write_text("\n".join(sorted(done_features)) + "\n", encoding="utf-8")

        meta_feat: Dict[str, Any] = {
            "cache_version": CACHE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "kind": "full_feature_cache",
            "schema_source": {
                "harmonic": str(schema_harmonic),
                "melodic": str(schema_melodic),
                "rhythmic": str(schema_rhythmic),
            },
            "feature_columns": feature_schema,
            "in_progress": True,
            "piece_count": int(len(done_features)),
        }
        _atomic_write_text(meta_path_for(output_features_csv), json.dumps(meta_feat, indent=2, sort_keys=True))

    scanned = 0
    total_records: Optional[int] = None
    work_items = 0
    written_since_checkpoint = 0
    last_written: Optional[str] = None
    last_written_composer: Optional[str] = None
    last_written_title: Optional[str] = None

    start_time = time.monotonic()

    def _maybe_print_progress(force: bool = False) -> None:
        if total_records is None or total_records <= 0:
            return
        if not force and (progress_every <= 0 or scanned % progress_every != 0):
            return
        elapsed = max(time.monotonic() - start_time, 1e-9)
        rate = scanned / elapsed
        remaining = max(total_records - scanned, 0)
        eta = remaining / rate if rate > 0 else float("nan")
        bar = _format_progress_bar(scanned, total_records)
        pct = (scanned / total_records) * 100.0
        embed_count = len(done_embeddings) if output_embedding_csv is not None else 0
        feat_count = len(done_features) if output_features_csv is not None else 0
        print(
            f"[progress] {bar} {scanned}/{total_records} ({pct:.1f}%) ETA {_format_eta(eta)} "
            f"| work={work_items} embeddings={embed_count} features={feat_count}",
            flush=True,
        )

    done_handle_embed = None
    done_handle_feat = None
    def _build_task_list() -> Tuple[List[Tuple[str, str, str]], int]:
        """Return (tasks, scanned_total).

        Each task is (composer, title, abs_path_str) for items that still need work.
        """

        tasks: List[Tuple[str, str, str]] = []

        if corpus_csv is not None:
            if not corpus_csv.exists():
                raise FileNotFoundError(f"Corpus CSV not found: {corpus_csv}")
            df = pd.read_csv(corpus_csv)
            scanned_total = int(len(df))

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
                resolved = _resolve_musicxml_path(raw_path, repo_root=repo_root, dataset_root=dataset_root_resolved)
                if resolved is None:
                    continue
                mxl_abs = normalize_path(resolved)
                needs_embedding = output_embedding_csv is not None and mxl_abs not in done_embeddings
                needs_features = output_features_csv is not None and mxl_abs not in done_features
                if not needs_embedding and not needs_features:
                    continue
                composer = _coerce_label(row.get(composer_col) if composer_col else None, "Unknown")
                title = _coerce_label(row.get(title_col) if title_col else None, resolved.stem)
                tasks.append((composer, title, str(resolved)))
                if limit is not None and len(tasks) >= limit:
                    break
            return tasks, scanned_total

        if paths_file is not None:
            if not paths_file.exists():
                raise FileNotFoundError(f"Paths file not found: {paths_file}")
            lines = [line.strip() for line in paths_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            scanned_total = int(len(lines))
            for raw in lines:
                resolved = _resolve_musicxml_path(raw, repo_root=repo_root, dataset_root=dataset_root_resolved)
                if resolved is None:
                    continue
                mxl_abs = normalize_path(resolved)
                needs_embedding = output_embedding_csv is not None and mxl_abs not in done_embeddings
                needs_features = output_features_csv is not None and mxl_abs not in done_features
                if not needs_embedding and not needs_features:
                    continue
                tasks.append(("Unknown", resolved.stem, str(resolved)))
                if limit is not None and len(tasks) >= limit:
                    break
            return tasks, scanned_total

        raise ValueError("Provide either --corpus-csv or --paths to define which pieces to cache.")

    try:
        if output_embedding_csv is not None:
            done_handle_embed = done_path_for(output_embedding_csv).open("a", encoding="utf-8")
        if output_features_csv is not None:
            done_handle_feat = done_path_for(output_features_csv).open("a", encoding="utf-8")

        tasks, scanned_total = _build_task_list()
        scanned = int(scanned_total)
        total_records = int(scanned_total)
        total_work = int(len(tasks))
        if total_records > 0 and total_work > 0:
            print(f"[info] Corpus records={total_records}, pending work items={total_work}")

        # Override progress display to track completed work items (not raw corpus rows).
        completed = 0

        def _maybe_print_work_progress(force: bool = False) -> None:
            if total_work <= 0:
                return
            if not force and (progress_every <= 0 or completed % progress_every != 0):
                return
            elapsed = max(time.monotonic() - start_time, 1e-9)
            rate = completed / elapsed
            remaining = max(total_work - completed, 0)
            eta = remaining / rate if rate > 0 else float("nan")
            bar = _format_progress_bar(completed, total_work)
            pct = (completed / total_work) * 100.0
            embed_count = len(done_embeddings) if output_embedding_csv is not None else 0
            feat_count = len(done_features) if output_features_csv is not None else 0
            print(
                f"[progress] {bar} {completed}/{total_work} ({pct:.1f}%) ETA {_format_eta(eta)} "
                f"| embeddings={embed_count} features={feat_count}",
                flush=True,
            )

        _maybe_print_work_progress(force=True)

        # Ensure workers is sane.
        workers_eff = int(max(1, workers))
        chunksize_eff = int(max(1, chunksize))

        iterator: Iterable[Tuple[str, str, str, Optional[Dict[str, object]], Optional[str]]]
        pool: Optional[MpPool] = None
        if workers_eff <= 1:
            iterator = (_worker_compute_features(task) for task in tasks)
        else:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(processes=workers_eff)
            assert pool is not None
            iterator = pool.imap_unordered(_worker_compute_features, tasks, chunksize=chunksize_eff)

        try:
            for composer, title, abs_path_str, feature_mapping, error in iterator:
                completed += 1
                _maybe_print_work_progress()

                mxl_abs = normalize_path(abs_path_str)
                needs_embedding = output_embedding_csv is not None and mxl_abs not in done_embeddings
                needs_features = output_features_csv is not None and mxl_abs not in done_features
                if not needs_embedding and not needs_features:
                    continue

                if error is not None or feature_mapping is None:
                    if skip_errors:
                        print(f"[warn] Skipping {abs_path_str} due to error: {error}", flush=True)
                        continue
                    raise RuntimeError(f"Feature extraction failed for {abs_path_str}: {error}")

                coords = None
                if needs_embedding:
                    if model_meta is None:
                        raise RuntimeError("Internal error: model_meta missing while embedding requested.")
                    coords = project_features_into_cached_pca(feature_mapping, model_meta)

                if needs_embedding and output_embedding_csv is not None and coords is not None:
                    row_embed = {
                        "composer_label": composer,
                        "title": title,
                        "mxl_path": mxl_abs,
                        "mxl_abs_path": mxl_abs,
                        "dim1": float(coords[0]),
                        "dim2": float(coords[1]),
                        "dim3": float(coords[2]),
                    }
                    _append_embedding_row(output_embedding_csv, row_embed)
                    done_embeddings.add(mxl_abs)
                    if done_handle_embed is not None:
                        done_handle_embed.write(mxl_abs + "\n")
                        done_handle_embed.flush()
                    last_written = mxl_abs
                    last_written_composer = composer
                    last_written_title = title

                if needs_features and output_features_csv is not None and feature_schema is not None:
                    fieldnames = list(FEATURE_CACHE_ID_COLUMNS) + list(feature_schema)
                    row_feat: Dict[str, object] = {
                        "composer_label": composer,
                        "title": title,
                        "mxl_path": mxl_abs,
                        "mxl_abs_path": mxl_abs,
                    }
                    for col in feature_schema:
                        row_feat[col] = _sanitize_metric_value(feature_mapping.get(col))
                    _append_feature_row(output_features_csv, fieldnames, row_feat)
                    done_features.add(mxl_abs)
                    if done_handle_feat is not None:
                        done_handle_feat.write(mxl_abs + "\n")
                        done_handle_feat.flush()
                    last_written = mxl_abs
                    last_written_composer = composer
                    last_written_title = title

                written_since_checkpoint += 1

                if checkpoint_every > 0 and written_since_checkpoint >= checkpoint_every:
                    now = datetime.now(timezone.utc).isoformat()
                    if output_embedding_csv is not None:
                        meta_embed = load_meta(output_embedding_csv)
                        meta_embed["piece_count"] = int(len(done_embeddings))
                        meta_embed["last_checkpoint_at"] = now
                        _atomic_write_text(meta_path_for(output_embedding_csv), json.dumps(meta_embed, indent=2, sort_keys=True))
                    if output_features_csv is not None:
                        meta_feat = json.loads(meta_path_for(output_features_csv).read_text(encoding="utf-8"))
                        meta_feat["piece_count"] = int(len(done_features))
                        meta_feat["last_checkpoint_at"] = now
                        _atomic_write_text(meta_path_for(output_features_csv), json.dumps(meta_feat, indent=2, sort_keys=True))
                    embed_count = len(done_embeddings) if output_embedding_csv is not None else 0
                    feat_count = len(done_features) if output_features_csv is not None else 0
                    if last_written is not None:
                        hint = f"last={last_written_composer or 'Unknown'} — {last_written_title or ''}".strip()
                        print(f"[checkpoint] embeddings={embed_count} features={feat_count} {hint}", flush=True)
                    else:
                        print(f"[checkpoint] embeddings={embed_count} features={feat_count}", flush=True)
                    written_since_checkpoint = 0
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        _maybe_print_work_progress(force=True)

    finally:
        try:
            if done_handle_embed is not None:
                done_handle_embed.close()
        except Exception:
            pass
        try:
            if done_handle_feat is not None:
                done_handle_feat.close()
        except Exception:
            pass

    if output_embedding_csv is not None:
        meta_embed = load_meta(output_embedding_csv)
        meta_embed["in_progress"] = False
        meta_embed["piece_count"] = int(len(done_embeddings))
        try:
            meta_embed["piece_id_digest"] = _piece_id_digest(done_embeddings)
        except Exception:
            pass
        _atomic_write_text(meta_path_for(output_embedding_csv), json.dumps(meta_embed, indent=2, sort_keys=True))

    if output_features_csv is not None:
        meta_feat = json.loads(meta_path_for(output_features_csv).read_text(encoding="utf-8"))
        meta_feat["in_progress"] = False
        meta_feat["piece_count"] = int(len(done_features))
        try:
            meta_feat["piece_id_digest"] = _piece_id_digest(done_features)
        except Exception:
            pass
        _atomic_write_text(meta_path_for(output_features_csv), json.dumps(meta_feat, indent=2, sort_keys=True))

    # Return count of the largest cache produced (useful progress signal).
    return int(max(len(done_embeddings), len(done_features)))


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
        "--output-features-csv",
        type=Path,
        default=None,
        help=(
            "Optional destination CSV to cache the full feature vector for each processed piece "
            "(derived from harmonic/melodic/rhythmic feature CSV headers)."
        ),
    )
    parser.add_argument(
        "--no-embedding-cache",
        action="store_true",
        help="In --project-only mode, do not write dim1..dim3 embeddings to --output-csv (features-only caching).",
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
        "--progress-every",
        type=int,
        default=50,
        help=(
            "Print a short progress line every N new work items (0 disables). "
            "Useful for long-running jobs so terminals don't appear idle."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes for --project-only feature extraction (default: 1). "
            "Set to >1 to use multiple CPU cores."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help=(
            "Multiprocessing task chunksize for --workers>1 in --project-only mode (default: 8). "
            "Higher can reduce overhead; lower can improve load balancing."
        ),
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
            write_embedding = not bool(args.no_embedding_cache)
            embedding_out = cache_csv if write_embedding else None
            model_cache = args.model_cache if write_embedding else None
            count = cache_corpus_long_running(
                model_cache_csv=model_cache,
                output_embedding_csv=embedding_out,
                output_features_csv=args.output_features_csv,
                corpus_csv=args.corpus_csv,
                paths_file=args.paths,
                dataset_root=args.dataset_root,
                schema_harmonic=args.harmonic,
                schema_melodic=args.melodic,
                schema_rhythmic=args.rhythmic,
                resume=args.resume,
                force=args.force,
                limit=args.limit,
                skip_errors=not args.no_skip_errors,
                checkpoint_every=int(args.checkpoint_every),
                progress_every=int(args.progress_every),
                workers=int(args.workers),
                chunksize=int(args.chunksize),
            )
        except KeyboardInterrupt:
            print("[warn] Interrupted; progress preserved. Re-run with --resume to continue.")
            return 130
        targets: List[str] = []
        if not args.no_embedding_cache:
            targets.append(str(cache_csv))
        if args.output_features_csv is not None:
            targets.append(str(args.output_features_csv))
        targets_str = ", ".join(targets) if targets else "(none)"
        print(f"Long-running cache updated: {targets_str} ({count} pieces)")
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
