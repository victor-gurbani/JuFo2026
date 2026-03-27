"""Calculate and report cluster evaluation metrics (Silhouette Score and Davies-Bouldin Index)
for the generated PCA embeddings and the raw 36D feature space."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

from feature_embedding import _load_features, _merge_feature_tables, _prepare_feature_matrix

def compute_metrics(coords: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if len(np.unique(labels)) < 2:
        return {"silhouette_score": 0.0, "davies_bouldin_index": 0.0}
        
    sil = silhouette_score(coords, labels)
    db = davies_bouldin_score(coords, labels)
    return {"silhouette_score": float(sil), "davies_bouldin_index": float(db)}

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate composer clusters mathematically.")
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings/pca_embedding_cache.csv"), help="Path to PCA embeddings cache CSV")
    parser.add_argument("--harmonic", type=Path, default=Path("data/features/harmonic_features.csv"), help="Path to harmonic features")
    parser.add_argument("--melodic", type=Path, default=Path("data/features/melodic_features.csv"), help="Path to melodic features")
    parser.add_argument("--rhythmic", type=Path, default=Path("data/features/rhythmic_features.csv"), help="Path to rhythmic features")
    parser.add_argument("--output", type=Path, default=Path("data/stats/cluster_metrics.json"), help="Output JSON map for metrics")
    args = parser.parse_args()

    print("Evaluating composer clusters...")
    results = {}

    # Load 3D coordinates
    if args.cache.exists():
        df_pca = pd.read_csv(args.cache)
        coords_3d = df_pca[["dim1", "dim2", "dim3"]].values
        labels_pca = df_pca["composer_label"].values
        metrics_3d = compute_metrics(coords_3d, labels_pca)
        print("\n--- 3D PCA Embeddings Space ---")
        print(f"Silhouette Score (higher is better, 1 max, <0 means misclassified): {metrics_3d['silhouette_score']:.4f}")
        print(f"Davies-Bouldin Index (lower is better, 0 min): {metrics_3d['davies_bouldin_index']:.4f}")
        results["pca_3d"] = metrics_3d
    else:
        print(f"[warning] Embedding cache not found at {args.cache}. Run feature_embedding.py first.")

    # Load full dimensional space
    harmonic = _load_features(args.harmonic)
    melodic = _load_features(args.melodic)
    rhythmic = _load_features(args.rhythmic)
    
    combined = _merge_feature_tables(harmonic, melodic, rhythmic)
    matrix, _, _ = _prepare_feature_matrix(combined)
    labels_full = combined["composer_label"].values

    metrics_full = compute_metrics(matrix, labels_full)
    print("\n--- Full 36D Feature Space (Scaled) ---")
    print(f"Silhouette Score (higher is better, 1 max, <0 means misclassified): {metrics_full['silhouette_score']:.4f}")
    print(f"Davies-Bouldin Index (lower is better, 0 min): {metrics_full['davies_bouldin_index']:.4f}")
    results["full_36d"] = metrics_full

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        print(f"\nSaved metrics to {args.output}")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
