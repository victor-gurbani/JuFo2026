#!/usr/bin/env python
"""Export cloud point coordinates and parameters for 3D modeling."""

import numpy as np
import pandas as pd
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Load feature tables
harmonic = pd.read_csv("data/features/harmonic_features.csv")
melodic = pd.read_csv("data/features/melodic_features.csv")
rhythmic = pd.read_csv("data/features/rhythmic_features.csv")

# Merge all features
key_cols = ["composer_label", "title", "mxl_path"]
combined = (
    harmonic.set_index(key_cols)
    .join(melodic.set_index(key_cols), how="inner", rsuffix="_mel")
    .join(rhythmic.set_index(key_cols), how="inner", rsuffix="_rhy")
    .reset_index()
)

# Prepare feature matrix
EXCLUDED_FEATURES = {
    "note_count",
    "note_event_count",
    "chord_event_count",
    "chord_quality_total",
    "roman_chord_count",
    "dissonant_note_count",
}

numeric_cols = [col for col in combined.columns if pd.api.types.is_numeric_dtype(combined[col])]
numeric_cols = [col for col in numeric_cols if col not in {"mxl_path"}]
numeric_cols = [col for col in numeric_cols if col not in EXCLUDED_FEATURES]

filled = combined[numeric_cols].copy()
filled = filled.fillna(filled.mean())
scaler = StandardScaler()
matrix = scaler.fit_transform(filled.values)

# PCA projection
pca = PCA(n_components=3, random_state=42)
coords = pca.fit_transform(matrix)

# Export point coordinates per composer
output = {}
for composer in sorted(combined["composer_label"].unique()):
    mask = combined["composer_label"] == composer
    comp_coords = coords[mask]
    comp_data = combined.loc[mask]
    
    points = []
    for i, (idx, row) in enumerate(comp_data.iterrows()):
        points.append({
            "title": row["title"],
            "x": float(comp_coords[i, 0]),
            "y": float(comp_coords[i, 1]),
            "z": float(comp_coords[i, 2]),
        })
    output[composer] = points

# Save JSON
with open("CLOUD_POINT_COORDINATES.json", "w") as f:
    json.dump(output, f, indent=2)

print("✓ Saved point coordinates to CLOUD_POINT_COORDINATES.json")
print(f"  Total: {sum(len(v) for v in output.values())} points")
for composer, points in output.items():
    print(f"    {composer}: {len(points)} points")
