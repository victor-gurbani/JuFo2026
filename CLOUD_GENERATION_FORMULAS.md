# 3D Cloud Generation Formulas & Data

This document contains all the mathematical formulas and computed parameters needed to recreate the 3D embedding clouds in any 3D modeling software (Blender, Maya, Cinema4D, etc.).

## Overview

The clouds are generated through 4 steps:
1. **Feature Extraction** → 30 musical features per piece
2. **Dimensionality Reduction** → PCA projects 30D features to 3D
3. **Gaussian Density Estimation** → Kernel Density Estimation per composer
4. **Isosurface Generation** → Extract 22% threshold surface as mesh

---

## Step 1: Data Collection & Features

### Input Data
- **144 musical pieces** total:
  - 36 pieces per composer (Bach, Chopin, Debussy, Mozart)
- **30 numeric features** extracted from sheet music:
  
  **Harmonic Features** (6):
  - chord_quality_major_pct, chord_quality_minor_pct
  - chord_quality_diminished_pct, chord_quality_augmented_pct
  - chord_quality_other_pct, harmonic_density_mean
  - dissonance_ratio, passing_tone_ratio
  - appoggiatura_ratio, other_dissonance_ratio
  - deceptive_cadence_ratio, modal_interchange_ratio
  
  **Melodic Features** (10):
  - pitch_range_semitones, avg_melodic_interval
  - conjunct_motion_ratio, pitch_class_entropy
  - melodic_interval_std, melodic_leap_ratio
  - voice_independence_index, contrary_motion_ratio
  - parallel_motion_ratio, oblique_motion_ratio
  
  **Rhythmic Features** (8):
  - avg_note_duration, std_note_duration
  - notes_per_beat, downbeat_emphasis_ratio
  - syncopation_ratio, rhythmic_pattern_entropy
  - micro_rhythmic_density, cross_rhythm_ratio

---

## Step 2: Standardization (Z-Score Normalization)

Convert each feature to zero mean, unit variance:

$$
\text{standardized}_{ij} = \frac{\text{feature}_{ij} - \mu_j}{\sigma_j}
$$

Where:
- $\mu_j$ = mean of feature $j$
- $\sigma_j$ = std dev of feature $j$
- Result: 144 × 30 matrix with mean ≈ 0, std ≈ 1

---

## Step 3: PCA Projection (30D → 3D)

Principal Component Analysis reduces 30 dimensions to 3:

**PCA Formula:**
$$
\text{PC}_i = \mathbf{X} \cdot \mathbf{w}_i
$$

Where:
- $\mathbf{X}$ = standardized feature matrix (144 × 30)
- $\mathbf{w}_i$ = $i$-th principal component (eigenvector)

**Variance Explained:**
- PC1 (Chromatik/Dissonanz): **22.3%** of total variance
- PC2 (Dichte/Klarheit): **16.1%** of total variance
- PC3 (Registral/Textur): **9.8%** of total variance
- **Total: 48.2%** cumulative

**PCA Coordinate Ranges:**
| Axis | Min | Max | Range |
|------|-----|-----|-------|
| PC1 | -5.4431 | 6.6685 | 12.1116 |
| PC2 | -4.6991 | 8.7667 | 13.4658 |
| PC3 | -3.8187 | 4.4070 | 8.2257 |

---

## Step 4: Grid & Density Estimation

### Grid Definition

Create a 3D regular grid encompassing all points with 10% padding:

**Grid Parameters:**
- Grid points per axis: **22**
- Total grid points: 22³ = **10,648**

**Grid Bounds** (with 10% padding):
- X-axis (PC1): [-6.6543, 7.8796]
- Y-axis (PC2): [-6.0457, 10.1132]
- Z-axis (PC3): [-4.6413, 5.2295]

### Gaussian Kernel Density Estimation (KDE)

For each composer group, compute a 3D Gaussian PDF at every grid point:

$$
\text{PDF}(\mathbf{p}) = \frac{1}{\sqrt{(2\pi)^3 \cdot \det(\Sigma)}} \exp\left(-\frac{1}{2}(\mathbf{p} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{p} - \boldsymbol{\mu})\right)
$$

Where:
- $\mathbf{p} = (x, y, z)$ = grid point
- $\boldsymbol{\mu}$ = mean (centroid) of composer's pieces
- $\Sigma$ = covariance matrix of composer's pieces
- $\det(\Sigma)$ = determinant of covariance matrix
- $\Sigma^{-1}$ = inverse of covariance matrix
- $(2\pi)^3 \approx 248.05$ = normalization constant

---

## Step 5: Composer Cloud Data

### Bach Cloud (Color: #636efa - Blue)

**Data:**
- Pieces: 36
- Centroid: $\boldsymbol{\mu} = [-0.3743, -1.1647, -0.6326]$

**Covariance Matrix $\Sigma$:**
```
[  4.95930   1.98650  -0.86586 ]
[  1.98650   3.63640   0.59088 ]
[ -0.86586   0.59088   1.77024 ]
```

**Properties:**
- det($\Sigma$) = 18.448
- max(PDF) = 0.014451
- **ISO threshold = 0.003179** (22% of max)

### Chopin Cloud (Color: #EF553B - Red)

**Data:**
- Pieces: 36
- Centroid: $\boldsymbol{\mu} = [0.6049, 0.0407, 0.4378]$

**Covariance Matrix $\Sigma$:**
```
[  4.89511  -0.57348  -1.71000 ]
[ -0.57348   5.41901   0.57986 ]
[ -1.71000   0.57986   3.25859 ]
```

**Properties:**
- det($\Sigma$) = 69.013
- max(PDF) = 0.007552
- **ISO threshold = 0.001662** (22% of max)

### Debussy Cloud (Color: #00cc96 - Green)

**Data:**
- Pieces: 36
- Centroid: $\boldsymbol{\mu} = [2.0773, 0.2903, 0.6443]$

**Covariance Matrix $\Sigma$:**
```
[  4.02019   0.35539  -0.50847 ]
[  0.35539   4.35746  -0.89257 ]
[ -0.50847  -0.89257   3.03613 ]
```

**Properties:**
- det($\Sigma$) = 48.796
- max(PDF) = 0.008943
- **ISO threshold = 0.001968** (22% of max)

### Mozart Cloud (Color: #ab63fa - Purple)

**Data:**
- Pieces: 36
- Centroid: $\boldsymbol{\mu} = [-2.3079, 0.8337, -0.4495]$

**Covariance Matrix $\Sigma$:**
```
[  3.22122  -0.88329   0.12449 ]
[ -0.88329   4.21719  -0.86128 ]
[  0.12449  -0.86128   2.82017 ]
```

**Properties:**
- det($\Sigma$) = 33.845
- max(PDF) = 0.010846
- **ISO threshold = 0.002386** (22% of max)

---

## Step 6: Isosurface Extraction

### Isosurface Parameters

For each composer:

1. Compute PDF values on the 22×22×22 grid
2. Extract isosurface at **ISO_THRESHOLD = 0.22 × max(PDF)**
3. Generate mesh with surface normals

**Marching Cubes Algorithm:**
- Algorithm: Standard marching cubes
- Isovalue: See thresholds above (per composer)
- No caps on isosurface boundaries
- Output: Triangle mesh

---

## Step 7: Mesh Rendering Properties

### Lighting & Materials

| Property | Value |
|----------|-------|
| Opacity | 0.35 (translucent) |
| Ambient lighting | 0.45 |
| Diffuse lighting | 0.60 |
| Specular highlight | 0.25 |
| Surface roughness | 0.40 |
| Fresnel effect | 0.10 |
| Light position | (80, 100, 60) |

### Colors (Plotly Standard Palette)

| Composer | Color Code | RGB |
|----------|-----------|-----|
| Bach | #636efa | (99, 110, 250) |
| Chopin | #EF553B | (239, 85, 59) |
| Debussy | #00cc96 | (0, 204, 150) |
| Mozart | #ab63fa | (171, 99, 250) |

---

## Implementation Guide

### For 3D Modeling Software (Blender/Maya/C4D):

1. **Create Grid:**
   - Use plugin or script to create 22×22×22 lattice
   - Apply transformations to map to coordinate ranges above

2. **Compute PDF:**
   - For each grid cell, evaluate the Gaussian PDF formula
   - Store as scalar field

3. **Extract Isosurface:**
   - Use marching cubes algorithm
   - Use ISO_THRESHOLD for each composer separately
   - Generate separate mesh per composer

4. **Apply Materials:**
   - Assign colors from palette above
   - Set opacity to 0.35
   - Apply lighting parameters

5. **Combine Visualization:**
   - Place all 4 isosurfaces in same 3D scene
   - Optional: Add point cloud (36 points per composer) at original coordinates

### Python/NumPy Reference Code:

```python
import numpy as np
from scipy.spatial.distance import mahalanobis

def compute_pdf_grid(grid_points, mean, cov, grid_shape):
    """Evaluate 3D Gaussian at grid points"""
    diff = grid_points - mean
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    
    exponent = np.einsum('...i,ij,...j->...', diff, inv_cov, diff)
    norm_const = np.sqrt((2 * np.pi)**3 * det_cov)
    
    pdf = np.exp(-0.5 * exponent) / norm_const
    return pdf.reshape(grid_shape)

# For each composer, call marching_cubes on PDF grid with appropriate threshold
```

---

## Axis Interpretations

Based on the PCA loadings:

- **PC1 (Chromatik/Dissonanz)**: 
  - Captures harmonic complexity and dissonance levels
  - Bach (negative) = more consonant, Mozart (negative) = consonant
  - Debussy (positive) = more dissonant, chromatic harmony

- **PC2 (Dichte/Klarheit)**:
  - Captures density and rhythmic clarity
  - Bach (negative) = dense, Mozart (positive) = sparse/clear
  
- **PC3 (Registral/Textur)**:
  - Captures pitch range and textural variety
  - Bach (negative) = compact range, Chopin (positive) = wider range

---

## Verification Checklist

When implementing in your modeling software:

- [ ] Grid bounds match coordinate ranges above
- [ ] Grid resolution is 22×22×22 (10,648 points)
- [ ] Covariance matrices are symmetric 3×3 (as shown)
- [ ] Isosurface threshold is 22% of maximum PDF value
- [ ] Colors match Plotly palette (#636efa, #EF553B, #00cc96, #ab63fa)
- [ ] Opacity set to 0.35
- [ ] 4 separate isosurfaces (one per composer)

---

## Additional Resources

- **PCA Components**: Available in sklearn from fitted PCA model
- **Feature CSVs**: Located in `data/features/` directory
  - harmonic_features.csv (144 × 19)
  - melodic_features.csv (144 × 14)
  - rhythmic_features.csv (144 × 12)

