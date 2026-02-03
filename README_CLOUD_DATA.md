# Complete Cloud Data Package - Summary

This package contains **all the mathematical formulas and computed data** needed to recreate the 3D embedding clouds in any 3D modeling software.

## Files Included

1. **CLOUD_GENERATION_FORMULAS.md** ← Main document with all formulas and parameters
2. **CLOUD_POINT_COORDINATES.json** ← Individual point coordinates (x, y, z) for all 144 pieces
3. **export_cloud_data.py** ← Python script to regenerate the data

---

## Quick Start: Using in Blender/Maya/C4D

### Step 1: Create the Grid
```
Grid bounds:
  X-axis: [-6.6543, 7.8796]
  Y-axis: [-6.0457, 10.1132]
  Z-axis: [-4.6413, 5.2295]
Resolution: 22 × 22 × 22 = 10,648 points
```

### Step 2: Compute Gaussian Density for Each Composer

For each of 4 composers (Bach, Chopin, Debussy, Mozart):

**Formula:**
```
PDF(x,y,z) = (1 / √((2π)³ × det(Σ))) × exp(-0.5 × (p-μ)ᵀ × Σ⁻¹ × (p-μ))
```

**Parameters by Composer:**

#### Bach (#636efa - Blue)
```
μ = [-0.3743, -1.1647, -0.6326]
Σ = [[4.95930,  1.98650, -0.86586],
     [1.98650,  3.63640,  0.59088],
     [-0.86586, 0.59088,  1.77024]]
ISO = 0.003179
```

#### Chopin (#EF553B - Red)
```
μ = [0.6049, 0.0407, 0.4378]
Σ = [[4.89511, -0.57348, -1.71000],
     [-0.57348, 5.41901,  0.57986],
     [-1.71000, 0.57986,  3.25859]]
ISO = 0.001662
```

#### Debussy (#00cc96 - Green)
```
μ = [2.0773, 0.2903, 0.6443]
Σ = [[4.02019,  0.35539, -0.50847],
     [0.35539,  4.35746, -0.89257],
     [-0.50847, -0.89257, 3.03613]]
ISO = 0.001968
```

#### Mozart (#ab63fa - Purple)
```
μ = [-2.3079, 0.8337, -0.4495]
Σ = [[3.22122, -0.88329,  0.12449],
     [-0.88329, 4.21719, -0.86128],
     [0.12449, -0.86128,  2.82017]]
ISO = 0.002386
```

### Step 3: Extract Isosurfaces

Use marching cubes algorithm with the ISO thresholds above for each composer's PDF grid. This generates 4 triangle meshes.

### Step 4: Apply Materials

```
All meshes:
  Opacity: 0.35
  Caps: Disabled (open surface)
  Lighting:
    Ambient: 0.45
    Diffuse: 0.60
    Specular: 0.25
    Roughness: 0.40
    Fresnel: 0.10
```

---

## Feature Space Axes

The 3D coordinates represent:

- **X-axis (PC1)**: "Chromatik/Dissonanz"
  - Harmonic complexity and dissonance levels
  - **Negative** = consonant (Bach, Mozart)
  - **Positive** = dissonant/chromatic (Debussy)

- **Y-axis (PC2)**: "Dichte/Klarheit"
  - Rhythmic density and harmonic clarity
  - **Negative** = dense (Bach)
  - **Positive** = sparse/clear (Mozart)

- **Z-axis (PC3)**: "Registral/Textur"
  - Pitch range and textural variety
  - **Negative** = compact range (Bach)
  - **Positive** = wide range (Chopin)

**Variance Explained:**
- PC1: 22.3%
- PC2: 16.1%
- PC3: 9.8%
- **Total: 48.2%**

---

## 30 Musical Features Used

The embeddings are computed from these 30 features extracted from sheet music:

**Harmonic Features (12):**
- chord_quality_major_pct, chord_quality_minor_pct
- chord_quality_diminished_pct, chord_quality_augmented_pct
- chord_quality_other_pct, harmonic_density_mean
- dissonance_ratio, passing_tone_ratio
- appoggiatura_ratio, other_dissonance_ratio
- deceptive_cadence_ratio, modal_interchange_ratio

**Melodic Features (10):**
- pitch_range_semitones, avg_melodic_interval
- conjunct_motion_ratio, pitch_class_entropy
- melodic_interval_std, melodic_leap_ratio
- voice_independence_index, contrary_motion_ratio
- parallel_motion_ratio, oblique_motion_ratio

**Rhythmic Features (8):**
- avg_note_duration, std_note_duration
- notes_per_beat, downbeat_emphasis_ratio
- syncopation_ratio, rhythmic_pattern_entropy
- micro_rhythmic_density, cross_rhythm_ratio

---

## Algorithmic Pipeline

```
Sheet Music (MusicXML files)
    ↓
Feature Extraction (30 features per piece)
    ↓
Data Normalization (Z-score standardization)
    ↓
PCA Projection (30D → 3D)
    ↓
Gaussian Kernel Density Estimation (per composer)
    ↓
3D Grid Evaluation (22×22×22 = 10,648 points)
    ↓
Isosurface Extraction (Marching Cubes algorithm)
    ↓
Mesh Visualization (4 translucent surfaces)
```

---

## Implementation Example (Python/NumPy)

```python
import numpy as np
from scipy.special import logsumexp

# For a given composer
mean = np.array([-0.3743, -1.1647, -0.6326])  # Bach
cov = np.array([
    [ 4.95930,  1.98650, -0.86586],
    [ 1.98650,  3.63640,  0.59088],
    [-0.86586,  0.59088,  1.77024]
])

# Grid points (22×22×22)
x = np.linspace(-6.6543, 7.8796, 22)
y = np.linspace(-6.0457, 10.1132, 22)
z = np.linspace(-4.6413, 5.2295, 22)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

# Compute Gaussian PDF
inv_cov = np.linalg.inv(cov)
det_cov = np.linalg.det(cov)
diff = points - mean
exponent = np.einsum('...i,ij,...j->...', diff, inv_cov, diff)
norm_const = np.sqrt((2*np.pi)**3 * det_cov)
pdf = np.exp(-0.5 * exponent) / norm_const

# Reshape to 3D grid
pdf_grid = pdf.reshape((22, 22, 22))

# Extract isosurface at 22% threshold
from skimage import measure
iso_level = 0.22 * pdf_grid.max()
verts, faces, _, _ = measure.marching_cubes(pdf_grid, level=iso_level)

# Scale vertices to world coordinates
verts_world = np.zeros_like(verts)
verts_world[:, 0] = np.interp(verts[:, 0], [0, 21], [-6.6543, 7.8796])
verts_world[:, 1] = np.interp(verts[:, 1], [0, 21], [-6.0457, 10.1132])
verts_world[:, 2] = np.interp(verts[:, 2], [0, 21], [-4.6413, 5.2295])

# Now export verts/faces to OBJ or other format
```

---

## Individual Point Coordinates

See `CLOUD_POINT_COORDINATES.json` for the (x, y, z) coordinates of all 144 pieces:
- 36 Bach pieces
- 36 Chopin pieces
- 36 Debussy pieces
- 36 Mozart pieces

Each point can be visualized as a small sphere in the 3D space.

---

## Verification

To verify your implementation:

1. ✓ Grid resolution: 22×22×22
2. ✓ Grid bounds match coordinate ranges
3. ✓ Covariance matrices are symmetric 3×3
4. ✓ ISO threshold = 0.22 × max(PDF) for each composer
5. ✓ Colors: #636efa (Bach), #EF553B (Chopin), #00cc96 (Debussy), #ab63fa (Mozart)
6. ✓ Opacity: 0.35 for all surfaces
7. ✓ 4 separate isosurfaces (one per composer)

---

## References & Software

**3D Modeling Software Support:**
- **Blender**: Use Python API + marching cubes addon
- **Maya**: MEL script or Python plugin
- **Cinema 4D**: Python script or xpresso
- **Houdini**: VEX expressions or Python

**Marching Cubes Libraries:**
- Python: `scikit-image.measure.marching_cubes`
- C++: `VTK`, `OpenVDB`
- GLSL: Compute shader implementation

**Optional: Load point cloud first**
```python
import json
with open('CLOUD_POINT_COORDINATES.json') as f:
    points = json.load(f)
# Place small spheres at each point coordinate
```

---

## Citation

If using this in publications, cite:
- Original 3D PCA embedding methodology
- Musical feature extraction pipeline
- The four composers dataset (Bach, Chopin, Debussy, Mozart)

---

## Questions?

For reproduction or modification:
1. See `CLOUD_GENERATION_FORMULAS.md` for complete mathematical details
2. Check `export_cloud_data.py` for the Python pipeline
3. Verify coordinates in `CLOUD_POINT_COORDINATES.json`
4. Test isosurface extraction with a subset of the grid first

