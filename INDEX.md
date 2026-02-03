# 3D Embedding Cloud Data - Complete Package Index

## Overview

This package contains **complete mathematical formulas and computed parameters** needed to recreate the 3D embedding clouds in any 3D modeling software (Blender, Maya, Cinema 4D, Houdini, etc.).

The clouds visualize 144 classical music pieces (36 each from Bach, Chopin, Debussy, Mozart) projected from 30-dimensional feature space into 3D using PCA, then rendered as translucent Gaussian density surfaces using marching cubes isosurface extraction.

---

## Main Documents

### 1. **CLOUD_GENERATION_FORMULAS.md** ⭐ START HERE
**Purpose:** Complete mathematical reference  
**Content:**
- Step-by-step algorithm walkthrough
- All 4 composers' covariance matrices
- Grid parameters and bounds
- PDF formula with LaTeX notation
- Rendering settings and colors
- Verification checklist

**Use when:** You need detailed math and full documentation

---

### 2. **QUICK_REFERENCE.md** 🚀 BEST FOR IMPLEMENTATION
**Purpose:** Fast lookup tables and code snippets  
**Content:**
- Parameter matrices (all composers at a glance)
- Covariance matrices in compact format
- Grid and isosurface parameters
- Python code example
- Implementation checklist

**Use when:** You're implementing and need quick lookups

---

### 3. **README_CLOUD_DATA.md** 📖 OVERVIEW & GUIDANCE
**Purpose:** Implementation guide and summary  
**Content:**
- Quick start guide
- Step-by-step instructions for 3D software
- Feature space axis interpretations
- All 30 musical features list
- Example implementation code
- Support for various 3D software

**Use when:** You're planning implementation and need guidance

---

## Data Files

### 4. **CLOUD_POINT_COORDINATES.json** 📊 POINT CLOUD DATA
**Purpose:** Individual coordinates for all 144 pieces  
**Format:** JSON with structure:
```json
{
  "Bach": [
    {"title": "...", "x": -2.67, "y": -2.57, "z": 0.07},
    ...
  ],
  "Chopin": [...],
  "Debussy": [...],
  "Mozart": [...]
}
```
**Use when:** You want to visualize individual points or verify coordinates

---

### 5. **export_cloud_data.py** 🐍 REPRODUCIBLE SCRIPT
**Purpose:** Python script to regenerate all data  
**Features:**
- Loads feature CSV files
- Performs PCA projection
- Exports point coordinates
- Can be modified to change parameters

**Use when:** You need to reproduce results or understand the pipeline

---

## Quick Access by Task

### "I want to implement this in 3D software"
1. Read: **README_CLOUD_DATA.md** (overview)
2. Reference: **QUICK_REFERENCE.md** (parameters)
3. Implement: Follow step-by-step guide

### "I need the mathematical details"
1. Read: **CLOUD_GENERATION_FORMULAS.md** (full docs)
2. Reference: **QUICK_REFERENCE.md** (summary tables)

### "I want to verify the coordinates"
1. Check: **CLOUD_POINT_COORDINATES.json** (all 144 points)
2. Compare: Run **export_cloud_data.py** to regenerate

### "I want to understand the pipeline"
1. Read: **CLOUD_GENERATION_FORMULAS.md** (algorithm steps)
2. Study: **export_cloud_data.py** (Python implementation)

---

## Key Parameters Summary

### Grid
- **Resolution:** 22 × 22 × 22 = 10,648 points
- **X-axis bounds:** [-6.6543, 7.8796]
- **Y-axis bounds:** [-6.0457, 10.1132]
- **Z-axis bounds:** [-4.6413, 5.2295]
- **Padding:** 10% of data range

### Isosurface
- **Method:** Marching Cubes
- **Threshold:** 22% of maximum PDF per composer
- **Rendering:** Separate mesh per composer

### Composers (μ, det(Σ), ISO threshold)
- **Bach:** [-0.3743, -1.1647, -0.6326], 18.448, 0.003179
- **Chopin:** [0.6049, 0.0407, 0.4378], 69.013, 0.001662
- **Debussy:** [2.0773, 0.2903, 0.6443], 48.796, 0.001968
- **Mozart:** [-2.3079, 0.8337, -0.4495], 33.845, 0.002386

### Colors
- **Bach:** #636efa (Blue)
- **Chopin:** #EF553B (Red)
- **Debussy:** #00cc96 (Green)
- **Mozart:** #ab63fa (Purple)

### Rendering
- **Opacity:** 0.35
- **Ambient lighting:** 0.45
- **Diffuse lighting:** 0.60
- **Specular highlight:** 0.25

---

## How to Use This Package

### Step 1: Choose Your Reference
- For **implementation:** Use QUICK_REFERENCE.md
- For **math:** Use CLOUD_GENERATION_FORMULAS.md
- For **guidance:** Use README_CLOUD_DATA.md

### Step 2: Understand the Algorithm
```
Sheet Music → Feature Extraction → Standardization 
→ PCA (30D to 3D) → Gaussian PDF Evaluation 
→ Marching Cubes Isosurface → Rendered Mesh
```

### Step 3: Gather Parameters
- Grid bounds and resolution
- 4 centroid vectors (μ)
- 4 covariance matrices (Σ)
- 4 isosurface thresholds
- Colors and rendering settings

### Step 4: Implement
- Create 22×22×22 grid
- Evaluate Gaussian PDF at each point
- Extract isosurface
- Apply materials and colors

### Step 5: Verify
- Check grid resolution and bounds
- Verify covariance matrices are symmetric
- Confirm color assignments
- Test with small subset first

---

## File Statistics

| File | Size | Lines | Purpose |
|------|------|-------|---------|
| CLOUD_GENERATION_FORMULAS.md | 8.5 KB | 329 | Complete math reference |
| README_CLOUD_DATA.md | 7.0 KB | 280 | Implementation guide |
| QUICK_REFERENCE.md | 5.4 KB | 206 | Fast lookup tables |
| CLOUD_POINT_COORDINATES.json | 24 KB | 873 | Point coordinates |
| export_cloud_data.py | 2.2 KB | 72 | Reproducible script |

---

## Technical Details

### Data Source
- **Dataset:** 144 classical music pieces
- **Composers:** Bach, Chopin, Debussy, Mozart (36 each)
- **Features:** 30 extracted from sheet music
  - 12 harmonic (dissonance, chord quality, etc.)
  - 10 melodic (intervals, ranges, motion, etc.)
  - 8 rhythmic (duration, syncopation, etc.)

### Dimensionality Reduction
- **Method:** Principal Component Analysis (PCA)
- **Input:** 30 dimensions
- **Output:** 3 dimensions
- **Variance explained:** 48.2%

### Density Estimation
- **Method:** Gaussian Kernel Density Estimation
- **Kernel:** 3D Gaussian
- **Per-group:** One PDF per composer

### Mesh Extraction
- **Algorithm:** Marching Cubes
- **Grid resolution:** 22³ = 10,648 points
- **Threshold:** 22% of local maximum

---

## Feature Space Interpretation

### PC1: "Chromatik/Dissonanz" (X-axis)
- Captures harmonic complexity and dissonance
- **Negative:** Bach, Mozart (consonant, traditional)
- **Positive:** Debussy (dissonant, chromatic)

### PC2: "Dichte/Klarheit" (Y-axis)
- Captures rhythmic density and clarity
- **Negative:** Bach (dense, complex)
- **Positive:** Mozart (sparse, clear)

### PC3: "Registral/Textur" (Z-axis)
- Captures pitch range and textural variety
- **Negative:** Bach (compact ranges)
- **Positive:** Chopin (wide ranges)

---

## Software Compatibility

### Recommended 3D Software
- **Blender** ✓ (Python API, marching cubes addon)
- **Houdini** ✓ (VEX, Python, native isosurface ops)
- **Maya** ✓ (MEL, Python plugins)
- **Cinema 4D** ✓ (Python, xpresso)
- **RealityCapture** (point cloud support)

### Required Libraries
- **Marching Cubes:** scikit-image, VTK, OpenVDB, or custom
- **Linear Algebra:** NumPy
- **Data Format:** JSON parser (built-in for most software)

---

## Verification & Testing

### Test Checklist
- [ ] Grid created with correct bounds
- [ ] Grid resolution: 22³
- [ ] Covariance matrices are symmetric
- [ ] Determinants computed correctly
- [ ] PDF evaluated at sample points
- [ ] Isosurface thresholds at 22% of max
- [ ] 4 separate meshes generated
- [ ] Colors assigned correctly
- [ ] Opacity set to 0.35
- [ ] Surface normals computed

### Quick Verification
```python
# Test PDF computation at origin
mu = [-0.3743, -1.1647, -0.6326]  # Bach
p = [0, 0, 0]
# PDF should be ≈ 0.00433
```

---

## Citation

If using this data in publications or presentations:

**Recommended citation:**
> Classical Music Embedding Cloud: 144-piece PCA projection with Gaussian density surfaces for Bach, Chopin, Debussy, and Mozart based on 30-dimensional feature extraction from sheet music.

**Include:**
- Dataset size and composition
- Feature extraction methodology
- PCA variance explained
- Marching cubes threshold (22%)

---

## Troubleshooting

### Issue: "Isosurfaces not visible"
- Check opacity is set to < 1.0
- Verify threshold values (compare with max(PDF))
- Confirm grid bounds and resolution

### Issue: "Surfaces are very smooth"
- Grid resolution (22³) is intentional for smoothness
- Use lower ISO threshold for more detail
- Check lighting and camera angle

### Issue: "Colors not matching"
- Verify Plotly palette hex codes
- Check color space (RGB vs other)
- Confirm per-composer color assignment

### Issue: "Coordinates don't match"
- Verify PCA random_state=42 (for reproducibility)
- Check standardization parameters
- Confirm feature selection (30 features)

---

## Additional Resources

### In This Package
- Source code: `export_cloud_data.py`
- Point data: `CLOUD_POINT_COORDINATES.json`
- Full math: `CLOUD_GENERATION_FORMULAS.md`

### External References
- **PCA:** scikit-learn documentation
- **Marching Cubes:** Lorensen & Cline (1987)
- **Gaussian KDE:** Rosenblatt (1956)

---

## Questions & Support

For detailed formulas → See **CLOUD_GENERATION_FORMULAS.md**  
For implementation steps → See **README_CLOUD_DATA.md**  
For quick parameters → See **QUICK_REFERENCE.md**  
For point coordinates → See **CLOUD_POINT_COORDINATES.json**  
For reproducibility → See **export_cloud_data.py**

---

**Package Generated:** January 26, 2026  
**Data Source:** 144 classical music pieces (sheet music)  
**Methodology:** PCA + Gaussian KDE + Marching Cubes  
**Format:** Complete with formulas, parameters, and point data

