# Composer-set examples (for fast subset clouds)

This project supports a “two-view” scientific comparison for any subset of pieces:

- **Canonical axes:** project the subset into the fixed PCA space fit on the canonical corpus (e.g., Bach/Mozart/Chopin/Debussy). This makes cross-subset plots comparable because all coordinates live in the same PCA basis.
- **Refit axes:** refit PCA on the subset itself. This emphasizes local structure but changes axes (PC1/PC2/PC3 can rotate/flip depending on the subset).

The helper script [src/clouds_from_feature_cache.py](src/clouds_from_feature_cache.py) reads the cached full-feature CSV and generates composer clouds without re-parsing MusicXML.

## Prerequisite: full feature cache

Create the full-feature cache for a large corpus (resumable):

```bash
python3 src/embedding_cache.py \
  --project-only \
  --no-embedding-cache \
  --corpus-csv 15571083/PDMX.csv \
  --dataset-root 15571083 \
  --output-features-csv data/features/full_pdmx_feature_cache.csv \
  --resume
```

## How the config works

The JSON config file (example: [configs/composer_sets.example.json](configs/composer_sets.example.json)) defines named groups with regex filters.

Each group can specify:

- `include_composer`: regex patterns applied to `composer_label`
- `exclude_composer`: regex patterns applied to `composer_label`
- `include_title`: regex patterns applied to `title`
- `exclude_title`: regex patterns applied to `title`

Patterns are case-insensitive.

### Why regex, not exact matches?

PDMX composer strings can vary (accents, initials, “Last, First”, etc.). Regex gives you a practical way to catch common variants quickly.

## Example runs

### 1) Generate both views for a jazz subset

```bash
python3 src/clouds_from_feature_cache.py \
  --feature-cache data/features/full_pdmx_feature_cache.csv \
  --config configs/composer_sets.example.json \
  --group jazz_classic \
  --axes both \
  --max-per-composer 60 \
  --write-subset-csv
```

Outputs go to `figures/embeddings/subsets/`:

- `jazz_classic__canonical_axes_clouds.html`
- `jazz_classic__refit_axes_clouds.html`
- `jazz_classic__run_manifest.json`

### 2) Spanish nationalist subset (good for “new axes” effects)

```bash
python3 src/clouds_from_feature_cache.py \
  --feature-cache data/features/full_pdmx_feature_cache.csv \
  --config configs/composer_sets.example.json \
  --group spanish_nationalist \
  --axes both \
  --max-per-composer 80
```

### 3) Pop/modern subset (noisy; filter titles aggressively)

```bash
python3 src/clouds_from_feature_cache.py \
  --feature-cache data/features/full_pdmx_feature_cache.csv \
  --config configs/composer_sets.example.json \
  --group pop_modern_piano \
  --axes both \
  --max-per-composer 80
```

### More ready-made groups

The example config also includes additional sets you can run as-is:

- `video_game_music`
- `modern_minimalist`
- `latin_tango`

## Tips for building better groups

- Start with a small smoke test using `--limit 500` to validate your patterns quickly.
- If you see “arrangement/tutorial” pollution, add `exclude_title` patterns first.
- If composer aliases appear (e.g., accents or alternative spellings), extend `include_composer` with additional alternatives.
- Keep comparisons fair by using `--max-per-composer` so one composer doesn’t dominate the PCA fit.

## Notes on interpretability

- Canonical axes make it meaningful to say “this jazz subset sits between Chopin and Debussy” (because it’s literally in that basis).
- Refit axes can change PC semantics; compare **shapes, overlaps, density**, and **relative separation** rather than interpreting PC1 as the same concept across subsets.
