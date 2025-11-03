# JuFo2026

## Project Scale Highlights
- Filtered a raw archive of 254,077 PDMX entries down to a safety-screened solo piano cohort.
- Balanced 124 curated works (31 per composer) spanning 18,925 measures and ~11.8 listening hours at a moderate tempo.
- Computed 36 hand-crafted features across three pillars: 16 harmonic, 11 melodic, and 9 rhythmic descriptors.
- Ran 36 omnibus ANOVA tests with 27 passing a 0.05 threshold and 162 Tukey HSD contrasts, surfacing 56 significant composer-to-composer gaps.
- Generated reusable CSVs, JSON summaries, and multi-view visualizations that feed the documentation in `Article.md`, `ShortArticle.md`, and `Significance_Features.md`.
- Reproduce these headline numbers locally with `python3 src/aggregate_metrics.py`.

## Automated Quickstart

This project includes a `quickstart.sh` script that automates the entire pipeline, from setting up the environment to running the final analysis.

### Usage

To run the quickstart script, execute the following command from the project root:

```bash
./quickstart.sh
```

By default, the script will create a Python virtual environment in the `venv` directory to avoid interfering with your system's Python packages.

### --no-venv Flag

If you prefer to use your system's Python installation, you can use the `--no-venv` flag:

```bash
./quickstart.sh --no-venv
```

This is useful if you have already installed the required dependencies globally or are using a package manager like `conda`.

## Corpus Curation (Phase 1 Step 1)

Follow these steps to regenerate the curated CSV and path list used in this project:

1. Ensure the PDMX dataset is extracted under `15571083/` at the project root and that `src/corpus_curation.py` is present.
2. From the repository root, run:
	```
	python3 src/corpus_curation.py --min-rating 0
	```
	This writes the balanced, safety-filtered corpus to `data/curated/solo_piano_corpus.csv` and the corresponding MusicXML paths to `data/curated/solo_piano_mxl_paths.txt`.
3. To explore alternative variants (e.g., relaxed filters or unbalanced counts), append the appropriate flags, for example:
	```
	python3 src/corpus_curation.py --min-rating 0 --skip-license-filter
	python3 src/corpus_curation.py --min-rating 0 --skip-license-filter --skip-deduplicated-filter --skip-unique-filter --max-per-composer 999
	```
	Each run accepts `--output-csv` and `--output-paths` if you want to save results under different filenames.

## Score Parsing (Phase 1 Step 2)

Follow these steps to generate or inspect parsed summaries:

1. Ensure `music21` is installed and the curated corpus exists (defaults under `data/curated/`).
2. Parse the full corpus:
	```
	python3 src/score_parser.py --output data/parsed/summaries.json
	```
3. Useful flags:
	- `--limit N`: parse only the first `N` scores (smoke tests).
	- `--no-skip-errors`: abort on the first parsing failure instead of skipping.
	- `--csv PATH` / `--paths PATH`: override the curated CSV or the MusicXML path list.
4. Recompute statistics without re-parsing:
	```
	python3 src/score_parser.py --stats-from data/parsed/summaries.json
	```

## Harmonic Features (Phase 2 Step 1)

Generate harmonic descriptors, dissonance profiles, and Roman numeral trends:

1. Ensure the curated CSV exists under `data/curated/` (use the commands above if needed).
2. Extract harmonic features and plots in one pass:
	```
	python3 src/harmonic_features.py --output-csv data/features/harmonic_features.csv
	```
3. Helpful flags:
	- `--limit N`: process only the first `N` scores (smoke tests).
	- `--no-skip-errors`: stop on the first extraction failure.
	- `--skip-plots`: omit boxplot generation when running headless pipelines.
	- `--features-from PATH`: load an existing CSV instead of recomputing, still enabling plots/statistics.
	- `--figure-dir DIR`: override the output directory for boxplots (default `figures/harmonic`).
	- `--csv PATH` / `--paths PATH`: point at alternate curated corpora or MusicXML path lists.
4. Example cached run that reuses the last export for quick statistics:
	```
	python3 src/harmonic_features.py --features-from data/features/harmonic_features.csv --skip-plots
	```

## Melodic Features (Phase 2 Step 2)

Extract melodic contour and independence metrics:

1. Ensure the curated corpus is available as described above.
2. Run the melodic extractor:
	```
	python3 src/melodic_features.py --output-csv data/features/melodic_features.csv
	```
3. Useful flags mirror the harmonic script:
	- `--limit N`: process the first `N` scores for smoke tests.
	- `--no-skip-errors`: halt on the first parsing/extraction error.
	- `--skip-plots`: suppress boxplot generation.
	- `--features-from PATH`: reuse a previously generated CSV while still emitting stats/plots.
	- `--figure-dir DIR`: change the destination for melodic feature boxplots (default `figures/melodic`).
	- `--csv PATH` / `--paths PATH`: override the curated corpus inputs.
4. Example reuse of cached results:
	```
	python3 src/melodic_features.py --features-from data/features/melodic_features.csv --skip-plots
	```

## Rhythmic Features (Phase 2 Step 3)

Extract rhythmic density, syncopation, and cross-hand subdivision metrics:

1. Ensure the curated corpus is available.
2. Run the rhythmic extractor:
	```
	python3 src/rhythmic_features.py --output-csv data/features/rhythmic_features.csv
	```
3. Useful flags (mirroring other scripts):
	- `--limit N`: process the first `N` scores for smoke tests.
	- `--no-skip-errors`: stop on the first extraction failure instead of skipping.
	- `--skip-plots`: suppress boxplot generation.
	- `--features-from PATH`: reuse a previously generated CSV for stats/plots without recomputation.
	- `--figure-dir DIR`: direct rhythmic feature plots to a custom directory (default `figures/rhythmic`).
	- `--csv PATH` / `--paths PATH`: override the curated inputs.
4. Example cached run:
	```
	python3 src/rhythmic_features.py --features-from data/features/rhythmic_features.csv --skip-plots
	```

## Significance Testing (Phase 2 Step 4)

Run omnibus ANOVA and Tukey HSD post-hoc comparisons across every exported feature:

1. Confirm harmonic, melodic, and rhythmic CSVs exist under `data/features/`.
2. Execute the significance pipeline:
	```
	python3 src/significance_tests.py \
	  --anova-output data/stats/anova_summary.csv \
	  --tukey-output data/stats/tukey_hsd.csv
	```
3. Flags mirror earlier scripts:
	- `--alpha VALUE`: change the significance threshold (default `0.05`).
	- `--min-group-size N`: require at least `N` pieces per composer (default `3`).
	- `--no-tukey`: skip Tukey HSD generation (useful if only SciPy is available).
	- Alternate feature tables can be supplied with `--harmonic-csv`, `--melodic-csv`, or `--rhythmic-csv`.
4. Results:
	- `data/stats/anova_summary.csv` lists F-statistics, p-values, and sample sizes for each feature.
	- `data/stats/tukey_hsd.csv` records composer-to-composer comparisons (using `statsmodels` when available, else SciPy's implementation).
	- Optional: produce bar charts and heatmaps with
		```
		python3 src/significance_visualizations.py --top-n 15
		```
		This writes figures to `figures/significance/`: a top ANOVA bar chart, three pairwise heatmaps (significant-feature counts, signed mean difference, absolute mean difference), plus two feature-level heatmaps (sym-log and normalized). Count-heavy metrics (e.g., `note_event_count`, `roman_chord_count`, `dissonant_note_count`) are filtered automatically; supply additional `--exclude-pattern` flags to customise. Use `--no-symlog`, `--skip-normalized`, or adjusted `--top-n` values to tailor the views.

## Feature Embedding Explorer

Create an interactive 3D scatter of every piece embedded in feature space:

```
python3 src/feature_embedding.py --output figures/embeddings/pca_3d.html --loadings-csv data/stats/pca_loadings.csv
```

- Default method is PCA; pass `--method tsne` (optionally `--perplexity 25`) for a non-linear view.
- Points are colour-coded by composer and expose the title and MusicXML path on hover. The HTML output can be opened in any browser.
- When using PCA the script prints variance explained per axis and writes the feature loadings to the CSV above, clarifying which metrics pull the cloud toward each direction.

## MusicXML Harmonic Annotation

Generate colour-coded MusicXML files that highlight dissonant material for inspection in MuseScore or other editors:

```
python3 src/annotate_musicxml.py \
	--mxl 15571083/mxl/0/46/Qmaug5pXs59BJq1VTdXhrKAU3DHxRqusrDgXraMS6xDtYk.mxl \
	--output figures/annotated/La_cathedrale_annotated.mxl \
	--renderer-template "mscore -o {output} {input}" --render-format pdf
```

- Notes classified as passing tones, appoggiaturas, or other dissonances reuse the feature-extraction heuristics, receive distinct colours, and gain lyric labels for quick inspection. Remaining notes in dissonant chords are tinted red and marked `dissonant-chord`.
- Supply a renderer template (shown above for the MuseScore CLI) to emit a PDF or PNG alongside the annotated MusicXML for collaborators without engraving software.
- Open the exported MusicXML or rendered file to review how the analysis aligns with the original notation and to spot-check the automated classifications.
