# JuFo2026

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
