# Phase 1 Overview

## Corpus Curation Summary (Step 1)

## Project Context
The goal of Phase 1 Step 1 was to curate a balanced set of solo piano scores for Bach, Mozart, Chopin, and Debussy from the PDMX dataset (`15571083/`). This summary documents the code we wrote, the dataset structure we relied on, the challenges encountered, the diagnostics we ran, and the conclusions we drew about the best corpus variant to carry forward.

## Dataset Structure Overview
- `15571083/PDMX.csv`: primary metadata table with composer names, ratings, subset flags, and relative paths to MusicXML and JSON metadata.
- `15571083/metadata/**`: JSON files containing instrumentation details (`data.score.instruments`, `instrumentations`, `instrumentation_text`, `parts`).
- `15571083/mxl/**`: compressed MusicXML files referenced in the CSV.
- `15571083/subset_paths/*.txt`: predefined curated subsets (e.g., `rated_deduplicated`, `no_license_conflict`).

Exploratory commands used to understand the structure included:
```
head -n 5 15571083/PDMX.csv
python3 - <<'PY'
import pandas as pd
df = pd.read_csv('15571083/PDMX.csv')
print(df.columns.tolist())
print(df['composer_name'].str.lower().value_counts().head())
PY
python3 - <<'PY'
import pandas as pd
df = pd.read_csv('15571083/PDMX.csv')
mask = df['composer_name'].str.contains('mozart', case=False, na=False)
print(df.loc[mask, 'composer_name'].unique())
PY
```

## Implementation Details
- New script: `src/corpus_curation.py` (pure Python, pandas, argparse).
- Core components:
  - `ComposerRule` dataclass normalizes composer names (required tokens, optional groups, exclusion tokens).
  - `normalize_text` removes accents and punctuation for reliable matching.
  - `load_score_metadata` caches JSON metadata reads to avoid repeated disk access.
  - `extract_instrument_names`, `extract_instrumentation_tags`, and `is_solo_piano` ensure the score is truly solo piano (checks instruments list, instrumentation tags, descriptive text, and part counts).
  - `curate_corpus` applies metadata filters, enforces solo piano, verifies file existence, and optionally balances counts per composer.
  - `write_outputs` saves both a CSV and a text file of absolute MusicXML paths for downstream parsing.

Sample command with all safety filters active and the default balancing:
```
python3 src/corpus_curation.py --min-rating 0
```
This creates `data/curated/solo_piano_corpus.csv` with 31 works per composer (124 total) and `data/curated/solo_piano_mxl_paths.txt` listing absolute paths.

Alternate command keeping all filters but disabling balancing:
```
python3 src/corpus_curation.py --min-rating 0 --max-per-composer 999 --output-csv data/curated/solo_piano_corpus_unbalanced.csv --output-paths data/curated/solo_piano_mxl_paths_unbalanced.txt
```
This produced 181 works spread unevenly: Bach 48, Mozart 39, Chopin 63, Debussy 31.

## Challenges and Decisions
1. **Composer normalization**: Raw metadata mixes variants (e.g., “Mozart, W. A.”, “Wolfgang Amadeus Mozart”). We introduced `ComposerRule` with required tokens and exclusion lists to filter out individuals like Leopold or Franz Xaver Mozart.
2. **Arrangements and ensembles**: Many metadata entries include “arr.” or “transcription”. We built a keyword detector (`is_arrangement`) to reject such cases. Additional instrumentation checks ensure no choir/ensemble terms slip through.
3. **Sparse instrumentation metadata**: Some JSON metadata lacks instrument lists but includes descriptive text. `is_solo_piano` now accepts valid evidence from instruments, instrumentation tags, or instrumentation text while dropping obvious ensemble clues.
4. **Quality filters vs. coverage**: Relying solely on `rating >= 4` and `is_original` left too few scores. Pivoting to dataset-provided subset flags (`subset:no_license_conflict`, `subset:rated_deduplicated`, `is_best_unique_arrangement`) yielded a sizeable, reliable corpus.

## Diagnostics and Tests
- Verified CLI help loads without errors:
```
python3 src/corpus_curation.py --help
```
- Measured counts under different filter combinations:
```
python3 src/corpus_curation.py
python3 src/corpus_curation.py --min-rating 0
python3 src/corpus_curation.py --min-rating 0 --skip-license-filter
python3 src/corpus_curation.py --min-rating 0 --skip-deduplicated-filter
python3 src/corpus_curation.py --min-rating 0 --skip-license-filter --skip-deduplicated-filter
python3 src/corpus_curation.py --min-rating 0 --skip-license-filter --skip-deduplicated-filter --skip-unique-filter
python3 src/corpus_curation.py --min-rating 0 --max-per-composer 999
```
- Summarized each CSV variant to compare composer counts and ratings:
```
python3 - <<'PY'
import pandas as pd
from pathlib import Path

variants = {
	'default': Path('data/curated/solo_piano_corpus.csv'),
	'skip_license': Path('data/curated/solo_piano_corpus_skiplicense.csv'),
	'skip_dedup': Path('data/curated/solo_piano_corpus_skipdedup.csv'),
	'skip_all_balanced': Path('data/curated/solo_piano_corpus_skipall.csv'),
	'skip_all_unbalanced': Path('data/curated/solo_piano_corpus_skipall_unbalanced.csv'),
}

for name, path in variants.items():
	if not path.exists():
		continue
	df = pd.read_csv(path)
	print(name, len(df), df.groupby('composer_label').size().to_dict(), f"mean rating {df['rating'].mean():.2f}")
PY
```

## Key Findings
- **Balanced filtered corpus** (`--min-rating 0` with default filters): 31 pieces per composer (124 total), mean rating 4.85, no duplicate titles, zero missing instrument info. Ready for downstream feature extraction and comparative analysis.
- **Unbalanced filtered corpus** (`--min-rating 0 --max-per-composer 999`): 181 pieces, uneven distribution (Chopin-heavy, Debussy-light), mean rating 4.79, already contains duplicate titles. Requires manual normalization before any fair comparison.

## Parsing & Summaries (Step 2)

### Objectives & Data Flow
- Source: the balanced Step 1 corpus (`data/curated/solo_piano_corpus.csv`) and MusicXML paths (`data/curated/solo_piano_mxl_paths.txt`).
- Goal: parse each MusicXML via `music21`, collect structural metadata (measure count, part count, duration), and persist lightweight summaries for downstream feature extraction.

### Implementation (`src/score_parser.py`)
- CLI built with `argparse` and `pandas`; depends on `music21.converter.parse`.
- `ParseResult` dataclass captures composer, title, absolute path, measures, parts, and duration.
- `read_mxl_paths` loads the curated CSV (or a fallback path list) into a DataFrame.
- `parse_score` coerces any `Score`, `Part`, or `Opus` returned by `music21` into a unified `stream.Score`. Special handling for `Part` (wrap in a new `Score`) and `Opus` (merge contained scores/parts) prevents crashes on compilations.
- `summarize_score` extracts measure counts (from first part), part counts, and quarter-length duration with guards around missing metadata.
- `parse_corpus` iterates the curated list, applies an optional `--limit`, and either skips or raises on parse errors. Warnings are printed to stderr when `--no-skip-errors` is not set.
- Results are serialized to JSON (`write_results`), and per-composer averages/counts are printed via `print_stats`. A `--stats-from` flag allows loading a previously written JSON to recompute statistics without re-parsing.

### Sample Commands
```
# Parse entire curated corpus with default settings
python3 src/score_parser.py --output data/parsed/summaries.json

# Quick smoke test on N scores
python3 src/score_parser.py --limit 2 --output data/parsed/test_parses.json

# Report statistics from existing summaries
python3 src/score_parser.py --stats-from data/parsed/summaries.json
```

### Issues Encountered & Mitigations
1. **music21 return variants**: `converter.parse` may yield `Score`, `Part`, or `Opus`. Without normalization we hit attribute errors; resolution involved wrapping parts into a new `Score` and flattening opus collections.
2. **Measure counting**: Some scores lack explicit measure objects, causing exceptions. All measure extraction is wrapped in try/except with a `None` fallback.
3. **Fraction durations**: Durations sometimes arrive as `Fraction`; casting to `float` keeps JSON serialization simple while preserving quarter-length precision.
4. **Large multi-work compilations**: Items like the complete Bach chorales produce extremely high measure counts. The script logs these but keeps them in the dataset for analysis, leaving later phases to decide on outlier handling.

### Diagnostics
- Validation run during development:
```
python3 src/score_parser.py --limit 2 --output data/parsed/test_parses.json
python3 src/score_parser.py --stats-from data/parsed/test_parses.json
```
- Full corpus parse:
```
python3 src/score_parser.py --output data/parsed/summaries.json
```

Each run prints per-composer averages for measures, parts, and duration, plus piece counts, enabling quick sanity checks before feature extraction.

## Conclusions
For Step 1, the balanced variant with all safety filters provides the best foundation. It offers:
- Equal representation for Bach, Mozart, Chopin, Debussy, enabling unbiased cross-composer analysis.
- High average ratings and curated subset flags to minimize noisy or questionable entries.
- Minimal manual cleanup before moving to Phase 2 tasks (feature extraction and analysis).

The unbalanced export remains useful for exploratory expansions but demands significant curation to avoid skewed results. We will continue Phase 2 using `data/curated/solo_piano_corpus.csv` and its companion path list generated by `python3 src/corpus_curation.py --min-rating 0`.

Parsing (Step 2) now delivers structured summaries for every curated score, stored in `data/parsed/summaries.json` and accompanied by command-line tooling to review averages without reprocessing. These artifacts form the inputs for upcoming feature extraction scripts.

## Harmonic Feature Extraction (Phase 2 Step 1)

### Objectives & Scope
- Build an end-to-end harmonic analysis pipeline using `src/harmonic_features.py` that consumes the curated corpus and parsed scores.
- Capture chord distributions, harmonic densities, dissonance behaviour, and Roman numeral trends across the four-composer corpus.
- Produce CSV feature tables and diagnostic plots to compare composers and to feed later modelling efforts.

### Implementation Highlights
- **Chord Processing**: chordified each score with `music21`, dropped zero-length or empty chords, and computed quality percentages (major/minor/diminished/augmented/other) plus harmonic density (unique pitch classes per chord).
- **Dissonance Metrics**: constructed a chord index `(offset, end, chord)` to locate harmony against melodic notes; classified dissonant notes as passing tones, appoggiaturas, or other based on intervallic motion and beat strength; captured aggregate ratios and raw counts.
- **Roman Numeral Analysis**: analysed tonal centres via `score.analyze("key")`, translated the chord stream to Roman numerals, calculated deceptive cadence incidence, and tracked modal mixture frequency.
- **Outputs**: persisted results as `data/features/harmonic_features.csv`, rendered per-feature boxplots under `figures/harmonic/`, and provided descriptive stats grouped by composer for CLI runs.

### Roadblocks & Resolutions
- **Zero Dissonance Readings**: Initial runs yielded zero dissonant notes because `note.pitchClass` is `None` until a `Pitch` object is accessed. Fix: retrieve `note.pitch` first, then its `pitchClass`, and reuse that pathway for resolution checks.
- **Chord-Tone Leakage**: Removing the melodic pitch from the supporting chord broke when the helper function was deleted as “unused.” Restored `remove_pitch_from_chord`, added a guard comment, and applied it before consonance checks so non-chord tones no longer misclassify as chord tones.
- **Resolution Detection**: Early attempts treated the next note as a chord tone even when it duplicated the melody. Adjusted the logic to strip the note from the resolution chord before interval analysis, yielding realistic passing/appoggiatura ratios.
- **Runtime Diagnostics**: Added targeted notebooks-free probes by running the classifier on a single score (`--limit 1/2/5`) and printing counts to confirm 80+ dissonances detected in short preludes; interrupted long Bach chorale extractions with `Ctrl+C` while iterating on classification.

### Validation & Usage Patterns
- Smoke tests with `python3 src/harmonic_features.py --limit 2 --output-csv /tmp/harmonic_features_sample.csv` to spot-check counts and ratios without reprocessing the full corpus.
- Larger dry runs (`--limit 10`) verified distribution stability before committing to full exports.
- `--features-from /path/to/csv --skip-plots` path supports descriptive stats and plotting without recomputation; exercised against `/tmp/harmonic_features_sample.csv` to confirm code paths work when reusing cached results.
- Observed meaningful dissonance ratios (e.g., ~8–15% passing tones in inventions) and modal mixture rates aligning with expectations for Bach-heavy subsets.

### Current Deliverables
- `src/harmonic_features.py`: fully documented CLI supporting extraction, cached reads, optional plotting, and error-skipping.
- `data/features/harmonic_features.csv`: corpus-wide harmonic metrics regenerated after the dissonance fixes.
- `figures/harmonic/boxplot_*.png`: per-feature composer comparisons suitable for reports.
- Terminal script outputs that echo composer-level descriptive stats, aiding regression detection during subsequent modifications.
