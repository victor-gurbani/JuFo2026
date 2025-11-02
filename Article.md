# Corpus Curation Summary

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

## Conclusion
For Step 1, the balanced variant with all safety filters provides the best foundation. It offers:
- Equal representation for Bach, Mozart, Chopin, Debussy, enabling unbiased cross-composer analysis.
- High average ratings and curated subset flags to minimize noisy or questionable entries.
- Minimal manual cleanup before moving to Phase 2 tasks (feature extraction and analysis).

The unbalanced export remains useful for exploratory expansions but demands significant curation to avoid skewed results. We will continue Phase 2 using `data/curated/solo_piano_corpus.csv` and its companion path list generated by `python3 src/corpus_curation.py --min-rating 0`.
