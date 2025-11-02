# Phase 1 Report

# Step 1: Corpus Objective
We set out to assemble a high-quality, composer-balanced solo piano corpus from the PDMX archive so that later analysis stages can compare Bach, Mozart, Chopin, and Debussy on equal footing.

## Dataset Structure
- `15571083/PDMX.csv`: master metadata table with composer, rating, subset flags, and file pointers.
- `15571083/metadata/<id>/<id>.json`: rich metadata per score (instrument lists, instrumentation text, movement count).
- `15571083/mxl/...`: MusicXML scores indexed by the CSV’s `mxl` field.
- Additional subset files (e.g., `subset_paths/all_valid.txt`) that identify curated groups we can cross-reference.

# Step 1 Implementation Summary
1. **Exploration** – Used `head`, `grep`, and ad-hoc `python3` snippets to understand composer naming variants, rating distributions, and instrumentation fields.
2. **Heuristics** – Defined solo piano requirements (presence of piano, exclusion of choir/ensemble tokens, limited part counts) and composer normalization rules that reject arrangements and non-target family members.
3. **Automation** – Created `src/corpus_curation.py`, a CLI tool that:
	- Normalizes composer strings via a `ComposerRule` dataclass with include/exclude token sets.
	- Loads per-score JSON metadata with caching to minimize disk hits.
	- Checks instruments, instrumentation tags, and free-text descriptions to confirm solo piano coverage.
	- Applies quality filters (minimum rating, `subset:no_license_conflict`, `subset:rated_deduplicated`, `is_best_unique_arrangement`).
	- Balances the resulting DataFrame by clipping each composer to the smallest available count and writes both a CSV and a plain-text list of MusicXML paths.

# Step 1 Challenges and Mitigations
- **Composer Ambiguity**: Multiple Mozart and Bach family members appear; resolved with stricter token filtering and arrangement keyword detection using regular expressions.
- **Instrumentation Gaps**: Some metadata lists omit instruments; we added fallbacks that inspect instrumentation tags and descriptive text while still rejecting obvious ensemble cues.
- **Filter Trade-offs**: Initial strict filters yielded too few pieces; iterative counts led us to focus on the trio of high-quality flags (license-safe, deduplicated, best unique arrangement) plus a rating threshold that we later relaxed after confirming it did not change the balanced counts.

# Step 1 Test Runs and Variants
- `python3 src/corpus_curation.py --min-rating 0` → 31 pieces per composer (124 total). Mean rating ≈ 4.85, zero duplicates, all safety flags active.
- Diagnostics: compared alternate variants (`--skip-license-filter`, `--skip-deduplicated-filter`, `--skip-unique-filter`, combinations, and unbalanced runs up to 685 works) by summarizing counts, ratings, duplicate titles, and instrumentation completeness.
- Observed that relaxing filters increases corpus size but introduces licensing uncertainty, near duplicates, and severe composer imbalance (e.g., Bach 267 vs. Debussy 52 when fully unbalanced).

# Step 1 Conclusion
The balanced export with all safety filters active satisfies Step 1: it supplies 31 high-quality solo piano works per composer, keeps manual cleanup minimal, and positions us for fair cross-composer analysis in later phases. The larger unbalanced sets are retained only for reference; we recommend continuing with the balanced corpus for downstream feature extraction and modeling.

# Step 2: Parsing Summary

- Implemented `src/score_parser.py`, a music21-based CLI that reads the curated corpus, parses each MusicXML file, and writes JSON summaries capturing measures, part counts, and quarter-note durations.
- Added resilience to music21’s varied return types (Score, Part, Opus) by wrapping or merging them into a unified `Score` before measurement extraction.
- Outputs are stored under `data/parsed/`, and every run prints per-composer averages plus total counts for quick validation.
- Key commands:
	- `python3 src/score_parser.py --output data/parsed/summaries.json`
	- `python3 src/score_parser.py --limit 5 --output data/parsed/test_parses.json`
	- `python3 src/score_parser.py --stats-from data/parsed/summaries.json`
- Common issues (e.g., massive multi-work compilations, missing measures) are handled gracefully by falling back to `None` and emitting warnings when necessary.
- Result: a ready-to-use structural dataset for Phase 2 feature extraction without needing to re-parse MusicXML files for quick stats or sanity checks.

# Phase 2 Step 1: Harmonic Features

- `src/harmonic_features.py` extracts chord quality mixes, harmonic density, dissonance classifications, and Roman numeral trends for every curated score.
- Restored a helper to strip melodic tones from chords and fixed pitch-class lookups so dissonant notes register correctly (passing/appoggiatura ratios now populate).
- Validated the classifier with `--limit` smoke tests (1–10 pieces) and quick inspection scripts before rerunning the full corpus export.
- Outputs: `data/features/harmonic_features.csv`, boxplots under `figures/harmonic/`, and a `--features-from` mode to reuse cached feature tables without recomputation.
