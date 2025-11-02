# Phase 1 Step 1 Report

## Corpus Objective
We set out to assemble a high-quality, composer-balanced solo piano corpus from the PDMX archive so that later analysis stages can compare Bach, Mozart, Chopin, and Debussy on equal footing.

## Dataset Structure
- `15571083/PDMX.csv`: master metadata table with composer, rating, subset flags, and file pointers.
- `15571083/metadata/<id>/<id>.json`: rich metadata per score (instrument lists, instrumentation text, movement count).
- `15571083/mxl/...`: MusicXML scores indexed by the CSV’s `mxl` field.
- Additional subset files (e.g., `subset_paths/all_valid.txt`) that identify curated groups we can cross-reference.

## Implementation Summary
1. **Exploration** – Used `head`, `grep`, and ad-hoc `python3` snippets to understand composer naming variants, rating distributions, and instrumentation fields.
2. **Heuristics** – Defined solo piano requirements (presence of piano, exclusion of choir/ensemble tokens, limited part counts) and composer normalization rules that reject arrangements and non-target family members.
3. **Automation** – Created `src/corpus_curation.py`, a CLI tool that:
	- Normalizes composer strings via a `ComposerRule` dataclass with include/exclude token sets.
	- Loads per-score JSON metadata with caching to minimize disk hits.
	- Checks instruments, instrumentation tags, and free-text descriptions to confirm solo piano coverage.
	- Applies quality filters (minimum rating, `subset:no_license_conflict`, `subset:rated_deduplicated`, `is_best_unique_arrangement`).
	- Balances the resulting DataFrame by clipping each composer to the smallest available count and writes both a CSV and a plain-text list of MusicXML paths.

## Challenges and Mitigations
- **Composer Ambiguity**: Multiple Mozart and Bach family members appear; resolved with stricter token filtering and arrangement keyword detection using regular expressions.
- **Instrumentation Gaps**: Some metadata lists omit instruments; we added fallbacks that inspect instrumentation tags and descriptive text while still rejecting obvious ensemble cues.
- **Filter Trade-offs**: Initial strict filters yielded too few pieces; iterative counts led us to focus on the trio of high-quality flags (license-safe, deduplicated, best unique arrangement) plus a rating threshold that we later relaxed after confirming it did not change the balanced counts.

## Test Runs and Variants
- `python3 src/corpus_curation.py --min-rating 0` → 31 pieces per composer (124 total). Mean rating ≈ 4.85, zero duplicates, all safety flags active.
- Diagnostics: compared alternate variants (`--skip-license-filter`, `--skip-deduplicated-filter`, `--skip-unique-filter`, combinations, and unbalanced runs up to 685 works) by summarizing counts, ratings, duplicate titles, and instrumentation completeness.
- Observed that relaxing filters increases corpus size but introduces licensing uncertainty, near duplicates, and severe composer imbalance (e.g., Bach 267 vs. Debussy 52 when fully unbalanced).

## Conclusion
The balanced export with all safety filters active satisfies Step 1: it supplies 31 high-quality solo piano works per composer, keeps manual cleanup minimal, and positions us for fair cross-composer analysis in later phases. The larger unbalanced sets are retained only for reference; we recommend continuing with the balanced corpus for downstream feature extraction and modeling.
