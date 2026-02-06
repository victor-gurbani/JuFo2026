# Phase 1 Report

## Project Metrics at a Glance
- Filtered 254,077 catalogue entries down to 144 balanced solo piano works (36 per composer) after expanding Debussy name matching and manually vetting the 14 newly admitted scores for solo-piano instrumentation.
- Parsed 71,585 quarter-note durations, about 13.26 hours of music at 90 BPM, with average pieces spanning 148.22 measures and 2.10 parts.
- Extracted 36 analytic descriptors grouped into harmonic (16), melodic (11), and rhythmic (9) feature families.
- Evaluated 36 ANOVA hypotheses and 174 Tukey contrasts; the initial α=0.05 screen flagged 29 metrics and 62 statistically significant pairwise comparisons, but realizing that many simultaneous tests nearly guaranteed a false positive we kept that column for reference while adding Bonferroni (α/36≈0.0014) and Benjamini–Hochberg FDR (q<0.05) guards, leaving 14 and 29 metrics respectively.
- Validated the most extreme rhythmic cases after introducing meter-aware syncopation detection: Debussy’s “La cathédrale engloutie” still shows the largest duration spread (std. note duration 1.63) from sustained pedal sonorities, and Bach’s chorale anthology remains the syncopation leader (ratio 0.356) because of suspension-rich half-bar ties across 11.3k note events.
- Delivered two new inspection tools: an interactive PCA/UMAP-style embedding (`src/feature_embedding.py`) that reports variance/loadings for each axis while automatically omitting the outsized count features (`note_count`, `note_event_count`, `chord_event_count`, `chord_quality_total`, `roman_chord_count`, `dissonant_note_count`) so density metrics can steer the view, adds a 2D scatter companion via `--output-2d`, and offers both 3D (`--clouds-output`) and 2D (`--clouds-output-2d`) four-cloud summaries for footprint comparisons; and an annotation pipeline (`src/annotate_musicxml.py`) that labels dissonant notes, mirrors chord analyses as both chord symbols and text expressions so MuseScore shows the Roman numerals outright, and flags chromatic/ambiguous harmonies in coloured MusicXML (with optional PDF/PNG renders via MuseScore’s CLI template).

Recreate these tallies locally with `python3 src/aggregate_metrics.py`.

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
- **Alias Cleanup**: Revisiting Debussy-labelled rows uncovered 14 solo-piano scores rejected only because the composer string was truncated or misspelled. Allowing initials/typos while keeping arrangement/co-composer exclusions lifted every composer bucket to 36 without adding ensembles.

# Step 1 Test Runs and Variants
- `python3 src/corpus_curation.py --min-rating 0` → 36 pieces per composer (144 total). Mean rating ≈ 4.84, zero duplicates, all safety flags active.
- Diagnostics: compared alternate variants (`--skip-license-filter`, `--skip-deduplicated-filter`, `--skip-unique-filter`, combinations, and unbalanced runs up to 685 works) by summarizing counts, ratings, duplicate titles, and instrumentation completeness.
- Observed that relaxing filters increases corpus size but introduces licensing uncertainty, near duplicates, and severe composer imbalance (e.g., Bach 267 vs. Debussy 52 when fully unbalanced).

# Step 1 Conclusion
The balanced export with all safety filters active satisfies Step 1: it now supplies 36 high-quality solo piano works per composer, keeps manual cleanup minimal, and positions us for fair cross-composer analysis in later phases. The larger unbalanced sets are retained only for reference; we recommend continuing with the balanced corpus for downstream feature extraction and modeling (the unbalanced run currently lands at 186 works spread 48/39/63/36 across Bach/Mozart/Chopin/Debussy).

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

# Phase 2 Step 2: Melodic Features

- Introduced `src/melodic_features.py` to capture ambitus, contour variability, pitch-class entropy, and soprano/bass interaction metrics.
- Expanded chords into per-note events, added interval standard deviation and leap ratios, and replaced the flat correlation metric with discrete contrary/parallel/oblique motion tallies.
- Debugged independence scores by aligning voice change points; Bach now shows a meaningful spread (mean −0.13, σ ≈ 0.33) instead of clustering at zero.
- Deliverables: `data/features/melodic_features.csv`, boxplots under `figures/melodic/`, and `Melodic_Features.md` documenting each metric.

# Phase 2 Step 3: Rhythmic Features

- Built `src/rhythmic_features.py` to measure duration stats, downbeat emphasis, syncopation, micro-density of fast notes, and cross-hand subdivision mismatches.
- Refined syncopation detection to require weak-beat starts that cross the following beat, and used sliding duration windows to highlight clustered ornamentation.
- Cross-rhythm detection compares per-measure duration denominators between staves; filtering rests kept ratios meaningful (≈0.62–0.66 across composers).
- Outputs: `data/features/rhythmic_features.csv`, boxplots under `figures/rhythmic/`, and CLI parity with earlier phases for cached runs and plotting.

# Phase 2 Step 4: Significance Testing

- Added `src/significance_tests.py`, a CLI that ingests all feature tables, enforces minimum sample sizes, and runs one-way ANOVA per metric.
- When an omnibus test is significant, the script applies Tukey HSD (preferring `statsmodels`, falling back to SciPy) to pinpoint which composer pairs differ.
- Top contrasts include registral span (`pitch_range_semitones`, F=39.66, p≈7e-18) and dissonance usage (Debussy vs. Bach, p≈1.3e-10); rhythmic entropy and duration variance also show strong era splits.
- Outputs land in `data/stats/anova_summary.csv` and `data/stats/tukey_hsd.csv`, capturing F-statistics, adjusted p-values, and confidence intervals for downstream reporting.
- Because inspecting the raw α=0.05 hits (29 metrics) raised the odds-of-a-false-positive question, we now log those alongside the Bonferroni and Benjamini–Hochberg counts (14 and 29), keeping the original column for transparency while grounding conclusions in the corrected thresholds.
- `src/significance_visualizations.py` now converts those tables into figures under `figures/significance/`, including a top-15 `-log10(p)` bar chart, a pairwise-count heatmap, signed/absolute mean-difference heatmaps, and two feature-level views (sym-log + normalized). Count-heavy metrics (note/chord totals, Roman counts, dissonant note totals) are filtered out; after filtering, Debussy–Mozart still shows 16 distinct metrics, Bach–Debussy 9, while Chopin–Debussy remains at four with modest effect sizes.

# Phase 2 Enhancements: Embedding & Annotation

- The embedding viewer (`feature_embedding.py`) standardises the feature matrix, drops the raw count columns listed above so dense chorales do not dominate, projects via PCA or t-SNE, and reports variance ratios plus per-feature loadings (`data/stats/pca_loadings.csv`). Composer centroids reveal the axes: PC1 tracks chromatic density and leap-heavy harmony (Mozart ≪ Bach < Chopin < Debussy), PC2 contrasts long, downbeat-emphasised phrases (Mozart high) against note-saturated textures (Bach low), and PC3 lifts the Romantic/Impressionist pair via oblique motion, cross-rhythms, and registral spread. Chopin therefore bridges the classical and modern worlds—his cadential syntax and phrase symmetry keep him near Bach/Mozart on PC2, while his chromatic colour and sustained pedal tones pull him toward Debussy on PC1/PC3. Debussy pushes those same traits to the limit (planed chords, unresolved dissonances, layered polyrhythms), landing in a separate tier. Because each cloud spans roughly two standard deviations per axis, flattening onto PC1×PC2 obscures these differences—PC3’s contrapuntal signal is what keeps Debussy/Chopin suspended above the classical cluster in 3D, while the 2D plot lets their footprints overlap.
- A notable outlier emerged: *Mozart – Sonata in B♭ K 570 I* combines unusually stepwise melody with a high share of analyser-labelled “other” sonorities and frequent passing/appoggiatura dissonances. Those blended traits pull its PCA position to PC1 ≈ −2.5 and, with only a handful of close neighbours, t-SNE (perplexity = 30) pushes it far from the Mozart cluster. The score’s metadata checks out, so we interpret the separation as musical and will review the chord labelling with Olivia.
- I also ran the refreshed highlighter (`src/highlight_pca_piece.py`) on Joe Hisaishi’s *One Summer’s Day (Spirited Away)*—a piece I wavered on tagging as Romantic or Impressionist. The CLI now accepts multiple MusicXML inputs in one call, optional `--composer`/`--title` overrides, and a dark diamond palette that keeps every highlight readable atop the translucent clouds. The 3D clouds placed the piece squarely between the Chopin and Debussy clusters, slightly nearer to Chopin, matching how the score alternates lush Romantic progressions with Impressionist shimmer (see `summerdayhighlight.png`).
- The annotation helper (`annotate_musicxml.py`) colour-codes dissonant material, stamps lyric labels (passing/appoggiatura/other/dissonant-chord), inserts chord symbols whose lyrics capture the Roman numerals when available, and highlights chromatic or unclassified harmonies with a turquoise tint and `chromatic-chord` lyric. It can invoke MuseScore’s CLI through a `{input}/{output}` template to emit shareable PDFs or PNGs alongside the annotated MusicXML.
- `python3 src/generate_selected_annotations.py` batches the eight flagship annotated scores (two per composer) and supports the same renderer flags for one-command PDF/PNG exports.
