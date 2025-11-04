# Phase 1 Overview

## Corpus Curation Summary (Step 1)

## Project Context
The goal of Phase 1 Step 1 was to curate a balanced set of solo piano scores for Bach, Mozart, Chopin, and Debussy from the PDMX dataset (`15571083/`). This summary documents the code we wrote, the dataset structure we relied on, the challenges encountered, the diagnostics we ran, and the conclusions we drew about the best corpus variant to carry forward.

## Project Scale Highlights
- Started from a catalog of 254,077 PDMX scores and tightened it to a license-safe, solo piano cohort.
- Retained 124 balanced works (31 per composer) spanning 18,925 measures and ~63.6k quarter notes, roughly 11.8 listening hours at a moderate 90 BPM assumption.
- Parsed structural summaries showing an average of 153 measures and 2.1 parts per score, ensuring downstream features reflect comparable texture.
- Engineered 36 descriptive metrics split across harmonic (16), melodic (11), and rhythmic (9) families before feeding them into the statistics stack.
- Ran 36 omnibus ANOVAs and 162 Tukey HSD contrasts; 27 features and 56 composer pairings cleared the α=0.05 bar, energising the narrative in the later sections of this article.
- Re-verified rhythmic outliers after feature extraction refinements: Debussy’s “La cathédrale engloutie” retains the widest rhythmic dispersion (std. note duration 1.63) thanks to alternating long pedal clusters and brief ornamental pickups, while Bach’s chorale anthology still posts the highest syncopation ratio (0.356) driven by suspension-heavy half-beat entries across 11k events. A refreshed syncopation detector now adapts to in-score meter changes, confirming these extremes are musical traits rather than tooling artifacts.
- Added an interactive embedding explorer (`feature_embedding.py`) that projects all 36 features into a 3D Plotly scatter. PCA runs now report per-axis variance, automatically drop the raw count-style fields (`note_count`, `note_event_count`, `chord_event_count`, `chord_quality_total`, `roman_chord_count`, `dissonant_note_count`) so density and contour metrics drive the axes, and dump loadings (`data/stats/pca_loadings.csv`) for interpretation; t-SNE remains available for curved manifolds when linear axes blur stylistic differences.
- Enriched score inspection tooling (`annotate_musicxml.py`) to tag passing tones, appoggiaturas, and other dissonances with colours and lyric labels, overlay chord analyses as visible chord symbols whose lyrics carry the Roman numeral (or fallback descriptors when analysis fails), tint chromatic/ambiguous harmonies turquoise with a `chromatic-chord` lyric, and optionally render PDF/PNG via a MuseScore command template for collaborators without notation software. A new batch script (`src/generate_selected_annotations.py`) regenerates the eight flagship annotated scores in one command.

Run `python3 src/aggregate_metrics.py` from the repository root to reproduce every figure in this highlight reel.

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

## Melodic Feature Extraction (Phase 2 Step 2)

### Objectives & Scope
- Extend the feature pipeline with melodic contour and contrapuntal metrics using a new CLI (`src/melodic_features.py`).
- Quantify ambitus, contour smoothness, pitch-class diversity, and coordination between outer voices for every curated piece.
- Produce CSVs and boxplots to track stylistic evolution across eras.

### Implementation Highlights
- **Note Collection**: Flattened each score and expanded chords into constituent note events so pitch-range and entropy calculations reflect stacked sonorities, not just single melodic lines.
- **Contour Metrics**: Computed mean interval size, standard deviation, step vs. leap ratio, and chromatic entropy for the primary melodic part, mirroring analytical music theory terminology.
- **Outer-Voice Coordination**: Replaced the earlier sliding correlation (which kept values near zero) with a timeline-based comparison of soprano and bass change points. Classified each event as contrary, parallel, or oblique motion and derived both ratios and a net independence index `(contrary - parallel) / total`.
- **Outputs**: Wrote `data/features/melodic_features.csv`, added boxplots under `figures/melodic/`, and surfaced descriptive stats via the CLI.

### Roadblocks & Iterations
- **Flat Independence Scores**: Initial correlation approach produced variance near zero for Bach, contradicting expectations. Switching to discrete motion categories exposed meaningful spreads (e.g., Bach mean −0.13 with σ ≈ 0.33).
- **Chord Tokenization Gaps**: Early pitch range and entropy values ignored chord tones. By expanding chords into individual proxy notes, registral spans and entropy better reflected harmonic texture.
- **Contour Detail**: Averages alone concealed leap-heavy writing; we introduced interval standard deviation and leap ratio to capture Romantic/Impressionist volatility.

### Validation & Diagnostics
- Smoke tests with `--limit` verified new columns populated correctly; spot-checked pieces like `BWV 847` and Debussy preludes to confirm contour metrics aligned with musical intuition.
- Full-corpus run (`python3 src/melodic_features.py --output-csv data/features/melodic_features.csv`) produced composer-level means showing expected trends: wider ranges and higher leap ratios for Chopin/Debussy, higher oblique motion in nocturnes, etc.
- Inspected extreme independence values (e.g., Mozart K.331 theme at −0.46 vs. Debussy “Cathédrale engloutie” at +0.24) to ensure classifications matched perceived motion.

### Current Deliverables
- `src/melodic_features.py`: CLI mirroring the harmonic script with plotting, cached reads, and motion statistics.
- `data/features/melodic_features.csv`: refreshed melodic dataset including new contour and motion fields.
- `figures/melodic/boxplot_*.png`: composer comparisons for contour, entropy, and motion ratios.
- `Melodic_Features.md`: narrative descriptions of every melodic metric, their calculation, and interpretation examples.

## Rhythmic Feature Extraction (Phase 2 Step 3)

### Objectives & Scope
- Implement a dedicated rhythmic pipeline (`src/rhythmic_features.py`) to quantify duration usage, metrical stress, syncopation, micro-rhythm density, and cross-hand subdivisions.
- Provide CSV/visual outputs that complement harmonic/melodic datasets and surface stylistic contrasts in timing practices across eras.

### Implementation Highlights
- **Event Harvesting**: Flattened each score, treated chords as single rhythmic events, and captured offset/length/beat-strength triples for subsequent analysis.
- **Duration Statistics**: Computed mean and standard deviation of note lengths alongside `notes_per_beat`, derived via `score.highestTime` and the primary beat unit.
- **Metrical Emphasis**: Used `music21` beat strengths to compute downbeat emphasis ratios; flagged syncopations by detecting weak-beat onsets whose durations spill past the next beat boundary.
- **Pattern & Density Metrics**: Built sliding windows over duration tokens (n=3) to calculate Shannon entropy, and counted high-density windows where at least three of four notes were sixteenth-or-faster to quantify rubato potential.
- **Cross-Rhythm Detection**: Compared denominator sets of duration fractions in corresponding measures for top/bottom staves; mismatched subdivision sets raise the cross-rhythm flag, approximating polyrhythmic interplay.
- **Outputs**: Saved `data/features/rhythmic_features.csv`, generated boxplots in `figures/rhythmic/`, and wired the CLI for cached runs identical to prior scripts.

### Roadblocks & Iterations
- **Time Signature Access**: `music21` type exports differ by version; switched to string-based lookups for time signatures and guarded beat-length extraction to avoid attribute errors.
- **Syncopation Heuristic**: Initial attempts over-counted eighth notes; refined by requiring weak-beat starts that extend beyond the remaining beat span.
- **Micro-Density Windows**: Experimented with ratios vs. sliding windows; settled on four-event windows with ≥3 fast notes to reflect clustered ornamental passages.
- **Cross-Rhythm Noise**: Denominator comparisons initially flagged rests; filtered rests out to focus on active subdivisions, yielding reasonable ratios (~0.62–0.66) without nullifying pieces lacking two-staff texture.

### Validation & Diagnostics
- `--limit` smoke tests confirmed per-piece outputs (e.g., BWV 847 showing near-uniform sixteenth-note density and low syncopation; chorales highlighting high downbeat emphasis but negligible micro-density).
- Full dataset run produced composer means aligned with expectations: Bach’s high `notes_per_beat`, Debussy’s elevated entropy, and Romantic/Impressionist works exhibiting greater syncopation and cross-rhythm frequency.
- Spot-checked extreme cases (e.g., Debussy planed textures) to confirm cross-rhythm flags coincided with triplet-vs-duplet passages.

### Current Deliverables
- `src/rhythmic_features.py`: fully featured CLI with plotting, cached reuse, and warning handling.
- `data/features/rhythmic_features.csv`: rhythmic dataset covering duration stats, syncopation, entropy, micro-density, and cross-rhythm ratios.
- `figures/rhythmic/boxplot_*.png`: composer comparisons for each rhythmic metric.

## Significance Testing (Phase 2 Step 4)

### Objectives & Scope
- Validate whether the extracted harmonic, melodic, and rhythmic metrics separate the four composers statistically.
- Implement an automated pipeline that performs one-way ANOVA per feature and executes Tukey HSD post-hoc comparisons when omnibus tests are significant.

### Implementation Highlights
- **Consolidated CLI**: Added `src/significance_tests.py`, which ingests the three feature CSVs (`data/features/*.csv`), filters numeric columns, and enforces minimum sample counts before testing.
- **ANOVA Engine**: Uses `scipy.stats.f_oneway` to compute F-statistics and p-values, tracking group sizes to guard against underpowered comparisons.
- **Post-Hoc Strategy**: Prefers `statsmodels.pairwise_tukeyhsd` for Tukey tables; when unavailable, falls back to SciPy's native `stats.tukey_hsd`, assembling confidence intervals and adjusted p-values manually.
- **Outputs**: Writes summary statistics to `data/stats/anova_summary.csv` and detailed pairwise comparisons to `data/stats/tukey_hsd.csv`, with configurable alpha thresholds and group-size requirements.
- **Visualization Layer**: Introduced `src/significance_visualizations.py` to transform the statistics into advanced plots—top-`-log10(p)` bar charts, three pairwise heatmaps (significant-feature counts, signed mean differences, absolute mean differences), and two feature-level Tukey heatmaps—stored under `figures/significance/`. Count-heavy metrics (note/chord totals, Roman counts, dissonant note tallies) are filtered out by default so the gradients focus on proportional effects.
- **Interpretive Reference**: Authored `Significance_Features.md`, cataloguing every tested feature with ANOVA strength, leading Tukey contrasts, and the real-world musical behaviours they imply. This bridges raw statistics to narrative-ready insights for the final report.

### Key Findings
- **Strong Differentiators**: `pitch_range_semitones` (F=39.66, p≈7.0×10⁻¹⁸) and `dissonance_ratio` (F=23.41, p≈5.3×10⁻¹²) showed the clearest cross-era separation, confirming wider Romantic/Impressionist registral spreads and heightened modern dissonance.
- **Rhythmic Contrast**: `std_note_duration` and `rhythmic_pattern_entropy` both yielded p-values < 1×10⁻⁷, highlighting increasingly varied rhythmic cells in Chopin and Debussy relative to Bach/Mozart.
- **Targeted Post-Hoc Insights**: Tukey tests flagged, for example, Debussy’s dissonance ratio exceeding Bach’s (p≈1.3×10⁻¹⁰) and Mozart’s chord counts falling well below Bach’s (p≈2.1×10⁻²). Augmented-chord usage likewise separates Debussy and Chopin from earlier eras.
- **Near-Threshold Measures**: `notes_per_beat` barely cleared the 0.05 cutoff (p≈0.041), while `avg_note_duration`, `cross_rhythm_ratio`, and `voice_independence_index` remained non-significant under current sample sizes.
- **Pairwise Landscape**: The pairwise-count heatmap shows Debussy–Mozart still diverging on 16 metrics after filtering, with Bach–Debussy and Bach–Mozart at 10 and 9 respectively, while Chopin–Debussy trails at three. The signed mean-difference heatmap indicates Debussy and Mozart differ most through positive deltas (e.g., higher rhythmic entropy, augmented usage), whereas Bach vs. Chopin skews slightly negative.
- **Normalized Perspective**: A companion heatmap divides mean differences by each feature’s across-pair standard deviation, emphasizing relative effect sizes and further dampening residual count bias.
- **Notable Findings**: Mozart unexpectedly tops the downbeat-emphasis metric (0.23 vs Bach’s 0.13), reinforcing the strength of Classical metric clarity. Debussy’s syncopation ratio doubles Bach’s and quadruples Mozart’s, echoing impressionist rubato; Bach’s notes-per-beat outpace Romantic/Impressionist works, highlighting how contrapuntal density generates motion without relying on syncopation.

## Feature Embedding & Score Annotation (Phase 2 Enhancements)

### Multidimensional Embedding
- `python3 src/feature_embedding.py --method pca --output figures/embeddings/pca_3d.html --output-2d figures/embeddings/pca_2d.html --loadings-csv data/stats/pca_loadings.csv` standardises the feature matrix, applies PCA, and writes both 3D and 2D Plotly views (30 numeric dimensions participate after dropping the raw count features listed above).
- The script prints the variance captured by PCs 1–3 (21.5%, 15.5%, 10.3%), clarifying that the axes are weighted blends of metrics—PC1 now balances harmonic density, registral reach, and dissonance ratios rather than being dominated by sheer event totals, PC2 pivots on rhythmic duration spread, and PC3 contrasts pitch-class entropy with downbeat clarity.
- Because the corpus is balanced and stylistic traits overlap (especially between late Classical and Romantic works), we observe a single elongated cloud rather than clean clusters. Loadings in `data/stats/pca_loadings.csv` identify which metrics tug points along each axis; hovering reveals the piece title and MusicXML path for contextual drill-down. For non-linear separations, `--method tsne --perplexity 25` offers curved manifolds but with axes that carry no standalone meaning.
- Passing `--clouds-output figures/embeddings/composer_clouds_3d.html` and `--clouds-output-2d figures/embeddings/composer_clouds_2d.html` now generates companion “four cloud” views: the first uses semi-transparent 3D iso-surfaces, while the second projects the same Gaussians as filled 2D contours so the footprint comparison works even in flat layouts.
- **Axis interpretation** – Composer centroids land at roughly Mozart (−2.05, 0.82, −0.45), Bach (−0.27, −1.06, −0.91), Chopin (0.41, −0.02, 0.47), and Debussy (1.92, 0.26, 0.90) in (PC1, PC2, PC3) space. PC1 is dominated by harmonic density, leap ratio, diminished/“other” chord share, and dissonance usage: Debussy races toward +PC1 because of chromatic harmonies and saturated sonorities, Mozart anchors the negative side with diatonic clarity, Bach stays near the origin thanks to dense harmony offset by contrapuntal consonance, and Chopin falls between eras. PC2 contrasts average duration and downbeat emphasis (positive loadings) with notes-per-beat and micro-density (negative). Mozart’s long phrases and metric regularity push him upward, Bach’s relentless note stream drives him downward, and Chopin/Debussy hover near zero because they blend long singing lines with ornamental bursts. PC3 rewards oblique motion, cross-rhythms, rhythmic entropy, and registral span while penalising parallel motion; Debussy and Chopin therefore sit well above the plane, whereas Mozart and Bach remain below it.
- **Why 2D collapses** – Each composer’s footprint spans σ≈1.7–2.4 units per axis, so projecting onto PC1×PC2 hides key contrasts: without PC3, the Chopin/Debussy ridge folds over the classical pair, Bach’s negative PC2 tail overlaps Chopin’s centre, and Mozart’s separation relies on the vertical lift contributed by contrapuntal motion. The 3D view restores the layout observed in MuseScore: Mozart and Bach occupy different “slots” along PC2 while remaining distinct on PC1, Chopin bridges toward Debussy through shared chromatic density, and Debussy separates further by soaring along PC3.
- **Musical reading** – Chopin’s blend of classical scaffolding and Romantic colour explains his midpoint. Harmonically he shares chromatic mediants, extended dominants, and sustained pedal pitches with Debussy, which drives him positive on PC1 and lifts PC3 through oblique motion and cross-rhythms. Yet his phrases still cadence like Mozart’s, with balanced periods and clear tonal gravity, keeping his downbeat emphasis and notes-per-beat near the classical mean. Debussy abandons that anchor: stacked planing chords, modal mixture, unresolved dissonances, and layered polyrhythms max out every chromatic and independence metric, pushing him into a distinct “third slot” beyond Chopin. Bach conversely pairs dense harmony with strict counterpoint, leaving him closer to the origin on PC1 and deep in negative PC2, while Mozart’s diatonic clarity and metrical poise hold the opposite corner.

### Annotated MusicXML
- `python3 src/annotate_musicxml.py --mxl <path> --output figures/annotated/<name>.mxl` colour-codes the original score: passing tones (orange), appoggiaturas (violet), other dissonances (red), and residual dissonant chord members (light red). Each flagged note gains a lyric label (e.g., `passing_tone`), and dissonant chords receive `dissonant-chord` text for quick filtering. Every harmonic slice additionally gains a chord symbol (reflecting the concert spelling) whose lyric carries the Roman numeral when analysis succeeds, otherwise a descriptive fallback. Chromatic or unclassified chords are tinted turquoise with a `chromatic-chord` lyric so they jump off the page.
- A synchronised text expression now mirrors the chord label so MuseScore displays the analysis directly even if chord-symbol inference falls back; when the Roman figure is available it becomes both the symbol figure and the lyric text.
- Supplying a renderer template such as `--renderer-template "mscore -o {output} {input}" --render-format pdf` drives MuseScore’s CLI to emit an engraved PDF/PNG next to the annotated MusicXML, giving reviewers without MuseScore an immediate visual.
- `python3 src/generate_selected_annotations.py` regenerates the eight flagship annotated scores (Bach: Fugue BWV 846, Toccata BWV 565; Mozart: Lacrimosa, Sonata K545/iii; Chopin: Ballade No.1, Etude Op.25 No.11; Debussy: Clair de Lune, La cathedrale engloutie) in one command and accepts the same renderer flags for batch PDF/PNG export.
- Combined with the embedding view, these artefacts allow rapid toggling between global stylistic trends and bar-level verification of how the heuristics interpret individual scores.

### Validation & Diagnostics
- Ran the CLI end-to-end against the full corpus; console output lists the top 20 ANOVA hits, verifying consistent sample counts (≥31 pieces per composer for most metrics).
- Confirmed Tukey fallback by executing on a SciPy-only environment (no `statsmodels`); the script now emits identical CSV schemas regardless of backend.
- Spot-checked CSV contents to ensure meandiff signs align with composer ordering and that confidence intervals bracket the reported contrasts.
- Generated the visualization suite to verify narratives: the bar chart corroborates the five most significant features, the symlog-scaled feature heatmap emphasizes Mozart’s dual proximity to Baroque and Impressionist voices, the normalized view confirms those contrasts once scales are standardised, and the new pair-level heatmaps separate sheer frequency of differences from their directional magnitude.
- Verified Tukey exports after canonicalising composer pairs and excluding raw counts; recomputed matrices match the filtered feature reference (Debussy–Mozart differences persist across 16 metrics, while Chopin–Debussy remains limited with minor effect sizes). The written `Significance_Features.md` summary now aligns with these final statistics.

### Current Deliverables
- `src/significance_tests.py`: reusable significance-testing CLI with ANOVA/Tukey logic and configurable thresholds.
- `src/significance_visualizations.py`: companion plotting tool that turns significance tables into interpretable graphics (bar chart plus two heatmaps with filtering, sym-log scaling, and normalization controls).
- `Significance_Features.md`: narrative interpretation of every statistically tested feature, linking numbers to stylistic traits.
- `data/stats/anova_summary.csv`: omnibus test catalog covering all harmonic, melodic, and rhythmic features.
- `data/stats/tukey_hsd.csv`: pairwise composer comparisons for each significant feature, including adjusted p-values and confidence intervals.
