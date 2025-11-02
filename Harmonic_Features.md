# Harmonic Feature Reference

This document explains every metric produced by `src/harmonic_features.py`. Each entry covers:
- **What it measures** – the musical concept behind the value.
- **How we calculate it** – the exact processing performed by the script.
- **Example** – a natural-language interpretation to help reason about real outputs.

All features are computed per score, then aggregated (e.g., averaged) across composers for reports.

## Core Event Counts

### chord_event_count
- **What**: Total number of chord events extracted from the chordified score. Each event represents a harmony spanning a rhythmic slice after `music21` chordification.
- **How**: `chordify_score` collapses the score into simultaneous sonorities, `extract_chords` filters out empty/zero-duration chords, and the list length is recorded.
- **Example**: “Prelude BWV 924 contains 182 distinct chord events once chordified.”

### chord_quality_total
- **What**: Number of chord events whose quality we successfully classified into major/minor/diminished/augmented/other.
- **How**: Same chord list as above; each chord’s `quality` attribute increments the matching bucket. Sum of all buckets (including “other”).
- **Example**: “Out of 182 chord events, 170 had recognizable qualities, leaving 12 as ‘other’ (suspensions, clusters, etc.).”

## Chord Quality Percentages

Percentages are relative to `chord_quality_total`; values sum to 100% (with rounding).

### chord_quality_major_pct
- **What**: Share of chords labelled “major.”
- **How**: Count major chords, divide by `chord_quality_total`, multiply by 100.
- **Example**: “Major sonorities account for 36% of the harmony, reflecting a predominance of bright tertian chords.”

### chord_quality_minor_pct
- **What**: Share of chords labelled “minor.”
- **How**: Count minor chords ÷ total × 100.
- **Example**: “Minor chords comprise 28%, signalling frequent tonicization or modal mixture.”

### chord_quality_diminished_pct
- **What**: Share of fully diminished triads or sevenths.
- **How**: Count quality `diminished` ÷ total × 100.
- **Example**: “Roughly 8% of chords are diminished, mostly leading-tone vii° chords approaching cadences.”

### chord_quality_augmented_pct
- **What**: Share of augmented triads.
- **How**: Count quality `augmented` ÷ total × 100.
- **Example**: “Only 1.2% augmented sonorities, suggesting occasional chromatic coloration rather than systematic use.”

### chord_quality_other_pct
- **What**: Remaining chords that music21 could not label as standard tertian qualities (includes sus chords, quartal stacks, clusters).
- **How**: `other = total - (major+minor+diminished+augmented)`; percentage computed same way.
- **Example**: “12% of chords fall outside simple quality labels, mostly suspensions in the chorales.”

## Texture & Consonance

### harmonic_density_mean
- **What**: Average number of distinct pitch classes sounding at once—an indicator of voicing density.
- **How**: For each chord, count unique pitch classes; average across all chords.
- **Example**: “A mean density of 3.4 indicates most sonorities are triads with occasional added tones.”

### dissonance_ratio
- **What**: Fraction of chords that `music21` deems non-consonant (intervallic dissonance beyond standard tertian consonance).
- **How**: Iterate chords, call `ch.isConsonant()`, count false responses, divide by evaluated chords.
- **Example**: “A dissonance ratio of 0.18 shows that about 18% of vertical sonorities feature suspensions or clash intervals.”

## Melodic Dissonance Classification

All ratios below refer to non-chord tones detected in the primary melodic part. `dissonant_note_count` is the denominator.

### dissonant_note_count
- **What**: Number of melodic notes classified as non-chord tones after filtering.
- **How**: For each melodic note (ignoring edges/ties), match concurrent chord, strip the melodic pitch from the chord, and test the remaining bass interval for consonance. Non-consonant results increment this count.
- **Example**: “The Invention contains 228 dissonant passing or leaning notes across the right-hand line.”

### passing_tone_ratio
- **What**: Share of dissonant notes labelled as passing tones (stepwise motion in the same direction, resolving to a chord tone).
- **How**: Inspect intervals to previous and next notes; both must be ≤ whole step, share direction, and resolve to a consonant tone in the following harmony.
- **Example**: “11.7% of dissonances function as passing tones, e.g., the right hand filling thirds with stepwise motion.”

### appoggiatura_ratio
- **What**: Share of dissonant notes that behave as accented appoggiaturas (leap into dissonance on a strong beat, stepwise resolution).
- **How**: Require incoming leap (≥ minor third), outgoing step (≤ whole step), strong beat (`beatStrength ≥ 0.5`), and consonant resolution in the next chord.
- **Example**: “1.5% are appoggiaturas, such as the accented upper note resolving down by step in measure 12.”

### other_dissonance_ratio
- **What**: Residual category for dissonant notes that are neither passing tones nor appoggiaturas (e.g., suspensions, neighbor tones, escape tones).
- **How**: `1 - (passing + appoggiatura)` when counts exist.
- **Example**: “The remaining 97% reflect suspensions and neighbor tones common in Bach chorales.”

## Tonal Function

### roman_chord_count
- **What**: Number of distinct Roman numeral chords identified (adjacent duplicates collapsed).
- **How**: Analyse score key via `score.analyze("key")`; convert each chord to a Roman numeral; skip repeats of the same figure.
- **Example**: “The prelude exhibits 145 unique Roman numeral events after collapsing identical repeats.”

### deceptive_cadence_ratio
- **What**: Proportion of Roman numeral transitions that match the classic V→vi deceptive cadence pattern within the analysed key.
- **How**: Iterate consecutive Roman numerals; count transitions where the first has scale degree 5 and the second 6; divide by total transitions.
- **Example**: “3% of transitions resolve deceptively, showing occasional surprises instead of perfect cadences.”

### modal_interchange_ratio
- **What**: Frequency of modal mixture (borrowed chords) detected by `romanNumeral.isMixture()`.
- **How**: Evaluate each Roman numeral’s `isMixture` method; ratio of positive responses to evaluated chords.
- **Example**: “Mixture ratio 0.12 indicates that ~12% of harmonies borrow from the parallel mode (e.g., ♭VI in C major).”

## Using the Metrics Together

- **Interpretation Pipeline**: For any piece, begin with `chord_event_count` to gauge material length, inspect quality percentages for tonal colour, check `harmonic_density_mean` and `dissonance_ratio` for textural tension, and read the melodic dissonance ratios for contrapuntal detail. Roman numeral metrics contextualise cadential and modal behaviour.
- **Practical Example**: Suppose a Chopin nocturne reports 480 chord events, 60% minor, density 3.8, dissonance ratio 0.22, dissonant note count 310 with 8% passing/appoggiatura and 0.14 mixture ratio. We can infer a dense chromatic texture rich in suspensions, prominent minor colouring, and notable modal borrowing typical of Romantic harmony.
