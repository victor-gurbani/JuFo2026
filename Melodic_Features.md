# Melodic Feature Reference

This guide details every metric emitted by `src/melodic_features.py`. Each section explains:
- **What it measures** – the underlying melodic idea.
- **How we calculate it** – the procedure in the extractor.
- **Example** – a plain-language interpretation of a typical value.

All metrics are computed per score and later aggregated by composer or era for analysis.

## Core Totals

### note_count
- **What**: Total number of melodic note events considered (upper part plus chord expansions).
- **How**: Flatten the score, convert chord tones into individual notes, and count the resulting `music21.note.Note` objects.
- **Example**: “The Bach chorale set yields ~5,000 melodic events because every voice-leading step in the chorale collection is counted.”

### pitch_range_semitones
- **What**: Melodic ambitus, measured in semitones between the lowest and highest note in the score.
- **How**: Compute the difference between maximum and minimum MIDI pitches across all melodic notes.
- **Example**: “Debussy’s ‘Feuilles mortes’ spans 68 semitones (~5½ octaves), reflecting his wide registral palette.”

## Melodic Motion

### avg_melodic_interval
- **What**: Mean absolute size (in semitones) of successive melodic intervals in the primary melodic voice (default: top staff).
- **How**: Create a melodic note sequence, take absolute differences of successive MIDI pitches, and average them.
- **Example**: “Mozart averages 3.4 semitones—predominantly stepwise or small leaps.”

### melodic_interval_std
- **What**: Dispersion of interval sizes, highlighting variability in melodic contour.
- **How**: Apply the population standard deviation to the interval list produced above.
- **Example**: “Chopin’s études show a standard deviation of 1.6, meaning the leap sizes fluctuate dramatically between steps and octaves.”

### conjunct_motion_ratio
- **What**: Fraction of intervals that are stepwise (≤ whole tone).
- **How**: Count intervals with magnitude ≤ 2 semitones and divide by the total number of intervals.
- **Example**: “A ratio of 0.52 indicates just over half the motion is stepwise, consistent with smooth classical phrasing.”

### melodic_leap_ratio
- **What**: Share of melodic intervals classified as leaps (> whole tone).
- **How**: Count intervals with magnitude > 2 semitones and divide by the total.
- **Example**: “Debussy’s pieces often exceed 0.45, showing that almost half of his melodic moves are leaps created by planing or registral shifts.”

## Pitch-Class Usage

### pitch_class_entropy
- **What**: Evenness of pitch-class distribution (12-tone entropy, base 2).
- **How**: Convert notes to pitch classes, compute the Shannon entropy of the normalized histogram.
- **Example**: “An entropy of 3.25 in Chopin reflects the chromaticism of his harmonic vocabulary versus Mozart’s tighter 2.96.”

## Two-Voice Interaction

### voice_independence_index
- **What**: Net balance between contrary and parallel motion in the outer voices (soprano vs. bass).
- **How**: Extract highest pitches from the top staff and lowest pitches from the bottom staff, align their change points, and compare pitch-direction signs at each event. The index is `(contrary - parallel) / total_events`, where `total_events` also counts oblique cases.
- **Example**: “Bach’s inventions score -0.13, meaning parallel motion slightly outweighs contrary motion once oblique motion is factored in.”

### contrary_motion_ratio
- **What**: Proportion of aligned events where upper and lower voices move in opposite directions.
- **How**: Using the same aligned event stream above, count contrary motion occurrences and divide by total events (contrary + parallel + oblique).
- **Example**: “Bach chorales average 0.31, confirming frequent contrary motion between outer voices.”

### parallel_motion_ratio
- **What**: Share of events where both voices move in the same direction (parallel or similar motion).
- **How**: Count events where both direction signs match (excluding stationary movement) and divide by total events.
- **Example**: “A ratio of 0.43 for Bach indicates that, despite contrary tendencies, parallel motion remains common when both voices move.”

### oblique_motion_ratio
- **What**: Fraction of events where one voice moves while the other stays static.
- **How**: Count events where either the soprano or the bass holds a repeated note while the other voice moves, divided by total events.
- **Example**: “Chopin’s nocturnes often produce ratios above 0.5 because the left hand sustains while the melody moves freely.”

## Putting It Together

- **Workflow**: Start with `note_count` and `pitch_range_semitones` to gauge corpus scope and registral spread. Use interval statistics (`avg_melodic_interval`, `melodic_interval_std`, `conjunct_motion_ratio`, `melodic_leap_ratio`) to understand contour shape. Inspect `pitch_class_entropy` for tonal vs. chromatic tendencies, and conclude with the outer-voice interaction metrics to assess contrapuntal independence.
- **Interpretive Example**: “A Chopin étude showing range 70, average interval 5.5, leap ratio 0.6, entropy 3.3, and oblique ratio 0.55 suggests virtuosic leaps, rich chromaticism, and a left hand that often sustains while the melody soars, aligning with Romantic pianism.”
