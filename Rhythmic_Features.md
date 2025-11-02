# Rhythmic Feature Reference

This guide documents every metric emitted by `src/rhythmic_features.py`. Each entry details:
- **What it measures** – the rhythmic idea behind the number.
- **How we calculate it** – the algorithm used in the extractor.
- **Example** – a plain-language interpretation drawn from typical corpus results.

All metrics are computed per score and later aggregated (e.g., composer averages, boxplots) for comparative studies.

## Core Event Totals

### note_event_count
- **What**: Total number of rhythmic events (notes and chordified notes) considered in the piece.
- **How**: Flatten the score, expand chords to single-note events, ignore zero-length artifacts, and count the resulting events.
- **Example**: "Bach chorales yield ~4,800 rhythmic events once every tied melodic step and chord tone is counted."

### avg_note_duration
- **What**: Mean duration (in quarter-note units) across all detected events.
- **How**: Average the `duration.quarterLength` of each note/chord event.
- **Example**: "A mean of 0.62 implies most events sit between eighths and quarters, typical of steady contrapuntal textures."

### std_note_duration
- **What**: Dispersion of event durations, revealing rhythmic variety.
- **How**: Apply the population standard deviation to the quarter-length list; returns 0.0 when only one event exists.
- **Example**: "Debussy's prelude shows 0.38, highlighting frequent switches between long pedals and short figurations."

## Beat-Level Activity

### notes_per_beat
- **What**: Average number of note events sounding per beat.
- **How**: Derive beat length from the primary time signature; divide total event count by the number of beats spanning the piece (`highestTime / beatLength`).
- **Example**: "A ratio of 5.8 for Bach's Prelude in C reflects dense sixteenth-note motion within each beat."

### downbeat_emphasis_ratio
- **What**: Share of events landing on strong beats.
- **How**: Use music21's `beatStrength`; count events with strength ≥ 0.75 and normalize by the total event count.
- **Example**: "Mozart sonatas hover near 0.41, indicating that fewer than half of the notes fall on metrically strong positions."

### syncopation_ratio
- **What**: Frequency of syncopated entries that begin on weak beats and sustain past the next beat.
- **How**: For each event on a weak beat (`beatStrength < 0.5`), check whether its duration crosses the upcoming beat boundary; the fraction of such events defines the ratio.
- **Example**: "Rag-inspired Debussy passages reach 0.18, showing nearly one in five notes create off-beat suspensions."

## Pattern Complexity

### rhythmic_pattern_entropy
- **What**: Entropy of duration trigrams, capturing diversity of local rhythmic cells.
- **How**: Tokenize durations to three-decimal strings, slide a 3-event window, count unique sequences, then compute Shannon entropy (base 2).
- **Example**: "An entropy of 4.9 for Ravel's waltz signals a wide variety of duration patterns from bar to bar."

### micro_rhythmic_density
- **What**: Proportion of sliding four-note windows dominated by fast notes.
- **How**: Slide a window of four events; if at least three durations are ≤ sixteenth-note (0.25 quarter-length), mark a hit; hits divided by total windows yield the ratio.
- **Example**: "Liszt etudes often reach 0.74, showing that most local spans are packed with rapid figures."

## Cross-Hand Interaction

### cross_rhythm_ratio
- **What**: Fraction of measures where the upper and lower staves deploy different duration denominators, signalling cross-rhythms or polyrhythms.
- **How**: For each aligned measure pair across the outer parts, gather denominators (ignoring rests), clamp to ≤64 to avoid spurious fractions, and compare sets; mismatches count as cross-rhythm measures.
- **Example**: "Chopin nocturnes average 0.63, meaning nearly two thirds of measures pit different subdivisions between hands."

## Putting It Together

- **Workflow**: Begin with `note_event_count`, `avg_note_duration`, and `std_note_duration` to gauge rhythmic material and spread. Inspect `notes_per_beat` and `downbeat_emphasis_ratio` for metric density, then study `syncopation_ratio` and `rhythmic_pattern_entropy` to understand off-beat tension and pattern variety. Conclude with `micro_rhythmic_density` and `cross_rhythm_ratio` to capture virtuosic figuration and cross-hand interplay.
- **Interpretive Example**: "A Debussy étude reporting 3,900 events, average duration 0.46, notes-per-beat 7.1, downbeat ratio 0.34, syncopation 0.21, entropy 5.1, micro density 0.83, and cross-rhythm 0.66 suggests relentless ornamental motion, weak-beat bias, and sustained cross-hand polyrhythms characteristic of late Romantic impressionism."
