# Significance Feature Interpretation Guide

This guide explains the Phase 2 significance testing outcomes across all harmonic, melodic, and rhythmic metrics. For each feature we summarize:
- **Measured trait** – the musical behaviour captured by the metric.
- **Statistical signal** – ANOVA strength and which composer pairs diverge most.
- **Musical reading** – how the numbers translate into audible characteristics.

All results use the balanced solo-piano corpus (31 works per composer). ANOVA employed `scipy.stats.f_oneway` at α = 0.05, followed by Tukey HSD comparisons. Reported means are rounded averages per composer.

## Harmonic Highlights

### Dissonance Ratio (F=23.41, p≈5.34×10⁻¹²)
- **Means**: Bach 0.373 · Mozart 0.310 · Chopin 0.510 · Debussy 0.566
- **What it measures**: Share of chordified sonorities containing non-consonant intervals after removing the melodic pitch.
- **Key contrasts**: Debussy vs Mozart (+0.256) and Chopin vs Mozart (+0.200) dominate Tukey results.
- **Musical interpretation**: Romantic and Impressionist textures linger on suspensions, added tones, and coloristic clashes, whereas Mozart’s Classical clarity keeps harmonies cleaner. Bach sits between—contrapuntal dissonance is present but resolves quickly.

### Harmonic Density Mean (F=12.25, p≈4.77×10⁻⁷)
- **Means**: Bach 2.31 · Mozart 2.10 · Chopin 2.51 · Debussy 2.84 pitch classes per chord.
- **Reading**: Debussy’s dense voicings (stacked 9ths, planed chords) increase simultaneous pitch-class variety, while Mozart’s triadic writing remains sparser. Chopin bridges the gap with embellished tertian harmony.

### Augmented Chord Share (F=9.70, p≈8.83×10⁻⁶)
- **Means (% of labeled chords)**: Bach 0.46 · Mozart 0.36 · Chopin 1.08 · Debussy 2.76.
- **Insight**: Augmented sonorities are rare before the Romantic era. Debussy’s threefold increase mirrors his use of whole-tone collections; Chopin’s elevated usage reflects chromatic dominant extensions.

### Modal Interchange Ratio (F=4.90, p≈3.02×10⁻³)
- **Means**: Bach 0.085 · Mozart 0.035 · Chopin 0.063 · Debussy 0.085.
- **Interpretation**: Both Bach and Debussy frequently borrow from parallel modes, though for different ends: Bach in chorale reharmonisations, Debussy for colour shifts. Mozart’s lower score underscores Classical adherence to diatonic functional harmony.

### Appoggiatura Ratio (F=4.65, p≈4.11×10⁻³)
- **Means**: Bach 0.012 · Mozart 0.050 · Chopin 0.028 · Debussy 0.018.
- **Takeaway**: Mozart’s melodic lines lean heavily on accented appoggiaturas—hallmark Classical sigh gestures. Bach and Debussy rely less on strong-beat leaning tones, favouring passing or planed dissonances instead.

## Melodic Highlights

### Pitch Range (F=39.66, p≈6.99×10⁻¹⁸)
- **Means (semitones)**: Mozart 45.94 · Bach 50.19 · Chopin 65.55 · Debussy 68.23.
- **Story**: Debussy and Chopin exploit nearly two octaves more keyboard range than Mozart, signalling Romantic/Impressionist fascination with extreme registers and resonance.

### Pitch-Class Entropy (F=16.74, p≈3.78×10⁻⁹)
- **Means**: Mozart 2.98 · Bach 3.18 · Chopin 3.25 · Debussy 3.28.
- **Meaning**: Entropy tracks tonal saturation. Debussy’s near-uniform pitch-class usage reflects modal and chromatic palettes; Mozart’s lower value lines with diatonic focus.

### Mean Melodic Interval (F=8.72, p≈2.80×10⁻⁵)
- **Means (semitones)**: Mozart 3.40 · Bach 4.55 · Chopin 4.75 · Debussy 4.94.
- **Implication**: Romantic/Impressionist melodies leap more frequently, contrasting Mozart’s stepwise rhetoric. Bach’s inventions already favour wider intervals than Classical norms.

### Melodic Interval Std Dev (F=7.35, p≈1.45×10⁻⁴)
- **Means**: Mozart 2.99 · Bach 3.33 · Chopin 3.50 · Debussy 4.05.
- **Interpretation**: Dispersion captures contour volatility. Debussy alternates between tiny inflections and sudden vaults, while Mozart’s melodic shapes stay within narrower bounds.

### Leap vs Step Ratios (Melodic)
- **Leap ratio (F=4.93, p≈2.88×10⁻³)**: Debussy 0.667 > Bach 0.615 ≈ Chopin 0.607 >> Mozart 0.489.
- **Conjunct ratio (F=3.01, p≈3.28×10⁻²)**: Mozart 0.406 > Bach 0.358 > Chopin 0.342 > Debussy 0.271.
- **Real world**: Mozart sustains classical smoothness, while Debussy balances contour with large gestures; Chopin sits between, mixing lyrical steps with expressive leaps.

## Rhythmic Highlights

### Note-Duration Variability (F=15.48, p≈1.42×10⁻⁸)
- **Std dev means (beats)**: Debussy 0.661 > Chopin 0.437 > Mozart 0.381 ≈ Bach 0.364.
- **Interpretation**: Debussy’s writing juxtaposes sustained washes and flurries, inviting rubato; Bach’s contrapuntal steady pulse keeps durations more uniform.

### Rhythmic Pattern Entropy (F=14.04, p≈6.63×10⁻⁸)
- **Means**: Debussy 5.13 >> Mozart 3.87 ≈ Chopin 3.68 ≈ Bach 3.67.
- **Meaning**: Debussy cycles through diverse duration trigrams, aligning with floating, less periodic rhythms.

### Notes per Beat (F=2.85, p≈4.05×10⁻²)
- **Means**: Bach 5.78 > Chopin 4.91 > Debussy 4.54 > Mozart 4.14.
- **Implication**: Bach’s contrapuntal figuration packs beats with events; later composers balance texture with sustained tones.

### Downbeat Emphasis (F=6.54, p≈3.92×10⁻⁴)
- **Means**: Mozart 0.231 > Debussy 0.196 > Chopin 0.179 > Bach 0.132.
- **Reading**: Mozart leads the pack in strong-beat onsets, reaffirming clear Classical metric hierarchies. Bach’s dense inner motion dilutes downbeat prominence.

### Syncopation Ratio (F=3.49, p≈1.80×10⁻²)
- **Means**: Debussy 0.040 > Bach 0.020 > Chopin 0.017 > Mozart 0.006.
- **Musical cue**: Debussy frequently sustains weak-beat notes across strong beats, producing floating rhythms; Mozart maintains predictable accents.

## Cross-Feature Observations
- Debussy exhibits the widest spreads across nearly every dimension: registral span, harmonic density, rhythmic variety, and syncopation. The statistics match the impression of impressionist colour and loosened meter.
- Mozart anchors the Classical profile—tight pitch range, diatonic focus, strong downbeats, and minimal dissonant linger. His elevated appoggiatura ratio highlights expressive simplicity rather than vertical tension.
- Bach combines contrapuntal density (high notes-per-beat) with moderate dissonance, placing him between Classical clarity and Romantic saturation.
- Chopin often bridges styles: richer harmony than Bach/Mozart, yet less dense than Debussy; melodic ranges almost as wide as Debussy but with smoother conjunct motion than impressionism.

## Validating the Logic
- Removing raw-count metrics (note/chord totals) from visual heatmaps prevents scale distortions and keeps focus on proportional behaviour.
- Canonical composer ordering ensures Tukey pair labels are consistent (e.g., "Bach vs Mozart" regardless of extraction order).
- Re-running the visualization suite after each adjustment confirms that Debussy–Mozart remains the most divergent pair (16 significant features post-filter), while Chopin–Debussy differences stay modest (three features with small effects).

Use these interpretations alongside the raw CSVs and plots in `figures/significance/` to tie statistical outcomes back to audible stylistic traits.
