# Plan: JNMR Academic Manuscript — Computational Stylistic Analysis of Western Piano Music

## Context

Write a complete, submission-ready ~6000-8000 word academic manuscript in `JournalMusicResearch/main.tex` for the **Journal of New Music Research** (Taylor & Francis). The paper synthesises two German-language Jugend Forscht documents into a single English paper about computational stylistic analysis of Bach, Mozart, Chopin, and Debussy using 36 music21-extracted features from 144 PDMX solo piano scores.

The manuscript must cover:
- **Static analysis**: ANOVA/Tukey HSD significance testing, PCA dimensionality reduction
- **Dynamic analysis**: Evolutionary coefficients (v), stylistic acceleration (a), Difference-in-Differences (DD), Triple-Difference (DDD) with the 'Romantic Reversal' finding
- **Validation**: Hisaishi and Ravel case studies projected into PCA space

## Scope

### IN Scope
- Bach, Mozart, Chopin, Debussy — 4 composers only
- 36 features (16 harmonic, 11 melodic, 9 rhythmic) extracted via music21
- 144 PDMX solo piano scores (36 per composer)
- ANOVA (29/36 significant), Tukey HSD (174 pairwise comparisons), PCA (3 components, 48.2% variance)
- Evolutionary coefficients, DD, DDD with mode effects
- Hisaishi and Ravel validation case studies
- Satie as additional validation if the agent deems it adds value
- All existing figures from `figures/` directory
- British English throughout ("-ise" spelling)
- T&F `interact.cls` template with `natbib`/`apacite`
- Anonymisation toggle (blind review support)

### OUT of Scope (EXCLUDED — must NOT appear in manuscript)
- Random Forest classifier
- Neural networks / MLP
- Elton John
- Pop music / modern popular music analysis
- Any machine learning classification
- The "Zyklische Evolution" section from v2

## Technical Stack
- LaTeX with `interact.cls` (Taylor & Francis)
- `natbib` with APA citation style (`\citep`, `\citet`)
- `subfig` for sub-figures (NOT subcaption)
- `\tbl{}{}` macro for tables
- `\bibliographystyle{apacite}` with `\bibliography{references}`
- `pdflatex` for compilation

## Key Data (Verified — Use These Exact Numbers)

### Corpus
- 144 solo piano scores, 36 per composer
- Sourced from PDMX dataset (254,077 total entries, filtered to 144)
- 21,344 total measures, 71,585 quarter-note durations

### ANOVA Results (from `data/stats/anova_summary.csv`)
- 29/36 features significant after FDR correction (q < 0.05)
- Top features by p_value_fdr: pitch_range_semitones (F=42.18, p_fdr < 6.32e-18), dissonance_ratio (F=30.65, p_fdr < 4.85e-14), pitch_class_entropy (F=21.79, p_fdr < 1.47e-10), std_note_duration (F=19.35, p_fdr < 1.35e-09), rhythmic_pattern_entropy (F=18.19, p_fdr < 3.63e-09), harmonic_density_mean (F=17.67, p_fdr < 5.25e-09)
- **IMPORTANT**: Always read the actual CSV for exact values. The numbers above are approximate from the CSV. Use the `p_value_fdr` column (last column) for FDR-corrected p-values, NOT the raw `p_value` column.

### PCA (from `data/stats/pca_loadings.csv`)
- 30 features used (6 count features excluded)
- PC1 = 22.3% variance, PC2 = 16.1%, PC3 = 9.8%, cumulative = 48.2%
- Random seed = 42

### Evolutionary Coefficients (from `data/stats/evolution_coefficients.csv`)
**CRITICAL**: Read the actual CSV for exact values. The values below are from the file:
- pitch_range_semitones: v_classical=-5.14, v_romantic=+20.00, v_impressionist=+2.78, accel_romantic(DD)=+25.14, accel_impressionist=-17.22
- dissonance_ratio: v_classical=-0.086, v_romantic=+0.229, v_impressionist=+0.048, accel_romantic(DD)=+0.315, accel_impressionist=-0.180
- harmonic_density_mean: v_classical=-0.253, v_romantic=+0.500, v_impressionist=+0.322, accel_romantic(DD)=+0.754, accel_impressionist=-0.178
- rhythmic_pattern_entropy: v_classical=+0.119, v_romantic=+0.017, v_impressionist=+1.430, accel_romantic(DD)=-0.101, accel_impressionist=+1.413

**INTERPRETATION**: The `v_epoch` columns are signed mean differences between consecutive eras. The `accel_romantic` column is the DD = v_romantic - v_classical. For pitch_range, the Classical era actually saw a *decrease* (v_classical=-5.14) while the Romantic era saw a dramatic *increase* (v_romantic=+20.00), yielding DD=+25.14. The Impressionist period then shows modest positive change (+2.78) but much less than the Romantic. The column headers in the CSV are: `feature,v_classical,p_classical,v_romantic,p_romantic,v_impressionist,p_impressionist,accel_romantic,accel_impressionist`.

### DDD Mode Effects
**CRITICAL**: DDD is defined as `DDD = a_Moll - a_Dur` (Minor minus Major) per the source document (`JuFoArbeit_RegionalRunde_v2.tex`, line 317).
- n_Major(Dur) = 97, n_Minor(Moll) = 47
- pitch_range DDD = +12.64 (a_Moll=+33.17, a_Dur=+20.53) → Minor-mode works show 12.64 semitones MORE Romantic acceleration than Major
- dissonance DDD = +0.271 (a_Moll=+0.482, a_Dur=+0.211) → Minor amplifies dissonance increase
- harmonic_density DDD = +0.428 (a_Moll=+1.011, a_Dur=+0.583) → Minor densifies texture more
- rhythmic_entropy DDD = +0.981 (a_Moll=+0.521, a_Dur=-0.460) → Most striking: Minor shows innovation while Major *stagnates* rhythmically
- **INTERPRETATION**: Positive DDD values mean **Minor-mode works** experienced greater Romantic acceleration than Major-mode works. The source document states: "die romantische Affekt-Intensivierung wurde primär über den Moll-Modus kanalisiert" — Romantic emotional intensification was primarily channelled through the minor mode. Composers used minor keys as experimental ground for harmonic and rhythmic boundary-pushing.

### Cluster Metrics (from `data/stats/cluster_metrics.json`)
- Silhouette Score: 0.010 (3D PCA), 0.018 (36D)
- Davies-Bouldin Index: 3.51 (3D), 4.10 (36D)
- **FRAMING**: These low scores MUST be presented as evidence of continuous stylistic evolution across eras, NOT as a failure of clustering. Musical style forms a continuum, not discrete clusters.

## Source Documents (German -> English synthesis)
- `/JuFoArbeit_RegionalRunde.tex` (v1): Full methodology, 36 features with formulas, ANOVA/PCA, Hisaishi/Ravel validation
- `/JuFoArbeit_RegionalRunde_v2.tex` (v2): Adds evolutionary coefficients (v, a), DD, DDD, "Romantic Reversal", mode effects

## Existing Bibliography (`references.bib` — ROOT directory)
Available keys: Benjamini1995, Cuthbert2010, PDMX2024, Simonetta2025, LinJeng1987, VonHippel2000, McKay2010, White2013, Hadjeres2018, DebussyCharacteristics, ChopinTransformations, JuFoGuide, IEEEStyle

**CRITICAL**: The bibliography is thin (~12 entries). It MUST be expanded with real, verifiable computational musicology references. **DO NOT hallucinate references.** Only add references that can be verified as real published works. When in doubt, omit rather than fabricate.

## Figure Paths (relative to JournalMusicResearch/ directory — use `../` prefix)

| Figure | Path (from JournalMusicResearch/) | Description |
|--------|-----------------------------------|-------------|
| Top ANOVA bar chart | `../figures/significance/top_anova_bar.png` | Top 15 significant features |
| Tukey pair heatmap | `../figures/significance/tukey_pair_heatmap.png` | Pairwise significance counts |
| Pitch range boxplot | `../figures/melodic/boxplot_pitch_range_semitones.png` | Per-composer pitch range |
| Dissonance boxplot | `../figures/harmonic/boxplot_dissonance_ratio.png` | Per-composer dissonance ratio |
| Rhythmic entropy boxplot | `../figures/rhythmic/boxplot_rhythmic_pattern_entropy.png` | Per-composer rhythmic entropy |
| 3D PCA clouds | `../figures/embeddings/composer_clouds_3d.png` | 3D Gaussian iso-surface PCA view |
| DDD comparison | `../figures/evolution/ddd_comparison.png` | Triple-difference bar chart |
| Hisaishi highlight | `../summerdayhighlight.png` | Hisaishi PCA projection |
| Ravel highlight | `../ravelhighlight_cropped.png` | Ravel PCA projection |

**NOTE**: Verify that `composer_clouds_3d.png` exists (there is definitely an HTML version; the PNG may need to be generated or a screenshot used). If a PNG doesn't exist, use the architecture pipeline figure or skip and note it.

## Template Structure (interact.cls requirements)

The `main.tex` already has the correct preamble set up. The agent must fill in content following this exact structure:

```latex
\articletype{RESEARCH ARTICLE}

% Anonymisation toggle: swap \maketitle and \author block
% When \blindreviewtrue: \maketitle appears BEFORE \author (hides author info)
% When \blindreviewfalse: \author appears BEFORE \maketitle (shows author info)

\title{...}
\author{
  \name{Victor Gurbani\textsuperscript{a}\thanks{CONTACT Victor Gurbani. Email: victor.gurbani@gmail.com. ORCID: 0009-0008-4571-5444}}
  \affil{\textsuperscript{a}Independent Researcher}
}
\maketitle

\begin{abstract} ... \end{abstract}  % <=200 words, unstructured
\begin{keywords} ... \end{keywords}   % 3-6 keywords, semicolon-separated

\section{Introduction}
\section{Literature review}
\section{Materials and methods}
  \subsection{Corpus curation}
  \subsection{Feature extraction}
    \subsubsection{Harmonic features}
    \subsubsection{Melodic features}
    \subsubsection{Rhythmic features}
  \subsection{Statistical analysis}
    \subsubsection{One-way ANOVA with FDR correction}
    \subsubsection{Tukey HSD post-hoc comparisons}
    \subsubsection{Principal component analysis}
  \subsection{Evolutionary dynamics}
    \subsubsection{Evolutionary coefficients}
    \subsubsection{Difference-in-differences}
    \subsubsection{Triple-difference analysis}
\section{Results}
  \subsection{Static analysis}
    \subsubsection{ANOVA and post-hoc findings}
    \subsubsection{PCA embedding space}
  \subsection{Dynamic analysis}
    \subsubsection{Evolutionary coefficients and the Romantic Reversal}
    \subsubsection{Mode-conditioned triple-difference}
\section{Discussion}
  \subsection{Interpreting the PCA axes}
  \subsection{Chopin as stylistic bridge}
  \subsection{Cluster metrics and the continuum hypothesis}
  \subsection{Validation case studies}
\section{Conclusion}
\section*{Disclosure statement}
\section*{Funding}
\bibliographystyle{apacite}
\bibliography{references}
```

## Formatting Rules (MANDATORY)

1. **British English**: Use "-ise" spellings (organise, analyse, recognise, characterise, summarise, specialise, emphasise, utilise, visualise, normalise, standardise, minimise, maximise, categorise, hypothesise). NEVER use "-ize" variants.
2. **Single quotation marks**: Use `\enquote{text}` or LaTeX single quotes `\lq text\rq` for quotations. NOT double quotes.
3. **No first person**: Use "This study demonstrates", "The pipeline extracts", NOT "I did", "We found".
4. **APA citations**: `\citep{key}` for parenthetical, `\citet{key}` for narrative. NEVER use `\cite{}` bare.
5. **SI units**: Use `\SI{}{}` from siunitx where applicable.
6. **Tables**: Use `\tbl{caption}{tabular}` macro, NOT standard `\caption` above tables.
7. **Figures**: Use `\subfloat[]{}` from subfig package for sub-figures.
8. **Cross-references**: Use `\cref{label}` for smart references ("Figure 1", "Table 2").

## Guardrails (from Metis analysis)

1. **Citation hallucination prevention**: Every new `.bib` entry must be a real, verifiable publication. Use only well-known references in computational musicology (Eerola & Toiviainen, Huron, Temperley, etc.) and verify titles match known works. When expanding the bibliography, prefer references already cited in the source documents or well-established works.
2. **Word count management**: Target ~1000 words for Introduction, ~2000 for Methods, ~1500 for Results, ~1500 for Discussion, ~500 for Conclusion. Check with `texcount` after each section.
3. **Compilation at every step**: Run `pdflatex -interaction=nonstopmode main.tex` from `JournalMusicResearch/` after every task. Fix errors before proceeding.
4. **Excluded content check**: After every writing task, run: `grep -iE "random forest|neural network|elton john|pop music|MLP|classification accuracy" JournalMusicResearch/main.tex` — must return no matches.
5. **British English check**: After final task, run: `grep -oiE "\b\w*(ize|ization)\b" JournalMusicResearch/main.tex` to catch American spellings (expect 0 matches outside of code/URLs).

## Task Dependency Graph

```
Task 1 (Setup) -> Task 2 (Intro + Lit Review) -> Task 3 (Methods) -> Task 4 (Results) -> Task 5 (Discussion + Conclusion) -> Task 6 (Final QA)
```

All tasks are strictly sequential — each section must flow from the previous one narratively, and LaTeX must compile at each step.

## Tasks

### Task 1: Setup Workspace and Bibliography
- **Category**: `writing`
- **Skills**: `[]`
- **Depends on**: None
- **What to do**:
  1. Copy `references.bib` from the repo root into `JournalMusicResearch/references.bib`
  2. Clean up the copied `.bib` file:
     - Remove German-language notes (e.g., `note = {Frühes Beispiel...}`)
     - Remove the `IEEEStyle` and `JuFoGuide` entries (not relevant for JNMR)
     - Ensure all entries have proper APA-compatible fields
  3. Expand the bibliography with ~10-15 additional **real, verified** references. Recommended additions (these are real works — verify titles before adding):
     - Eerola, T. & Toiviainen, P. (2004). MIDI Toolbox: MATLAB tools for music research. University of Jyvaskyla.
     - Temperley, D. (2007). Music and Probability. MIT Press.
     - Huron, D. (2006). Sweet Anticipation: Music and the Psychology of Expectation. MIT Press.
     - Lerdahl, F. & Jackendoff, R. (1983). A Generative Theory of Tonal Music. MIT Press.
     - Pearce, M. T. & Wiggins, G. A. (2012). Auditory expectation: The information dynamics of music perception and cognition. Topics in Cognitive Science, 4(4), 625-652.
     - Conklin, D. (2003). Music generation from statistical models. Proceedings of the AISB Symposium on Artificial Intelligence and Creativity in the Arts and Sciences.
     - Patel, A. D. (2008). Music, Language, and the Brain. Oxford University Press.
     - Cope, D. (2005). Computer Models of Musical Creativity. MIT Press.
     - Dannenberg, R. B. (1993). Music representation issues, techniques, and systems. Computer Music Journal, 17(3), 20-30.
     - Müllensiefen, D. & Frieler, K. (2007). Modelling experts' notions of melodic similarity. Musicae Scientiae, Discussion Forum 4A, 183-210.
     - Meyer, L. B. (1989). Style and Music: Theory, History, and Ideology. University of Chicago Press.
     - Kirlin, P. B. & Utgoff, P. E. (2008). A framework for automated Schenkerian analysis. Proceedings of ISMIR.
     **IMPORTANT**: Verify each reference is real before adding. If uncertain about exact details, omit the reference rather than risk hallucination. Use web search to verify if needed.
  4. Add the anonymisation toggle to `main.tex` preamble (after `\draftversionfalse`):
     ```latex
     % Anonymisation toggle for blind review
     \newif\ifblindreview
     \blindreviewfalse % Set to \blindreviewtrue for anonymous submission
     ```
  5. Set up the document body skeleton in `main.tex`:
     - Set the title: "Quantifying stylistic evolution in Western piano music: A computational analysis of Bach, Mozart, Chopin, and Debussy"
     - Set up author block with blind review toggle:
       ```latex
       \ifblindreview
         \title{Quantifying stylistic evolution...}
         \maketitle
         \author{\name{[Redacted for blind review]}\affil{[Redacted]}}
       \else
         \title{Quantifying stylistic evolution...}
         \author{
           \name{Victor Gurbani\textsuperscript{a}\thanks{CONTACT Victor Gurbani. Email: victor.gurbani@gmail.com. ORCID: 0009-0008-4571-5444}}
           \affil{\textsuperscript{a}Independent Researcher}
         }
         \maketitle
       \fi
       ```
     - Write the abstract (<=200 words):
       The abstract should summarise: (1) the research question (quantifying stylistic evolution computationally), (2) the method (36 features, 144 scores, ANOVA/PCA + DD/DDD), (3) key findings (29/36 features significant, PCA reveals stylistic continuum, Romantic Reversal in DD analysis, mode conditioning in DDD), (4) validation (Hisaishi and Ravel projections confirm framework validity).
     - Write 3-6 keywords: `computational musicology; stylistic analysis; principal component analysis; difference-in-differences; music21; piano music`
     - Add all section headings as empty sections (placeholder structure)
     - Set up Disclosure and Funding sections:
       ```latex
       \section*{Disclosure statement}
       The author reports there are no competing interests to declare.
       
       \section*{Funding}
       This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.
       ```
     - Configure bibliography: `\bibliographystyle{apacite}` and `\bibliography{references}`
  6. Verify compilation: `cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex`
- **QA Scenarios**:
  - `ls JournalMusicResearch/references.bib` succeeds
  - `pdflatex` compiles without fatal errors from `JournalMusicResearch/`
  - `grep -c "@" JournalMusicResearch/references.bib` shows >= 15 entries
  - The abstract word count is <= 200 words

### Task 2: Write Introduction and Literature Review
- **Category**: `writing`
- **Skills**: `[]`
- **Depends on**: Task 1
- **What to do**:
  1. Read the source documents for context:
     - Read `/JuFoArbeit_RegionalRunde.tex` sections 1-2 (Introduction and Related Work)
     - Read `/JuFoArbeit_RegionalRunde_v2.tex` sections 1-2
  2. Write `\section{Introduction}` (~600-800 words):
     - Open with the broad question: Can musical style be quantified computationally?
     - Establish the four composers as representatives of major Western art music periods (Baroque, Classical, Romantic, Impressionist)
     - State the research gap: Most computational musicology studies focus on classification accuracy rather than understanding stylistic evolution dynamics
     - Present the contribution: (a) a comprehensive 36-feature extraction pipeline, (b) static significance mapping via ANOVA/PCA, (c) dynamic evolutionary modelling via DD/DDD, (d) empirical validation via external case studies
     - Briefly preview the structure of the paper
     - Cite: `\citet{Cuthbert2010}` for music21, `\citet{PDMX2024}` for the dataset, `\citet{Simonetta2025}` for the survey, `\citet{LinJeng1987}` for the closest prior work
  3. Write `\section{Literature review}` (~400-600 words):
     - Review prior computational musicology approaches: jSymbolic (`\citep{McKay2010}`), statistical corpus studies (`\citep{White2013}`), information-theoretic approaches (`\citep{Pearce2012}` if added)
     - Review music-theoretic foundations: pitch proximity (`\citep{VonHippel2000}`), tonal hierarchy, Lerdahl & Jackendoff if added
     - Identify the specific gap this work fills: no prior study combines static significance testing with dynamic evolutionary modelling (DD/DDD) on a balanced symbolic corpus
     - Maintain British English, academic tone, no first person
  4. Verify: `cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex`
  5. Check word count: `texcount -1 -sum JournalMusicResearch/main.tex`
- **QA Scenarios**:
  - Document compiles without fatal errors
  - No excluded terms present (`grep -iE "random forest|neural network|elton john" JournalMusicResearch/main.tex` returns nothing)
  - Running word count shows progress toward 6000-8000 target

### Task 3: Write Materials and Methods
- **Category**: `writing`
- **Skills**: `[]`
- **Depends on**: Task 2
- **What to do**:
  1. Read the source documents for detailed methodology:
     - Read `/JuFoArbeit_RegionalRunde.tex` sections 3-4 (Corpus and Features)
     - Read `/JuFoArbeit_RegionalRunde_v2.tex` sections 3-5 (adds evolutionary formulas)
     - Read `data/stats/anova_summary.csv` for feature names
     - Read `data/stats/pca_loadings.csv` for PCA details
     - Read `data/stats/evolution_coefficients.csv` for exact formulas and values
  2. Write `\subsection{Corpus curation}` (~300 words):
     - PDMX dataset: 254,077 entries filtered to 144 solo piano works
     - Balanced design: 36 works per composer
     - Selection criteria: solo piano, public domain, score quality
     - Cite `\citet{PDMX2024}`
  3. Write `\subsection{Feature extraction}` (~500-600 words) with three subsubsections:
     - **Harmonic features** (16): chord_quality_major_pct, chord_quality_minor_pct, chord_quality_diminished_pct, chord_quality_augmented_pct, chord_quality_other_pct, chord_quality_total, chord_event_count, roman_chord_count, dissonance_ratio, dissonant_note_count, harmonic_density_mean, passing_tone_ratio, appoggiatura_ratio, other_dissonance_ratio, modal_interchange_ratio, deceptive_cadence_ratio
     - **Melodic features** (11): pitch_range_semitones, pitch_class_entropy, avg_melodic_interval, melodic_interval_std, melodic_leap_ratio, conjunct_motion_ratio, note_count, note_event_count, avg_note_duration, voice_independence_index, (check if there are 11 — read the harmonic/melodic/rhythmic feature CSVs to get the authoritative list)
     - **Rhythmic features** (9): rhythmic_pattern_entropy, std_note_duration, notes_per_beat, syncopation_ratio, downbeat_emphasis_ratio, avg_note_duration (if rhythmic), micro_rhythmic_density, cross_rhythm_ratio, (check the rhythmic features CSV)
     - **IMPORTANT**: The authoritative feature names MUST be derived from the actual feature CSV files at `data/features/harmonic_features.csv`, `data/features/melodic_features.csv`, `data/features/rhythmic_features.csv`. Read these files' column headers to get the exact feature names and groupings. The ANOVA CSV (`data/stats/anova_summary.csv`) has a `source` column indicating which group each feature belongs to (harmonic/melodic/rhythmic). Use this for the definitive grouping.
     - Describe the music21 pipeline: Score parsing -> Part extraction -> Feature computation per score
     - Cite `\citet{Cuthbert2010}` for music21
     - Include a table summarising the 36 features grouped by category, using `\tbl{}{}` macro
  4. Write `\subsection{Statistical analysis}` (~400 words) with three subsubsections:
     - **ANOVA with FDR**: One-way ANOVA per feature, Benjamini-Hochberg FDR correction (`\citep{Benjamini1995}`), significance threshold q < 0.05
     - **Tukey HSD**: Post-hoc pairwise comparisons for all 6 composer pairs across significant features
     - **PCA**: StandardScaler normalisation, 6 count features excluded, 30 features retained, 3 components extracted (48.2% cumulative variance), seed=42
  5. Write `\subsection{Evolutionary dynamics}` (~400-500 words) with three subsubsections:
     - **Evolutionary coefficients**: Define v_epoch as the mean difference between consecutive eras. Formula: $v_{\text{epoch}} = \bar{x}_{\text{later}} - \bar{x}_{\text{earlier}}$ for each transition (Baroque->Classical, Classical->Romantic, Romantic->Impressionist)
     - **Stylistic acceleration**: $a = v_{\text{romantic}} - v_{\text{classical}}$ measuring whether the Romantic period accelerated or decelerated change
     - **Difference-in-differences**: DD = v_romantic - v_classical, quantifying the "Romantic Reversal" (large positive DD for pitch_range and dissonance, negative for rhythmic_entropy)
     - **Triple-difference**: DDD = a_Moll - a_Dur (Minor minus Major), conditioning on mode (Major vs Minor) to isolate mode-specific evolutionary trajectories. Positive DDD means minor-mode works experienced greater Romantic acceleration. n_Major(Dur)=97, n_Minor(Moll)=47.
     - Include the mathematical formulas in equation environments
  6. Verify compilation and excluded terms
- **QA Scenarios** (all must be executable commands):
  - `cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex` exits without fatal errors
  - `grep -c "pitch_range_semitones\|dissonance_ratio\|harmonic_density_mean\|rhythmic_pattern_entropy" JournalMusicResearch/main.tex` returns >= 4 (key features mentioned)
  - `grep -c "\\\\begin{equation}" JournalMusicResearch/main.tex` returns >= 2 (formulas present)
  - `grep -iE "random forest|neural network|elton john|pop music|MLP|classification accuracy" JournalMusicResearch/main.tex` returns exit code 1 (no excluded terms)
  - `texcount -1 -sum JournalMusicResearch/main.tex` shows word count progressing (should be ~3500-4500 after this task)

### Task 4: Write Results Section
- **Category**: `writing`
- **Skills**: `[]`
- **Depends on**: Task 3
- **What to do**:
  1. Read the verified data files for exact numbers:
     - `data/stats/anova_summary.csv` — exact p-values and F-statistics
     - `data/stats/tukey_hsd.csv` — exact meandiff values and significance
     - `data/stats/pca_loadings.csv` — exact PC loadings
     - `data/stats/evolution_coefficients.csv` — exact v and DD values
     - `data/stats/cluster_metrics.json` — silhouette and Davies-Bouldin
  2. Write `\subsection{Static analysis}` (~700-800 words):
     - **ANOVA and post-hoc findings** (~400 words):
       - 29 of 36 features significant after FDR (q < 0.05)
       - Highlight top features by FDR-corrected p-value (**always use `p_value_fdr` column, NOT raw `p_value`**): pitch_range_semitones (F=42.18, p_fdr=6.32e-18), dissonance_ratio (F=30.65, p_fdr=4.85e-14), pitch_class_entropy (F=21.79, p_fdr=1.47e-10)
       - **Read exact values from `data/stats/anova_summary.csv`** for all numbers. The columns are: feature, source, groups_tested, min_group_size, max_group_size, f_statistic, p_value, significant, alpha_threshold, significant_alpha, bonferroni_threshold, p_value_bonferroni, significant_bonferroni, p_value_fdr, significant_fdr
       - Include figure: `../figures/significance/top_anova_bar.png` — top 15 ANOVA features
       - Discuss Tukey HSD patterns: which composer pairs differ most? (Bach-Debussy, Bach-Chopin tend to show largest differences)
       - Include figure: `../figures/significance/tukey_pair_heatmap.png`
       - Optionally include 1-2 boxplots (pitch_range, dissonance_ratio) showing per-composer distributions
     - **PCA embedding space** (~300-400 words):
       - PC1 (22.3%): loaded by harmonic complexity features (dissonance_ratio, unique_chord_ratio)
       - PC2 (16.1%): loaded by melodic range features (pitch_range_semitones, pitch_std_dev)
       - PC3 (9.8%): loaded by rhythmic features (rhythmic_pattern_entropy)
       - Include figure: `../figures/embeddings/composer_clouds_3d.png` (or note if PNG unavailable)
       - Describe the visual separation: Bach cluster tight and low-complexity; Debussy diffuse and high-dissonance; Chopin overlapping with both Mozart and Debussy
  3. Write `\subsection{Dynamic analysis}` (~700-800 words):
     - **Evolutionary coefficients and the Romantic Reversal** (~400-500 words):
       - Present the evolutionary coefficients table (use `\tbl{}{}` macro). **Read exact values from `data/stats/evolution_coefficients.csv`**:
         | Feature | v_classical | v_romantic | v_impressionist | DD (accel_romantic) |
         | pitch_range_semitones | -5.14 | +20.00 | +2.78 | +25.14 |
         | dissonance_ratio | -0.086 | +0.229 | +0.048 | +0.315 |
         | harmonic_density_mean | -0.253 | +0.500 | +0.322 | +0.754 |
         | rhythmic_pattern_entropy | +0.119 | +0.017 | +1.430 | -0.101 |
       - Interpret the "Romantic Reversal": The Romantic era shows dramatically accelerated change in pitch range, harmonic density, and dissonance, followed by a partial retrenchment in the Impressionist era
       - Include figure: `../figures/evolution/ddd_comparison.png`
     - **Mode-conditioned triple-difference** (~300 words):
       - Present DDD values: pitch_range DDD = +12.64, dissonance DDD = +0.271, harmonic_density DDD = +0.428, rhythmic_entropy DDD = +0.981
       - **CRITICAL**: DDD = a_Moll - a_Dur (Minor minus Major). Positive DDD means **minor-mode works** experienced greater Romantic acceleration. NOT major-mode.
       - Interpretation: The Romantic acceleration was significantly more pronounced in **minor-mode** compositions. Minor keys served as experimental ground for harmonic and rhythmic boundary-pushing.
       - Most striking finding: For rhythmic entropy, Major-mode works actually *stagnated* during the Romantic era (a_Dur = -0.460) while Minor-mode works innovated (a_Moll = +0.521), yielding DDD = +0.981
       - n_Major(Dur) = 97, n_Minor(Moll) = 47
  4. Verify compilation and excluded terms
- **QA Scenarios** (all must be executable commands):
  - `cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex` exits without fatal errors (all figure paths with `../` resolve)
  - `grep -c "p_\\|p\\\\textless\|p.*fdr\|p.*<" JournalMusicResearch/main.tex` returns >= 3 (p-values reported in Results)
  - `grep -c "\\\\includegraphics" JournalMusicResearch/main.tex` returns >= 4 (figures included)
  - `grep -iE "random forest|neural network|elton john|pop music" JournalMusicResearch/main.tex` returns exit code 1
  - `texcount -1 -sum JournalMusicResearch/main.tex` shows ~5000-6000 words

### Task 5: Write Discussion and Conclusion
- **Category**: `writing`
- **Skills**: `[]`
- **Depends on**: Task 4
- **What to do**:
  1. Read the source documents' discussion sections for interpretive context
  2. Read `data/stats/cluster_metrics.json` for silhouette/Davies-Bouldin values
  3. Write `\subsection{Interpreting the PCA axes}` (~200-300 words):
     - PC1 as a "harmonic complexity" axis (Baroque simplicity -> Impressionist complexity)
     - PC2 as a "melodic expressiveness" axis (Classical restraint -> Romantic breadth)
     - PC3 as a "rhythmic regularity" axis
  4. Write `\subsection{Chopin as stylistic bridge}` (~200-300 words):
     - Chopin's PCA cloud overlaps with both Mozart and Debussy
     - This is consistent with musicological understanding: Chopin bridged Classical form with Romantic harmonic innovation
     - The overlap is not noise — it reflects genuine stylistic hybridity
  5. Write `\subsection{Cluster metrics and the continuum hypothesis}` (~300-400 words):
     - **CRITICAL FRAMING**: The silhouette score of 0.010 (3D) and 0.018 (36D) is low
     - This does NOT indicate failure — it indicates that musical style is a continuum, not discrete clusters
     - Davies-Bouldin index of 3.51 confirms overlapping style distributions
     - This is the **expected** result for a historically continuous art form
     - The low cluster separation is itself a finding: it empirically demonstrates that stylistic evolution is gradual, not punctuated
     - Contrast with the strong ANOVA significance: features ARE statistically different between composers, but the full multivariate space shows overlap — composers share many features while differing on specific ones
  6. Write `\subsection{Validation case studies}` (~300-400 words):
     - **Hisaishi** (One Summer's Day): Projects between Chopin and Debussy clouds, slightly nearer Chopin. Consistent with the piece's blend of Romantic harmony and Impressionist texture.
       - Include figure: `../summerdayhighlight.png`
     - **Ravel** (String Quartet): Projects within the Debussy cloud. Expected given Ravel's Impressionist roots, though with some unique features.
       - Include figure: `../ravelhighlight_cropped.png`
     - Optionally mention Satie if it adds analytical value (the agent should read the source documents for any Satie data; if no data exists, skip)
     - These external validations confirm the PCA space captures meaningful stylistic relationships
  7. Write `\section{Conclusion}` (~300-500 words):
     - Summarise the three-level contribution: (a) 36-feature extraction pipeline, (b) static ANOVA/PCA mapping, (c) dynamic DD/DDD modelling
     - Key finding: The "Romantic Reversal" — Chopin-era compositions show dramatically accelerated stylistic change compared to the Classical and Impressionist transitions
     - Key finding: Mode conditioning (DDD) reveals that **minor-mode** works drove the Romantic acceleration more than major-mode works — composers used minor keys as experimental ground for harmonic and rhythmic innovation
     - Limitations: (1) balanced but small corpus (36 per composer), (2) single instrument (piano), (3) features capture surface statistics not deep structural relationships, (4) evolutionary model assumes linear transitions between eras
     - Future work: Expand corpus, add more composers/eras, incorporate structural features (form, phrase), apply to non-Western traditions
  8. Ensure Disclosure and Funding sections are present (should be from Task 1)
  9. Verify compilation and excluded terms
- **QA Scenarios** (all must be executable commands):
  - `cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex` exits without fatal errors
  - `grep -i "silhouette" JournalMusicResearch/main.tex` returns matches (silhouette score is discussed)
  - `grep -i "continuum\|continuous\|gradual" JournalMusicResearch/main.tex` returns matches (positive framing present)
  - `grep -c "\\\\includegraphics" JournalMusicResearch/main.tex` returns >= 6 (all figures including Hisaishi/Ravel)
  - `grep -iE "random forest|neural network|elton john|pop music" JournalMusicResearch/main.tex` returns exit code 1
  - `texcount -1 -sum JournalMusicResearch/main.tex` shows ~6500-8000 words

### Task 6: Final QA, Formatting, and Verification
- **Category**: `quick`
- **Skills**: `[]`
- **Depends on**: Task 5
- **What to do**:
  1. Run full LaTeX compilation (2 passes for references):
     ```bash
     cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
     ```
  2. Check word count:
     ```bash
     texcount -1 -sum JournalMusicResearch/main.tex
     ```
     Must be between 6000-8000 words. If under, expand Discussion or add detail to Methods. If over, trim Conclusion or condense Literature Review.
  3. Check excluded terms:
     ```bash
     grep -iE "random forest|neural network|elton john|pop music|MLP|classification accuracy" JournalMusicResearch/main.tex
     ```
     Must return NO results.
  4. Check British English:
     ```bash
     grep -oiE "\b\w*(ize|ization)\b" JournalMusicResearch/main.tex
     ```
     Must return no matches outside of bibliography entries, code listings, or proper nouns. Fix any American spellings found (organize->organise, analyze->analyse, etc.).
  5. Check for first-person pronouns:
     ```bash
     grep -iE "\b(I |we |my |our )\b" JournalMusicResearch/main.tex
     ```
     Must return no matches in body text (may appear in bib entries or author info — those are acceptable).
  6. Verify all `\cref` and `\label` cross-references resolve (check LaTeX log for undefined references)
  7. Verify all `\citep` and `\citet` references resolve (check for undefined citation warnings)
  8. If any issues found, fix them and re-compile
- **QA Scenarios**:
  - Final PDF compiles cleanly (no fatal errors, minimal warnings)
  - Word count in 6000-8000 range
  - Zero excluded terms
  - Zero American English spellings in body text
  - Zero first-person pronouns in body text
  - All cross-references and citations resolve

## Final Verification Wave

**IMPORTANT**: After all tasks complete, the agent must present a summary of:
1. Final word count
2. Number of figures included
3. Number of tables included  
4. Number of bibliography entries
5. Any LaTeX warnings remaining
6. Confirmation that excluded topics are absent
7. Confirmation that British English is consistent

**Do NOT mark work as complete until the user explicitly confirms the output is acceptable.**

## Commit Strategy

- After Task 1: `feat(jnmr): set up LaTeX skeleton, bibliography, and anonymisation toggle`
- After Task 2: `feat(jnmr): write Introduction and Literature Review`
- After Task 3: `feat(jnmr): write Materials and Methods section`
- After Task 4: `feat(jnmr): write Results section with figures`
- After Task 5: `feat(jnmr): write Discussion and Conclusion`
- After Task 6: `fix(jnmr): final QA — British English, word count, compilation`
