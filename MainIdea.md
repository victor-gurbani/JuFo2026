## Project: "The Algorithmic Fingerprint: A Quantitative Analysis of Compositional Styles"

### 1. Project Goal

To determine if composers from four distinct musical eras (e.g., Baroque, Classical, Romantic, Impressionist) can be accurately distinguished using a set of quantifiable mathematical and statistical features extracted directly from their musical scores.

### 2. Research Question & Hypothesis

* **Question:** Do distinct compositional styles and eras leave a measurable "mathematical fingerprint" in musical data?
* **Hypothesis:** Yes. Composers like Bach (Baroque), Mozart (Classical), Chopin (Romantic), and Debussy (Impressionist) will show statistically significant, progressive trends in features related to harmony, melody, and rhythm, allowing an algorithm to classify them with high accuracy.

### 3. Technical Stack

* **Language:** Python 3.x
* **Core Libraries:**
    * **music21:** The core library for parsing musical files. **[OPTIMIZED]** The project will focus exclusively on **MusicXML**, as this format is the only one that provides the necessary semantic information (e.g., the enharmonic difference between G-sharp and A-flat) required for deep musicological analysis.
    * **pandas:** **[OPTIMIZED]** Used for two critical phases: 1) Programmatically filtering and curating the PDMX metadata catalog to build a high-quality corpus, and 2) Organizing the extracted features into a clean DataFrame.
    * **numpy:** For numerical computations.
    * **matplotlib & seaborn:** For data visualization (box plots, histograms).
    * **scipy.stats:** For statistical analysis (**[NEW]** e.g., **ANOVA** to compare the four groups).
    * **scikit-learn:** (Optional, for Phase 3) For building the ML classification model.

---

### Phase 1: Data Collection & Processing

This phase is strategically redesigned for maximum methodological rigor.

**1. The Dataset**
* **Source: [OPTIMIZED] PDMX Dataset.** Instead of a manual collection from various sources (like Mutopia), we will use the "PDMX: A Large-Scale Public Domain MusicXML Dataset." This is a massive but "messy" corpus. The scientific value comes from *programmatically filtering* this dataset.
* **Selection: [UPDATED] "Stylistic Evolution" (Depth) Approach.** We will select four composers representing four distinct eras. To ensure a fair comparison, we will **only use solo piano works** from each.
    1.  **Baroque:** Johann Sebastian Bach
    2.  **Classical:** Wolfgang Amadeus Mozart
    3.  **Romantic:** Frédéric Chopin
    4.  **Impressionist:** Claude Debussy
* **Processing Step 1: Corpus Curation [OPTIMIZED]**
    * Load the PDMX metadata CSV file into a `pandas` DataFrame.
    * Programmatically filter this DataFrame to create a high-quality, balanced sub-corpus.
    * **Filter Criteria:** `composer` (matches one of our four); `instrument == 'Solo Piano'` (or equivalent); `is_original == True` (to exclude arrangements); `user_rating >= 4.0` (or similar, to remove bad scores).
    * **Result:** A clean list of file paths to high-quality, relevant MusicXML files, balanced as evenly as possible across the four composers.

* **Processing Step 2: Parsing**
    * Write a Python script (the "parser").
    * This script iterates *only* through the filtered list of MusicXML files.
    * For each file, it uses `music21.converter.parse()` to load the **MusicXML data** into a `music21.stream.Score` object.

---

### Phase 2: Feature Extraction & Analysis

This is the core of the project. We will expand the feature list to include "Level 3" (musicologically-informed) metrics, which demonstrate a higher degree of novelty.

**A. Features to Extract**

#### 1. Harmonic Features:
* **(Level 1) Chord Quality:** (As you suggested) Percentage of major, minor, diminished, and augmented chords.
* **(Level 1) Harmonic Density:** (As you suggested) Average number of notes per "chordified" chord.
* **(Level 2) Harmonic Dissonance:** (As you suggested) `dissonance_ratio` (ratio of chords containing at least one dissonant interval).
* **[NEW] (Level 3) Dissonance Profile:** An extension of L2. We classify the *function* of the dissonance.
    * **How:** Analyze non-chord tones in their melodic context.
    * **Feature:** `passing_tone_ratio` (dissonances resolved by step) vs. `appoggiatura_ratio` (dissonances landing on a strong beat).
    * **Hypothesis:** The `appoggiatura_ratio` and other "expressive" dissonances will increase from Bach (low) to Chopin and Debussy (high).
* **[NEW] (Level 3) Harmonic Progression:**
    * **How:** Use `music21` to translate chords into Roman Numerals (`roman.RomanNumeral`). Analyze the N-grams (pairs) of chords.
    * **Feature:** `deceptive_cadence_ratio` (e.g., ratio of V-vi progressions) or `modal_interchange_ratio`.
    * **Hypothesis:** Harmonic complexity and non-standard progressions will increase significantly after the Classical era.

#### 2. Melodic Features:
* **(Level 1) Pitch Range (Ambitus):** (As you suggested) `pitch_range_semitones`.
* **(Level 2) Melodic Motion:** (As you suggested) `avg_melodic_interval` (leap size) and `conjunct_motion_ratio` (ratio of stepwise motion).
* **(Level 2) Pitch Class Distribution:** (As you suggested) `pitch_class_entropy` (how evenly are the 12 tones used?).
    * **Hypothesis:** Entropy will increase from Bach (strongly tonal) to Debussy (who uses whole-tone and pentatonic scales).
* **[NEW] (Level 3) Contrapuntal Independence:**
    * **How:** Isolate the highest (soprano) and lowest (bass) voices in the piano works. Calculate the correlation of their melodic contours (direction of movement) over a sliding window.
    * **Feature:** `voice_independence_index` (A value near -1 means high contrary motion; +1 means high parallel motion).
    * **Hypothesis:** Bach will show the highest independence (most contrary motion), which will decrease through the eras.

#### 3. Rhythmic Features:
* **(Level 1) Note Duration Distribution:** (As you suggested) `avg_note_duration` and `std_dev_note_duration` (rhythmic variety).
* **(Level 2) Rhythmic Density:** (As you suggested) `notes_per_beat`.
* **[NEW] (Level 3) Syncopation Index:**
    * **How:** Measure how often notes are tied over strong beats or begin on weak off-beats and hold into strong beats.
    * **Feature:** `syncopation_ratio` (ratio of syncopated notes to all notes).
    * **Hypothesis:** This will be low for Bach/Mozart and much higher for Chopin/Debussy.

**B. Analysis & Visualization** (Method updated for 4 groups)
1.  **Descriptive Statistics:** Calculate mean, median, and std. dev. for every feature, grouped by composer.
2.  **Visualization:** **Box plots** are the most crucial tool. Create one for each feature (e.g., `dissonance_ratio`) with the four composers on the x-axis. This will visually demonstrate the trend across eras.
3.  **Significance Testing: [UPDATED]**
    * **How:** Since you have more than two groups, you will use an **ANOVA (Analysis of Variance)** test (e.g., `scipy.stats.f_oneway()`) for each feature. This tests if there is a significant difference *somewhere* among the four composers.
    * **Follow-up:** If the ANOVA p-value is significant (p < 0.05), you can run a **Post-Hoc Test** (like a Tukey HSD test) to find out *which specific pairs* are different (e.g., "Bach-Mozart was not significant, but Bach-Chopin was").
    * **The JuFo Sentence:** (Perfect) "An ANOVA revealed a statistically significant difference in mean dissonance ratios across the four eras, F(3, 156) = 22.1, p < .001. Post-hoc tests confirmed that Debussy's scores (M=0.45) were significantly more dissonant than Bach's (M=0.12)."

---

### Phase 3: Machine Learning Classification (The "Wow" Extra)

(Updated for multi-class classification)
This is the "wow" factor that solidifies your findings.

1.  **Model Selection:** `LogisticRegression` (with `multi_class='ovr'`) or `SVC`. These are interpretable, which is better than a black box (like a neural net) for this project.
2.  **Training:** Your DataFrame is the input. (X = all your feature columns, y = `composer_label` [0=Bach, 1=Mozart, 2=Chopin, 3=Debussy]). Use `train_test_split()` (e.g., 80% train, 20% test).
3.  **Evaluation:**
    * Process: Use the trained model to predict on the 20% of data it has never seen: `predictions = model.predict(X_test)`.
    * Metrics: Generate a `classification_report` and a **4x4 confusion_matrix**.
4.  **Result (The "Money" Slide):** "The model was able to predict the composer of an unknown piece with X% accuracy."
5.  **Bonus (The Core Insight):**
    * Show the **4x4 confusion matrix (with younger and older composers (same composer))**. This is your best result. It will visually show, for example, that the model *never* confused Bach and Debussy, but *did* confuse Mozart and Bach 10% of the time. This is a powerful musicological insight.
    * Show the `model.feature_importances_` (or `model.coef_`) to prove *which* of your features (e.g., `pitch_class_entropy` and `dissonance_profile`) were the most powerful predictors. This links the ML result directly back to your research question.

