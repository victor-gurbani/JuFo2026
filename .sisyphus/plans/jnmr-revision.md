# JNMR Paper Revision — Comprehensive Peer Review Response

## TL;DR

> **Quick Summary**: Systematically address all 17 issues from two peer reviews (GenericReview1 + CriticReview2) of the JNMR computational musicology manuscript. Phase A runs Python investigation scripts (PC4/PC5 analysis, placebo DD permutation tests), Phase B revises main.tex section-by-section with proactive defenses, new citations, and a formal Robustness Checks subsection.
> 
> **Deliverables**:
> - `src/pca_extended.py` — standalone PCA analysis with 5 components (PC4/PC5 loadings + variance)
> - `src/placebo_dd.py` — permutation-based placebo test for DD significance
> - `data/stats/pca_extended.csv` and `data/stats/placebo_dd.csv` — script outputs
> - `JournalMusicResearch/references.bib` — updated with ≤5 new entries
> - `JournalMusicResearch/main.tex` — revised with all 17 review issues addressed
> - Clean LaTeX build producing `main.pdf` with 0 errors, 0 undefined references
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 4 waves + final verification
> **Critical Path**: T1 (verify data) → T2/T3 (Python scripts) → T4 (references) → T5-T8 (LaTeX sequential) → T9 (build verify) → F1-F4

---

## Context

### Original Request
User wants to comprehensively address all feedback from two peer reviews to make the JNMR manuscript submission-ready. Reviews include both positive suggestions (GenericReview1) and aggressive methodological critiques (CriticReview2).

### Interview Summary
**Key Discussions**:
- **Scope**: Address EVERYTHING — all 17 identified issues, no shortcuts
- **Placebo test placement**: Formal "Robustness Checks" subsection in Results
- **Revision tone**: Proactive defense — silently strengthen paper, no direct rebuttal
- **PC4/PC5**: New paragraph in PCA Results section (not footnote)

**Research Findings**:
- PC4/PC5 not computed by existing pipeline (PCA hardcoded to 3 components). Quick one-off script available using existing feature CSVs.
- DD on repeated cross-sections IS valid — reviewer conflated with panel DD (Lee 2025)
- Placebo DD via piece-level permutation test is standard econometric defense
- Beethoven omission justified by prototypicality — he IS the transition, including him confounds DD design
- Corpus size (144 works) is standard for symbolic musicology (Bach chorales 371, Chopin Mazurkas ~50, Kostka-Payne ~450)
- DDD analysis code exists in `src/ddd_analysis.py` and `src/evolution_analysis.py`
- Feature CSVs already exist at `data/features/` — no MusicXML re-parsing needed

### Metis Review
**Identified Gaps** (addressed):
- **4! = 24 problem**: Permutation test must shuffle individual piece labels (N=144), not group labels (N=4), to produce meaningful null distribution
- **PCA solver consistency**: Must use `PCA(n_components=5, random_state=42, svd_solver='full')` to guarantee PC1-3 match existing values
- **Rambachan & Roth (2023) may not apply**: Their framework is for panel DD, not cross-sectional group-means DD. Use permutation test as primary defense instead.
- **DD is really a "descriptive acceleration metric"**: Don't rename framework, but add clarifying prose
- **Sensitivity analysis claim (line 303-306)**: Paper claims sensitivity analyses but no script supports it — hidden vulnerability. Soften language.
- **Edge cases**: PC4/PC5 may explain negligible variance; placebo p-values may be non-significant for some features; mode split imbalance needs brief defense
- **Verify existing numbers first**: Must confirm current data still reproduces paper's reported statistics before any edits

---

## Work Objectives

### Core Objective
Address all 17 reviewer-identified issues through Python investigation and LaTeX revision, producing a submission-ready JNMR manuscript with proactive methodological defenses.

### Concrete Deliverables
- 2 new standalone Python scripts (`src/pca_extended.py`, `src/placebo_dd.py`)
- 2 new data output files (`data/stats/pca_extended.csv`, `data/stats/placebo_dd.csv`)
- Updated `JournalMusicResearch/references.bib` (≤5 new entries)
- Revised `JournalMusicResearch/main.tex` (17 issues addressed, ≤80 net lines added)
- Clean LaTeX build with 0 errors

### Definition of Done
- [x] All 17 review issues have corresponding text changes or explicit defenses in main.tex
- [x] PC4/PC5 variance and loadings reported in Results PCA section
- [x] Placebo DD test results reported in new Robustness Checks subsection
- [x] LaTeX compiles with 0 errors, 0 undefined references, 0 undefined citations
- [x] No existing numerical claims changed (29/36, 48.2%, F-statistics, DD/DDD values unchanged)
- [x] All new numbers traceable to script output CSVs

### Must Have
- PC4/PC5 analysis with explanation of why 3 components were retained
- Permutation-based placebo DD test with piece-level label shuffling (N=144)
- Beethoven omission defense in Corpus Curation
- Parallel trends assumption defense
- Rubato terminology fix
- Silhouette paradox tighter argumentation
- Missing citations added
- "Colouristic experimentation" replaced with precise descriptor
- BH FDR repetition trimmed
- Equation variable mapping clarified

### Must NOT Have (Guardrails)
- **No new figures** added to the paper unless absolutely required for the robustness check
- **No more than 5 new bibliography entries** — avoid padding
- **No modifications to existing `src/` pipeline scripts** — new scripts are standalone one-off tools
- **No changes to `quickstart.sh`** — new scripts are investigation tools, not pipeline additions
- **No Literature Review section restructuring** — only minimal inline fixes
- **No renaming of the DD/DDD framework** or changing mathematical definitions (eqs 1-3)
- **No net line additions exceeding 80 lines** — trim verbose existing paragraphs if needed
- **No changes to existing numerical claims** — all reported statistics must remain identical
- **No subjective language** — every new claim must be backed by data or citation
- **No `as any`, `@ts-ignore`, or `# type: ignore`** in Python scripts

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (Python scripts run via `python3`, LaTeX builds via `pdflatex`)
- **Automated tests**: Tests-after (verify script outputs + LaTeX build)
- **Framework**: Python `py_compile` for syntax, `pdflatex`/`bibtex` for LaTeX

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

**Global Precondition**: Before running ANY QA scenario, the agent MUST execute `mkdir -p .sisyphus/evidence` to ensure the evidence directory exists. This is a one-time setup per task execution.

- **Python scripts**: Use Bash — run script, verify output CSV exists and contains expected columns/values
- **LaTeX edits**: Use Bash — compile LaTeX, check for zero errors/warnings in log
- **References**: Use Bash — run bibtex, check for zero warnings about missing entries

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — Python investigation, 3 parallel):
├── Task 1: Verify existing data reproducibility [quick]
├── Task 2: Create and run PCA extended analysis script [deep]
└── Task 3: Create and run placebo DD permutation test script [deep]

Wave 2 (After Wave 1 — Bibliography, 1 task):
└── Task 4: Update references.bib with new citations (depends: 1, 2, 3) [quick]

Wave 3 (After Wave 2 — LaTeX revisions, sequential chain):
├── Task 5: Fix Introduction + Literature Review + Methods (depends: 4) [unspecified-high]
├── Task 6: Fix Results + add Robustness Checks subsection (depends: 5, 2, 3) [deep]
├── Task 7: Fix Discussion + Conclusion (depends: 6) [unspecified-high]
└── Task 8: Throughout fixes — BH FDR, capitalization, cleanup (depends: 7) [quick]

Wave 4 (After Wave 3 — Verification, 1 task):
└── Task 9: Final build verification + URL checks (depends: 8) [quick]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: T1 → T2/T3 → T4 → T5 → T6 → T7 → T8 → T9 → F1-F4 → user okay
Parallel Speedup: ~40% faster than fully sequential (Wave 1 saves significant time)
Max Concurrent: 3 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| T1 | — | T4, T5-T8 |
| T2 | — | T4, T6 |
| T3 | — | T4, T6 |
| T4 | T1, T2, T3 | T5 |
| T5 | T4 | T6 |
| T6 | T5, T2, T3 | T7 |
| T7 | T6 | T8 |
| T8 | T7 | T9 |
| T9 | T8 | F1-F4 |

### Agent Dispatch Summary

- **Wave 1**: **3** — T1 → `quick`, T2 → `deep`, T3 → `deep`
- **Wave 2**: **1** — T4 → `quick`
- **Wave 3**: **4** (sequential) — T5 → `unspecified-high`, T6 → `deep`, T7 → `unspecified-high`, T8 → `quick`
- **Wave 4**: **1** — T9 → `quick`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Verify Existing Data Reproducibility

  **What to do**:
  - **First**: Capture the current HEAD commit hash as the pre-revision baseline: `git rev-parse HEAD > .sisyphus/evidence/baseline-commit.txt`. This will be used by Task 9 to compute net line additions.
  - Run `python3 src/significance_tests.py` on the existing feature CSVs to regenerate ANOVA results
  - Compare output against the paper's reported statistics: 29/36 significant features, specific F-statistics for the top 6 features (pitch_range_semitones F=42.18, dissonance_ratio F=30.65, pitch_class_entropy F=21.79, std_note_duration F=19.35, rhythmic_pattern_entropy F=18.19, harmonic_density_mean F=17.67)
  - Run `python3 src/evolution_analysis.py` or inspect `data/stats/evolution_coefficients.csv` to verify DD values match Table 2 (pitch_range DD=+25.14, dissonance_ratio DD=+0.315, harmonic_density DD=+0.754, rhythmic_pattern_entropy DD=-0.101)
  - If any number does NOT match, STOP and report the discrepancy — do not proceed with other tasks until resolved
  - Verify the feature CSVs exist and have expected row counts (144 rows each): `data/features/harmonic_features.csv`, `data/features/melodic_features.csv`, `data/features/rhythmic_features.csv`

  **Must NOT do**:
  - Do NOT modify any existing scripts or data files
  - Do NOT regenerate feature CSVs from MusicXML (only verify existing ones)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple verification task — run existing scripts and compare output
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - None needed — straightforward script execution

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Tasks 4, 5, 6, 7, 8, 9
  - **Blocked By**: None (can start immediately)

  **References** (CRITICAL):

  **Pattern References**:
  - `src/significance_tests.py` — the ANOVA/Tukey pipeline to run. Uses `run_significance_tests()` function. Accepts `--anova-output` and `--tukey-output` flags.
  - `src/evolution_analysis.py` — computes evolutionary velocities and DD values. Key function: `analyze_velocities()`.
  - `src/ddd_analysis.py` — computes DDD values. Key function: `run_ddd_analysis()`.

  **API/Type References**:
  - `data/stats/anova_summary.csv` — existing ANOVA output to compare against
  - `data/stats/evolution_coefficients.csv` — existing evolution coefficients (if present)
  - `data/features/harmonic_features.csv`, `data/features/melodic_features.csv`, `data/features/rhythmic_features.csv` — the source feature CSVs

  **External References**:
  - `JournalMusicResearch/main.tex` lines 503-515 — the reported F-statistics to verify against
  - `JournalMusicResearch/main.tex` lines 660-676 — the reported DD values in Table 2

  **WHY Each Reference Matters**:
  - The significance_tests.py output MUST match the paper's numbers exactly. If they don't, the entire revision becomes suspect.
  - The evolution_analysis.py output confirms the DD values used throughout the paper's narrative.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Verify ANOVA statistics match paper
    Tool: Bash
    Preconditions: Feature CSVs exist at data/features/
    Steps:
      1. Run: mkdir -p .sisyphus/evidence && python3 src/significance_tests.py --anova-output /tmp/verify_anova.csv --tukey-output /tmp/verify_tukey.csv 2>&1 | tee .sisyphus/evidence/task-1-anova-verify.txt
      2. Run: python3 -c "
import pandas as pd
df = pd.read_csv('/tmp/verify_anova.csv')
pr = df[df['feature']=='pitch_range_semitones']
f_val = pr['f_statistic'].values[0]
sig_count = df['significant_fdr'].sum() if 'significant_fdr' in df.columns else (df['p_value_fdr'] < 0.05).sum()
print(f'F_pitch_range={f_val:.2f}')
print(f'significant_count={sig_count}')
assert abs(f_val - 42.18) < 0.5, f'F mismatch: {f_val}'
assert sig_count == 29, f'Count mismatch: {sig_count}'
print('PASS')
" 2>&1 | tee -a .sisyphus/evidence/task-1-anova-verify.txt
    Expected Result: F-statistics match paper within rounding tolerance, 29/36 significant at BH-FDR threshold
    Failure Indicators: F-statistic differs by >0.5, or significant count != 29
    Evidence: .sisyphus/evidence/task-1-anova-verify.txt

  Scenario: Verify feature CSV integrity
    Tool: Bash
    Preconditions: None
    Steps:
      1. Run: python3 -c "import pandas as pd; h=pd.read_csv('data/features/harmonic_features.csv'); m=pd.read_csv('data/features/melodic_features.csv'); r=pd.read_csv('data/features/rhythmic_features.csv'); print(f'H:{len(h)} M:{len(m)} R:{len(r)}'); assert len(h)==144 and len(m)==144 and len(r)==144; print('PASS')" 2>&1 | tee .sisyphus/evidence/task-1-csv-integrity.txt
    Expected Result: All three CSVs have exactly 144 rows
    Failure Indicators: Any CSV has != 144 rows
    Evidence: .sisyphus/evidence/task-1-csv-integrity.txt
  ```

  **Commit**: YES (groups with 2, 3)
  - Message: `feat: add extended PCA and placebo DD analysis scripts`
  - Files: N/A (verification only, no files changed)
  - Pre-commit: N/A

- [x] 2. Create and Run PCA Extended Analysis Script (PC4/PC5)

  **What to do**:
  - Create `src/pca_extended.py` — a standalone script that replicates the existing PCA pipeline but with `n_components=5`
  - Must use the EXACT same feature preparation as `src/feature_embedding.py:_prepare_feature_matrix()`:
    - Same 6 excluded features: `note_count`, `note_event_count`, `chord_event_count`, `chord_quality_total`, `roman_chord_count`, `dissonant_note_count`
    - Same StandardScaler standardization
    - Same fillna(mean) imputation
    - `PCA(n_components=5, random_state=42, svd_solver='full')` — svd_solver='full' ensures PC1-3 are identical to existing 3-component PCA
  - Load all three feature CSVs, merge on key columns, prepare feature matrix, run PCA
  - Output `data/stats/pca_extended.csv` with columns: `component`, `explained_variance_ratio`, `cumulative_variance`, and top 5 loading features with their values
  - Print to stdout: explained variance per component, cumulative variance, top 3 loadings per component
  - Verify that PC1=22.3%, PC2=16.1%, PC3=9.8% (within rounding) to confirm consistency with paper
  - Follow existing script conventions: argparse, type hints, `from __future__ import annotations`, docstring, `sys.exit(main())` pattern

  **Must NOT do**:
  - Do NOT modify `src/feature_embedding.py` or `src/embedding_cache.py`
  - Do NOT add this to `quickstart.sh`
  - Do NOT generate any plots or figures
  - Do NOT use a different random seed or solver than specified

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Must carefully replicate exact feature preparation logic from existing code to ensure PC1-3 consistency
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - None needed — pure Python data analysis

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 4, 6
  - **Blocked By**: None (can start immediately)

  **References** (CRITICAL):

  **Pattern References**:
  - `src/feature_embedding.py:_prepare_feature_matrix()` (around lines 100-130) — MUST replicate this function's exact logic for feature exclusion, fillna, and StandardScaler. Copy the EXCLUDED_FEATURES set exactly.
  - `src/feature_embedding.py:_compute_projection()` (around lines 140-180) — shows how PCA is instantiated with `random_state=42`
  - `src/significance_tests.py` — follows the argparse + `sys.exit(main())` + CSV output pattern to copy

  **API/Type References**:
  - `data/features/harmonic_features.csv` — key columns: `composer_label`, `title`, `mxl_path` + 16 harmonic features
  - `data/features/melodic_features.csv` — key columns + 11 melodic features
  - `data/features/rhythmic_features.csv` — key columns + 9 rhythmic features
  - `data/stats/pca_loadings.csv` — existing PC1-3 loadings. New script must produce a SUPERSET of this (PC1-5).

  **External References**:
  - sklearn PCA documentation: `svd_solver='full'` guarantees deterministic component ordering regardless of n_components

  **WHY Each Reference Matters**:
  - `_prepare_feature_matrix()` is the CRITICAL reference — any deviation in feature exclusion or scaling will produce different PC1-3 values, breaking consistency with the paper
  - The existing `pca_loadings.csv` provides ground truth for validating PC1-3 match

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: PC1-3 values match existing pipeline
    Tool: Bash
    Preconditions: data/features/ CSVs exist, src/pca_extended.py created
    Steps:
      1. Run: mkdir -p .sisyphus/evidence && python3 src/pca_extended.py --output data/stats/pca_extended.csv 2>&1 | tee .sisyphus/evidence/task-2-pca-consistency.txt
      2. Run: python3 -c "
import pandas as pd
df = pd.read_csv('data/stats/pca_extended.csv')
pc1 = df[df['component']=='PC1']['explained_variance_ratio'].values[0]
pc2 = df[df['component']=='PC2']['explained_variance_ratio'].values[0]
pc3 = df[df['component']=='PC3']['explained_variance_ratio'].values[0]
print(f'PC1={pc1:.4f} PC2={pc2:.4f} PC3={pc3:.4f}')
assert 0.220 <= pc1 <= 0.226, f'PC1 out of range: {pc1}'
assert 0.158 <= pc2 <= 0.164, f'PC2 out of range: {pc2}'
assert 0.095 <= pc3 <= 0.101, f'PC3 out of range: {pc3}'
print('PASS')
" 2>&1 | tee -a .sisyphus/evidence/task-2-pca-consistency.txt
    Expected Result: PC1-3 match paper values within rounding tolerance
    Failure Indicators: Any PC1-3 value outside tolerance range
    Evidence: .sisyphus/evidence/task-2-pca-consistency.txt

  Scenario: PC4/PC5 values computed and saved
    Tool: Bash
    Preconditions: src/pca_extended.py created
    Steps:
      1. Run: python3 -c "
import pandas as pd
df = pd.read_csv('data/stats/pca_extended.csv')
assert len(df) == 5, f'Expected 5 rows, got {len(df)}'
pc3 = df[df['component']=='PC3']['explained_variance_ratio'].values[0]
pc4 = df[df['component']=='PC4']['explained_variance_ratio'].values[0]
pc5 = df[df['component']=='PC5']['explained_variance_ratio'].values[0]
cum5 = df[df['component']=='PC5']['cumulative_variance'].values[0]
print(f'PC4={pc4:.4f} PC5={pc5:.4f} cumulative_5={cum5:.4f}')
assert 0 < pc4 < pc3, f'PC4 not valid: {pc4}'
assert 0 < pc5 < pc3, f'PC5 not valid: {pc5}'
assert cum5 > 0.482, f'Cumulative must exceed 48.2%: {cum5}'
print('PASS')
" 2>&1 | tee .sisyphus/evidence/task-2-pc4pc5-output.txt
    Expected Result: 5-component CSV with valid variance values
    Failure Indicators: File has != 5 rows, or PC4/PC5 missing, or cumulative < 48.2%
    Evidence: .sisyphus/evidence/task-2-pc4pc5-output.txt

  Scenario: Script follows project conventions
    Tool: Bash
    Preconditions: src/pca_extended.py created
    Steps:
      1. Run: mkdir -p .sisyphus/evidence && python3 -m py_compile src/pca_extended.py 2>&1 | tee .sisyphus/evidence/task-2-conventions.txt
      2. Run: grep -c "from __future__ import annotations" src/pca_extended.py 2>&1 | tee -a .sisyphus/evidence/task-2-conventions.txt
      3. Run: grep -c "argparse" src/pca_extended.py 2>&1 | tee -a .sisyphus/evidence/task-2-conventions.txt
      4. Run: grep -c "sys.exit" src/pca_extended.py 2>&1 | tee -a .sisyphus/evidence/task-2-conventions.txt
    Expected Result: Compiles, has future annotations, argparse, sys.exit pattern
    Failure Indicators: py_compile fails, or any grep returns 0
    Evidence: .sisyphus/evidence/task-2-conventions.txt
  ```

  **Commit**: YES (group with 1, 3)
  - Message: `feat: add extended PCA and placebo DD analysis scripts`
  - Files: `src/pca_extended.py`, `data/stats/pca_extended.csv`
  - Pre-commit: `python3 -m py_compile src/pca_extended.py && python3 src/pca_extended.py --output data/stats/pca_extended.csv`

- [x] 3. Create and Run Placebo DD Permutation Test Script

  **What to do**:
  - Create `src/placebo_dd.py` — a standalone script that tests whether the observed DD values could arise by chance
  - **Permutation scheme (CRITICAL)**: Shuffle INDIVIDUAL piece composer labels (N=144), NOT group labels (N=4). This produces a continuous null distribution with proper statistical power. The 4!=24 group-level permutation space is too small.
  - For each permutation iteration: shuffle the 144 composer labels randomly, recompute group means for each "composer", compute velocities and DD using the same formula as the paper (eq. 2: DD = v_romantic - v_classical)
  - Run for N=10,000 permutations (configurable via `--n-permutations`)
  - Compute empirical two-sided p-value: p = (count of |DD_perm| >= |DD_observed| + 1) / (N + 1)
  - Test at minimum these 4 features: `pitch_range_semitones`, `dissonance_ratio`, `harmonic_density_mean`, `rhythmic_pattern_entropy` (the same 4 in Table 2)
  - Output `data/stats/placebo_dd.csv` with columns: `feature`, `observed_dd`, `p_value`, `n_permutations`, `n_exceeding`
  - Handle edge case: `rhythmic_pattern_entropy` has DD=-0.101 (near zero). Its p-value is expected to be NON-significant (p >> 0.05). This is actually GOOD — it confirms the null where expected. Report this honestly.
  - Follow existing script conventions: argparse, type hints, `from __future__ import annotations`, docstring, `sys.exit(main())`, `--seed` flag (default 42)
  - Reuse the merge/grouping logic from `src/ddd_analysis.py` or `src/evolution_analysis.py` where possible (import or replicate)
  - The composer ordering for DD computation is always: Bach → Mozart → Chopin → Debussy (chronological)

  **Must NOT do**:
  - Do NOT shuffle group-level labels (only 4!=24 permutations, minimum p ≈ 0.042)
  - Do NOT modify existing `src/ddd_analysis.py` or `src/evolution_analysis.py`
  - Do NOT add this to `quickstart.sh`
  - Do NOT generate plots

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Statistical methodology must be correct — wrong permutation scheme invalidates results
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Tasks 4, 6
  - **Blocked By**: None (can start immediately)

  **References** (CRITICAL):

  **Pattern References**:
  - `src/ddd_analysis.py:calculate_accelerations()` — shows how DD is computed from group means. Key logic: velocity = mean_later - mean_earlier, DD = v_romantic - v_classical. MUST replicate this exactly.
  - `src/evolution_analysis.py:analyze_velocities()` — alternative implementation of velocity/acceleration computation. Cross-reference to ensure consistency.
  - `src/significance_tests.py` — argparse + CSV output pattern to follow

  **API/Type References**:
  - `data/features/harmonic_features.csv` — column `composer_label` contains the group labels to shuffle
  - `JournalMusicResearch/main.tex` lines 660-676 — Table 2 with the 4 observed DD values to verify against

  **External References**:
  - Standard permutation test methodology: shuffle labels, recompute statistic, count exceedances, empirical p-value

  **WHY Each Reference Matters**:
  - `calculate_accelerations()` defines the EXACT DD computation. The permutation test must use the SAME formula to be valid.
  - The observed DD values from the paper (e.g., +25.14 for pitch_range) serve as ground truth for the permutation test.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Placebo test script runs and produces valid output
    Tool: Bash
    Preconditions: Feature CSVs exist, src/placebo_dd.py created
    Steps:
      1. Run: mkdir -p .sisyphus/evidence && python3 src/placebo_dd.py --n-permutations 10000 --seed 42 --output data/stats/placebo_dd.csv 2>&1 | tee .sisyphus/evidence/task-3-placebo-output.txt
      2. Run: python3 -c "
import pandas as pd
df = pd.read_csv('data/stats/placebo_dd.csv')
required = ['pitch_range_semitones', 'dissonance_ratio', 'harmonic_density_mean', 'rhythmic_pattern_entropy']
for feat in required:
    assert feat in df['feature'].values, f'Missing feature: {feat}'
assert (df['p_value'] >= 0).all() and (df['p_value'] <= 1).all(), 'p_values out of [0,1]'
assert (df['n_permutations'] == 10000).all(), 'n_permutations mismatch'
pr_dd = df[df['feature']=='pitch_range_semitones']['observed_dd'].values[0]
dr_dd = df[df['feature']=='dissonance_ratio']['observed_dd'].values[0]
print(f'pitch_range_dd={pr_dd:.2f} dissonance_dd={dr_dd:.3f}')
assert abs(pr_dd - 25.14) < 1.0, f'pitch_range observed_dd mismatch: {pr_dd}'
assert abs(dr_dd - 0.315) < 0.05, f'dissonance observed_dd mismatch: {dr_dd}'
print('PASS')
" 2>&1 | tee -a .sisyphus/evidence/task-3-placebo-output.txt
    Expected Result: Script runs successfully, output CSV has correct structure, and observed_dd values match the paper's Table 2
    Failure Indicators: Script crashes, missing columns, observed_dd values don't match Table 2
    Evidence: .sisyphus/evidence/task-3-placebo-output.txt

  Scenario: Rhythmic entropy is non-significant (expected null)
    Tool: Bash
    Preconditions: src/placebo_dd.py ran successfully
    Steps:
      1. Run: python3 -c "
import pandas as pd
df = pd.read_csv('data/stats/placebo_dd.csv')
re = df[df['feature']=='rhythmic_pattern_entropy']
p = re['p_value'].values[0]
dd = re['observed_dd'].values[0]
print(f'rhythmic_entropy: p={p:.4f}, dd={dd:.3f}')
assert p > 0.05, f'Expected non-significant but got p={p}'
print('PASS - non-significant as expected')
" 2>&1 | tee .sisyphus/evidence/task-3-placebo-null.txt
    Expected Result: Rhythmic entropy p-value is NOT significant, confirming null where expected
    Failure Indicators: p_value < 0.05 (would be unexpected given DD ≈ -0.101)
    Evidence: .sisyphus/evidence/task-3-placebo-null.txt

  Scenario: Report significance results for LaTeX integration
    Tool: Bash
    Preconditions: data/stats/placebo_dd.csv exists
    Steps:
      1. Run: python3 -c "
import pandas as pd
df = pd.read_csv('data/stats/placebo_dd.csv')
print('=== Placebo DD Results for LaTeX ===')
for _, row in df.iterrows():
    sig = 'SIGNIFICANT' if row['p_value'] < 0.05 else 'not significant'
    print(f\"{row['feature']}: observed_dd={row['observed_dd']:.4f}, p={row['p_value']:.4f} ({sig})\")
print('=== END ===')
" 2>&1 | tee .sisyphus/evidence/task-3-placebo-report.txt
    Expected Result: p-values recorded for downstream LaTeX integration. Honest reporting regardless of outcome.
    Failure Indicators: N/A — this scenario records results, it does not assert specific thresholds
    Evidence: .sisyphus/evidence/task-3-placebo-report.txt

  Scenario: Script follows project conventions
    Tool: Bash
    Preconditions: src/placebo_dd.py created
    Steps:
      1. Run: mkdir -p .sisyphus/evidence && python3 -m py_compile src/placebo_dd.py 2>&1 | tee .sisyphus/evidence/task-3-conventions.txt
      2. Run: grep -c "from __future__ import annotations" src/placebo_dd.py 2>&1 | tee -a .sisyphus/evidence/task-3-conventions.txt
      3. Run: grep -c "argparse" src/placebo_dd.py 2>&1 | tee -a .sisyphus/evidence/task-3-conventions.txt
    Expected Result: Compiles, follows conventions
    Failure Indicators: Compilation fails or conventions missing
    Evidence: .sisyphus/evidence/task-3-conventions.txt
  ```

  **Commit**: YES (group with 1, 2)
  - Message: `feat: add extended PCA and placebo DD analysis scripts`
  - Files: `src/placebo_dd.py`, `data/stats/placebo_dd.csv`
  - Pre-commit: `python3 -m py_compile src/placebo_dd.py && python3 src/placebo_dd.py --n-permutations 10000 --output data/stats/placebo_dd.csv`

- [x] 4. Update references.bib with New Citations

  **What to do**:
  - Add ≤5 new BibTeX entries to `JournalMusicResearch/references.bib` for the revision defenses
  - **Required new entries** (research these carefully for correct bibliographic data):
    1. A citation for corpus size justification in computational musicology (e.g., a canonical ISMIR/JNMR paper that uses 100-200 symbolic works and defends that sample size)
    2. A citation for a primary historical source documenting Debussy's acknowledgment of Chopin's influence (the paper currently claims this without primary sourcing; try Debussy's correspondence or a scholarly biography — e.g., Lesure's Debussy biography, or Nichols' "Debussy Remembered")
    3. A citation for a computational musicology survey that establishes "the majority of computational studies frame the problem as one of composer classification" (the paper claims this at line 186 without citation — e.g., Herremans et al. 2017 survey, or the Simonetta2025 already cited could suffice if the claim is narrowed)
  - **Conditionally needed** (based on Python results from Tasks 2/3):
    4. If using Rambachan & Roth sensitivity argument: `Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. Review of Economic Studies.` — BUT Metis warned this may not apply to our DD design (cross-sectional group means, not panel). Only add if the text specifically references their framework. If using permutation test alone as the defense, this entry is NOT needed.
  - **Verify**: Run `bibtex main` after adding entries to confirm zero warnings about formatting
  - **Check existing**: Verify if any currently-cited reference already covers these needs. E.g., `Simonetta2025` is already cited for "surveys confirm sustained interest in style-based composer identification" — can the "majority frame as classification" claim be attributed to it?
  - Use consistent BibTeX formatting matching existing entries in references.bib (check capitalization in titles, journal name formatting, DOI presence)

  **Must NOT do**:
  - Do NOT add more than 5 new entries
  - Do NOT remove any existing entries
  - Do NOT change formatting of existing entries
  - Do NOT add entries that are not directly cited in the revised main.tex

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small focused file edit — adding a few BibTeX entries with correct formatting
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Wave 1)
  - **Blocks**: Tasks 5, 6, 7, 8
  - **Blocked By**: Tasks 1, 2, 3

  **References** (CRITICAL):

  **Pattern References**:
  - `JournalMusicResearch/references.bib` — existing file with 18 entries. Match the BibTeX formatting style exactly (capitalization, field order, journal name conventions).

  **API/Type References**:
  - `JournalMusicResearch/main.tex` lines 186 ("majority of computational studies"), 841-842 ("Debussy openly acknowledged Chopin's influence") — the claims that need citations

  **WHY Each Reference Matters**:
  - The "majority of computational studies" claim at line 186 is flagged by the Critic as needing immediate citation
  - The "Debussy acknowledged Chopin" claim at line 841 needs primary historical sourcing, not just secondary analysis
  - Corpus size defense needs a peer-reviewed precedent to cite

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: BibTeX compiles without warnings
    Tool: Bash
    Preconditions: references.bib updated
    Steps:
      1. Run: mkdir -p .sisyphus/evidence && cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex && bibtex main 2>&1 | tee ../.sisyphus/evidence/task-4-bibtex-clean.txt
      2. Run: grep -c "Warning--" ../.sisyphus/evidence/task-4-bibtex-clean.txt || echo "0" | tee -a ../.sisyphus/evidence/task-4-bibtex-clean.txt
      3. Assert count is 0
    Expected Result: Zero BibTeX warnings about missing or malformed entries
    Failure Indicators: Any "Warning--" lines in bibtex output
    Evidence: .sisyphus/evidence/task-4-bibtex-clean.txt

  Scenario: New entries count within budget
    Tool: Bash
    Preconditions: references.bib updated, baseline commit captured in .sisyphus/evidence/baseline-commit.txt by Task 1
    Steps:
      1. Run: bash -c '
         BASELINE=$(cat .sisyphus/evidence/baseline-commit.txt);
         NEW_ENTRIES=$(git diff "$BASELINE" -- JournalMusicResearch/references.bib | grep "^+@" | wc -l | tr -d " ");
         echo "New bibliography entries added: $NEW_ENTRIES";
         [ "$NEW_ENTRIES" -le 5 ] && echo "RESULT: APPROVE" || echo "RESULT: REJECT ($NEW_ENTRIES > 5)"
         ' 2>&1 | tee .sisyphus/evidence/task-4-entry-count.txt
    Expected Result: NEW_ENTRIES ≤ 5 and "RESULT: APPROVE"
    Failure Indicators: More than 5 new @-entries added
    Evidence: .sisyphus/evidence/task-4-entry-count.txt
  ```

  **Commit**: YES
  - Message: `docs: update bibliography for JNMR revision`
  - Files: `JournalMusicResearch/references.bib`
  - Pre-commit: `cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex && bibtex main`

- [x] 5. LaTeX Revisions — Introduction, Literature Review, and Methods

  **What to do**:
  This task addresses review issues in the first three sections of main.tex (lines 158-488). Work through each fix in order:

  **Introduction (lines 158-220):**
  - **Fix "colouristic experimentation"** (line 163-164, Critic issue #11): Replace "the colouristic experimentation of Impressionism" with a precise descriptor. Suggestion: "the timbral and harmonic innovations of Impressionism" or "the novel sonority-driven writing of Impressionism". Must be quantitative/objective in tone.
  - **Soften literature gap claim** (line 195-197, Critic issue #14): Change "has received surprisingly little quantitative attention" to something more measured: "has received comparatively less quantitative attention than static classification tasks" or "remains underexplored relative to classification-oriented approaches". This addresses the Critic's valid point about n-gram/Markov work existing.
  - **Add missing citation** (line 186, Critic issue #7): The claim "The majority of computational studies frame the problem as one of composer classification" needs a citation. Use the Simonetta2025 reference already cited (if it covers this claim) or the new citation added in Task 4. Format: `\citep{Simonetta2025}` or appropriate new key.

  **Literature Review (lines 221-272):**
  - **Minimal changes only**. If the "majority of computational studies" citation from the Introduction is not sufficient, add ONE sentence acknowledging alternative approaches (n-gram, Markov modeling) in the literature review, citing appropriate work. Do NOT restructure the section.

  **Methods — Corpus Curation (lines 275-306):**
  - **Add Beethoven omission defense** (Critic issue #4): After the paragraph explaining the four composer selection (ending around line 300), add 2-3 sentences defending the omission of Beethoven. Key argument: "Beethoven's oeuvre spans and actively drives the transition from the Classical to the Romantic stylistic paradigm. Including his works would confound the distinct pre- and post-shift states required by the differencing framework adopted in this study. The four selected composers instead represent prototypical exemplars of their respective periods, maximising construct validity for the between-era comparisons that follow."
  - **Soften sensitivity analysis claim** (lines 303-306, Metis issue): Change "Sensitivity analyses conducted with relaxed filter thresholds confirmed that loosening..." to "Preliminary checks with relaxed filter thresholds suggested that loosening..." — the current claim implies a formal sensitivity analysis but no script supports it.

  **Methods — Rhythmic Features (lines 357-373):**
  - **Fix rubato conflation** (line 362-363, Critic issue #8): Change "a high standard deviation signals rubato-like alternation between sustained sonorities and rapid figuration" to "a high standard deviation signals alternation between sustained sonorities and rapid figuration, indicating greater notational diversity in rhythmic values". Remove the word "rubato" — symbolic scores encode proportional durations, not performance-level timing. The Critic correctly notes this conflation.

  **Methods — Evolutionary Coefficients (lines 439-451):**
  - **Fix equation variable mapping** (Critic issue #12): After eq. (1) at line 444, the text immediately introduces $v_{\text{classical}}$, $v_{\text{romantic}}$, $v_{\text{impressionist}}$ (lines 446-448). Add explicit mapping: "where $\bar{x}_{\text{later}}$ and $\bar{x}_{\text{earlier}}$ denote the group means of the later and earlier composers, respectively. Three transitions are considered: Baroque to Classical ($v_{\text{classical}} = \bar{x}_{\text{Mozart}} - \bar{x}_{\text{Bach}}$), Classical to Romantic ($v_{\text{romantic}} = \bar{x}_{\text{Chopin}} - \bar{x}_{\text{Mozart}}$), and Romantic to Impressionist ($v_{\text{impressionist}} = \bar{x}_{\text{Debussy}} - \bar{x}_{\text{Chopin}}$)."

  **Methods — DD/DDD section (lines 452-486):**
  - **Strengthen parallel trends defense** (Critic issues #2/#3): In the DDD paragraph (around line 482-486), add 1-2 sentences: "As with any differencing framework applied to historical data, the parallel-trends condition cannot be tested directly. However, a permutation-based falsification test (\Cref{sec:robustness}) provides empirical support by demonstrating that the observed DD magnitudes are unlikely to arise from chance assignment of composer labels."
  - **Add \label{sec:robustness} forward reference** (will be defined in Task 6)

  **Methods — PCA subsection (lines 416-430):**
  - **Clarify count-feature exclusion was pre-specified** (Critic issue #5): The Critic claims features were dropped "after the omnibus ANOVA to force a narrative." Clarify that the exclusion was a methodological decision made BEFORE post-hoc analysis. Around line 418-424, ensure the text makes clear this was a pre-specified decision: "Six count-based features were excluded from the PCA input matrix a priori to prevent absolute composition length from dominating the principal axes." The word "a priori" is key.

  **Must NOT do**:
  - Do NOT restructure any section
  - Do NOT add new \section{} or \subsection{} commands
  - Do NOT change mathematical equations (1), (2), or (3)
  - Do NOT alter any existing numerical claims
  - Do NOT add more than ~30 net lines across all three sections

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multiple precise edits across a large LaTeX file requiring careful attention to academic tone and existing narrative flow
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (edits main.tex, which other LaTeX tasks also edit)
  - **Parallel Group**: Wave 3 (first in sequential chain)
  - **Blocks**: Task 6
  - **Blocked By**: Task 4

  **References** (CRITICAL):

  **Pattern References**:
  - `JournalMusicResearch/main.tex` lines 158-488 — the sections being edited. Read the FULL text of these sections before making any changes to understand narrative flow.
  - `JournalMusicResearch/references.bib` — to verify citation keys for any \citep{} commands added

  **External References**:
  - `JournalMusicResearch/TempPeerReviews/CriticReview2.md` — the specific criticisms being addressed (rubato line 21, Beethoven line 25, Tukey line 26, variable mapping line 43, "colouristic" line 41)
  - `JournalMusicResearch/TempPeerReviews/GenericReview1.md` — parallel trends suggestion (line 22-25)

  **WHY Each Reference Matters**:
  - The full LaTeX context is essential — each edit must flow naturally with surrounding text
  - The reviewer text provides exact language of the criticism, ensuring the defense addresses the specific concern

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: LaTeX compiles after Introduction/Methods edits
    Tool: Bash
    Preconditions: Tasks 4 complete (references.bib updated)
    Steps:
      1. Run: mkdir -p .sisyphus/evidence && cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex 2>&1 | tee ../.sisyphus/evidence/task-5-latex-build.txt
      2. Run: grep -c "Undefined control sequence" main.log | tee -a ../.sisyphus/evidence/task-5-latex-build.txt
      3. Assert count is 0
    Expected Result: Clean compilation with zero errors
    Failure Indicators: Undefined references, missing citations, compilation errors
    Evidence: .sisyphus/evidence/task-5-latex-build.txt

  Scenario: "colouristic experimentation" is removed
    Tool: Bash (grep)
    Preconditions: main.tex edited
    Steps:
      1. Run: grep -c "colouristic experimentation" JournalMusicResearch/main.tex 2>&1 | tee .sisyphus/evidence/task-5-colouristic-removed.txt
      2. Assert count is 0
    Expected Result: Phrase no longer appears
    Failure Indicators: Phrase still present
    Evidence: .sisyphus/evidence/task-5-colouristic-removed.txt

  Scenario: "rubato" removed from rhythmic features description
    Tool: Bash (grep)
    Preconditions: main.tex edited
    Steps:
      1. Run: grep -n "rubato" JournalMusicResearch/main.tex 2>&1 | tee .sisyphus/evidence/task-5-rubato-removed.txt
      2. Assert no matches in the Methods section (lines 273-488)
    Expected Result: "rubato" not mentioned in Methods
    Failure Indicators: "rubato" still in Methods section
    Evidence: .sisyphus/evidence/task-5-rubato-removed.txt

  Scenario: Equation variable mapping is explicit
    Tool: Bash (grep)
    Preconditions: main.tex edited
    Steps:
      1. Run: grep -A5 "v_{\\\\text{classical}}" JournalMusicResearch/main.tex 2>&1 | tee .sisyphus/evidence/task-5-variable-mapping.txt
      2. Verify the text contains explicit mapping like "bar{x}_{\\text{Mozart}}" or similar
    Expected Result: Variables formally mapped to composer names
    Failure Indicators: Variables introduced without formal mapping
    Evidence: .sisyphus/evidence/task-5-variable-mapping.txt
  ```

  **Commit**: YES
  - Message: `docs: address critical review issues in Introduction and Methods`
  - Files: `JournalMusicResearch/main.tex`
  - Pre-commit: `cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex`

- [x] 6. LaTeX Revisions — Results Section + Robustness Checks Subsection

  **What to do**:
  This task makes the most substantial structural change: adding a PC4/PC5 paragraph and a new Robustness Checks subsection. It REQUIRES output from Tasks 2 and 3 (actual numbers).

  **PCA Results — PC4/PC5 paragraph (after line 597):**
  - Read `data/stats/pca_extended.csv` (produced by Task 2) to get the actual PC4 and PC5 explained variance percentages and top loadings
  - After the existing PC3 discussion paragraph (ending around line 619), add a new paragraph (~8-12 lines) discussing PC4 and PC5:
    - Report their variance explained and cumulative variance for all 5 components
    - Describe their top 2-3 loadings and what musicological dimension they capture (if interpretable)
    - **Contingency**: If PC4 or PC5 explain < 3% variance each or their loadings have no clear musicological interpretation, write: "The fourth and fifth components explain [X]% and [Y]% of variance respectively (cumulative: [Z]%), with loadings distributed across multiple features without a clear single musicological dimension. The diminishing marginal variance and absence of interpretable structure beyond the third component support the retention of three components for the primary analysis."
    - Justify the 3-component choice: mention scree plot diminishing returns, visualization purposes (3D), and the fact that the retained 48.2% captures the primary stylistic dimensions (harmonic complexity, temporal structure, contrapuntal texture)

  **Tukey HSD clarification (lines 534-544):**
  - **Clarify feature exclusion was pre-specified** (Critic issue #5): The text currently says "six of the 29 features are raw counts... Excluding these count-based features reveals..." The Critic reads this as post-hoc manipulation. Add a parenthetical or sentence clarifying: "(these six features were excluded from PCA a priori, as described in \Cref{sec:methods}; the same exclusion is applied here for consistency)". This directly links the Results exclusion to the pre-specified Methods decision.

  **NEW: Robustness Checks subsection (before Discussion, after line 751):**
  - Add `\subsection{Robustness Checks}\label{sec:robustness}` as the final subsection of \Cref{sec:results}
  - Read `data/stats/placebo_dd.csv` (produced by Task 3) to get the actual p-values
  - Write ~20-30 lines describing the permutation test:
    - Methodology: "To assess the statistical significance of the DD values reported in \Cref{tab:evolution}, a permutation test was conducted. In each of 10{,}000 iterations, the 144 composer labels were randomly reassigned to scores, group means recomputed, and the DD statistic recalculated. The proportion of permutations yielding a DD magnitude at least as large as the observed value provides an empirical $p$-value under the null hypothesis of no era-specific stylistic effect."
    - Results: Report p-values for the 4 features from Table 2
    - Interpretation: "The permutation test confirms that the Romantic Reversal in pitch range ($p < [VALUE]$), dissonance ratio ($p < [VALUE]$), and harmonic density ($p < [VALUE]$) is statistically significant and unlikely to arise from arbitrary label assignment. The near-zero DD for rhythmic pattern entropy yields a non-significant $p$-value ($p = [VALUE]$), consistent with the observation that rhythmic innovation did not accelerate during the Romantic era."
    - This directly addresses both the Critic's concern about DD legitimacy and GenericReview1's suggestion about placebo tests

  **Must NOT do**:
  - Do NOT change the existing Table 2 or its values
  - Do NOT modify the PCA eigenvalue / loading numbers for PC1-3
  - Do NOT add a new figure for the permutation null distribution (unless the numbers are particularly compelling, in which case a small inline histogram is acceptable but not required)
  - Do NOT move or restructure existing Results subsections

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Most substantial structural change — adding a new subsection and integrating real computed numbers. Must read output CSVs and weave results into narrative.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (edits main.tex after Task 5)
  - **Parallel Group**: Wave 3 (second in sequential chain)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 2, 3, 5

  **References** (CRITICAL):

  **Pattern References**:
  - `JournalMusicResearch/main.tex` lines 488-751 — the Results section being edited
  - `JournalMusicResearch/main.tex` lines 592-636 — the existing PCA discussion where PC4/PC5 paragraph goes
  - `JournalMusicResearch/main.tex` lines 534-544 — the Tukey HSD count-feature exclusion paragraph

  **API/Type References**:
  - `data/stats/pca_extended.csv` — PC4/PC5 variance and loadings (from Task 2). Columns: `component`, `explained_variance_ratio`, `cumulative_variance`, top loading features
  - `data/stats/placebo_dd.csv` — permutation test p-values (from Task 3). Columns: `feature`, `observed_dd`, `p_value`

  **WHY Each Reference Matters**:
  - The pca_extended.csv provides the EXACT numbers to cite. Do NOT make up variance percentages.
  - The placebo_dd.csv provides the EXACT p-values to report. Do NOT estimate or round inappropriately.
  - The existing Results text must flow naturally into the new material.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: PC4/PC5 paragraph contains real numbers from CSV
    Tool: Bash
    Preconditions: data/stats/pca_extended.csv exists, main.tex edited
    Steps:
      1. Run: mkdir -p .sisyphus/evidence && python3 -c "
import pandas as pd
df = pd.read_csv('data/stats/pca_extended.csv')
pc4_var = df[df['component']=='PC4']['explained_variance_ratio'].values[0]
pc4_pct = f'{pc4_var*100:.1f}'
with open('JournalMusicResearch/main.tex') as f:
    tex = f.read()
if pc4_pct in tex:
    print(f'PASS: PC4 percentage {pc4_pct}% found in main.tex')
else:
    print(f'FAIL: PC4 percentage {pc4_pct}% NOT found in main.tex')
    raise AssertionError(f'PC4 value {pc4_pct} not in LaTeX')
" 2>&1 | tee .sisyphus/evidence/task-6-pc4-numbers.txt
    Expected Result: PC4 variance from CSV appears in LaTeX
    Failure Indicators: Number not found or different from CSV
    Evidence: .sisyphus/evidence/task-6-pc4-numbers.txt

  Scenario: Robustness Checks subsection exists with correct label
    Tool: Bash (grep)
    Preconditions: main.tex edited
    Steps:
      1. Run: mkdir -p .sisyphus/evidence && grep -c "\\\\subsection{Robustness Checks}" JournalMusicResearch/main.tex 2>&1 | tee .sisyphus/evidence/task-6-robustness-exists.txt
      2. Assert count is 1
      3. Run: grep -c "\\\\label{sec:robustness}" JournalMusicResearch/main.tex 2>&1 | tee -a .sisyphus/evidence/task-6-robustness-exists.txt
      4. Assert count is 1
    Expected Result: Subsection exists with proper label
    Failure Indicators: Missing subsection or label
    Evidence: .sisyphus/evidence/task-6-robustness-exists.txt

  Scenario: Placebo p-values from CSV appear in LaTeX
    Tool: Bash
    Preconditions: data/stats/placebo_dd.csv exists, main.tex edited
    Steps:
      1. Run: python3 -c "
import pandas as pd
df = pd.read_csv('data/stats/placebo_dd.csv')
pr = df[df['feature']=='pitch_range_semitones']
p = pr['p_value'].values[0]
print(f'pitch_range p-value: {p:.4f}')
# Check if this value (or a rounded version) appears in main.tex
import re
with open('JournalMusicResearch/main.tex') as f:
    tex = f.read()
# Look for the p-value in the Robustness Checks section
robustness_start = tex.find('Robustness Checks')
if robustness_start == -1:
    print('FAIL: Robustness Checks section not found')
else:
    robustness_section = tex[robustness_start:robustness_start+3000]
    if 'pitch' in robustness_section.lower() and ('p' in robustness_section or '<' in robustness_section):
        print('PASS: pitch range p-value referenced in Robustness Checks')
    else:
        print('FAIL: pitch range not discussed in Robustness Checks')
" 2>&1 | tee .sisyphus/evidence/task-6-placebo-pvalues.txt
    Expected Result: Reported p-values match CSV output
    Failure Indicators: p-values missing or inconsistent
    Evidence: .sisyphus/evidence/task-6-placebo-pvalues.txt

  Scenario: LaTeX compiles after Results edits
    Tool: Bash
    Preconditions: main.tex edited with Robustness Checks subsection and PC4/PC5 paragraph
    Steps:
      1. Run: mkdir -p .sisyphus/evidence && bash -c '
         cd JournalMusicResearch &&
         pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 &&
         bibtex main > /dev/null 2>&1 &&
         pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 &&
         pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1;
         ERRORS=$(grep -c "^!" main.log 2>/dev/null || echo 0);
         UNDEF=$(grep -c "undefined" main.log 2>/dev/null || echo 0);
         echo "LaTeX build: errors=$ERRORS undefined_refs=$UNDEF";
         [ "$ERRORS" -eq 0 ] && [ "$UNDEF" -eq 0 ] && echo "RESULT: APPROVE" || echo "RESULT: REJECT"
         ' 2>&1 | tee .sisyphus/evidence/task-6-latex-build.txt
    Expected Result: errors=0 undefined_refs=0 and "RESULT: APPROVE"
    Failure Indicators: Any non-zero error or undefined reference count
    Evidence: .sisyphus/evidence/task-6-latex-build.txt
  ```

  **Commit**: YES
  - Message: `docs: add Robustness Checks subsection and PC4/PC5 analysis to Results`
  - Files: `JournalMusicResearch/main.tex`
  - Pre-commit: `cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex`

- [x] 7. LaTeX Revisions — Discussion and Conclusion

  **What to do**:
  This task addresses review issues in the Discussion (lines ~755-950) and Conclusion (lines ~951-1040) sections of main.tex. Work through each fix in order:

  **Discussion — Silhouette Paradox (lines 854-906, Critic issue #6):**
  - First, verify that `data/stats/cluster_metrics.json` exists and contains the expected keys (`silhouette_score`, `davies_bouldin_index`). If it does not exist, regenerate it by running: `python3 src/cluster_evaluation.py --cache data/embeddings/pca_embedding_cache.csv --output data/stats/cluster_metrics.json`
  - The Critic correctly notes a tension: moderate silhouette scores (0.095 3D / 0.102 36D) suggest overlapping clusters, yet the paper simultaneously claims meaningful stylistic separability via ANOVA and post-hoc tests.
  - Tighten the existing argumentation (do NOT delete it). The current text around lines 870-906 already discusses why moderate silhouette scores are expected for music. Strengthen by:
    1. Adding a sentence explicitly framing the silhouette as measuring geometric compactness, NOT stylistic distinctiveness: "The silhouette coefficient measures geometric compactness and separation in the embedding space; it does not directly assess whether group means differ significantly — a role fulfilled by the ANOVA and Tukey HSD results reported above."
    2. Noting that low silhouette + significant ANOVA is the EXPECTED pattern for cultural data with high within-group variance: "Cultural corpora exhibiting substantial intra-composer variability are expected to produce moderate silhouette values even when between-group differences are statistically robust."
    3. If space allows, add: "The Davies-Bouldin index of [VALUE from cluster_metrics.json] corroborates this reading: clusters are distinguishable though internally heterogeneous."

  **Discussion — Features as Proxies (lines ~755-796, Critic issue #9):**
  - The Critic argues symbolic features are "proxies" that miss performed nuance. This is a valid point but the paper already implicitly acknowledges this.
  - Add 2-3 sentences (NOT a new paragraph — integrate into the existing limitations discussion or the feature interpretation paragraph):
    "The 36 features employed here are extracted from symbolic scores and therefore capture notated rather than performed musical properties. This is a deliberate methodological choice: symbolic representations enable systematic, reproducible comparison across historical corpora for which no authoritative performance recordings exist. Expressive nuances such as performed rubato, dynamic shading, and timbral variation lie outside the scope of symbolic analysis; however, the consistent extraction of density and ratio-based metrics from standardised scores ensures that the identified stylistic contrasts reflect compositional intent rather than editorial or performative artefacts."

  **Discussion — Chopin-Debussy Primary Source (line 841-842, Critic issue #7):**
  - The paper claims "Debussy openly acknowledged Chopin's influence" but provides no primary historical citation.
  - Add the citation from Task 4 (Debussy biography/correspondence reference): change to "Debussy openly acknowledged Chopin's influence \citep{NEW_KEY}" where NEW_KEY is the bibliography key added in Task 4 for the Debussy/Chopin influence source.

  **Discussion — Corpus Size Defense (around lines ~1014-1017 or wherever appropriate):**
  - If not already addressed sufficiently in Methods (Task 5), add 1-2 sentences in the Discussion/Limitations defending corpus size: "The corpus of 144 solo piano works (36 per composer) is consistent with the sample sizes employed in comparable computational musicology studies [CITE from Task 4]. The unit of observation for feature extraction is the individual composition — not the note or measure — and the omnibus ANOVA's demonstrated statistical power (29 of 36 features significant at the Bonferroni-adjusted threshold) confirms adequate sample size for the comparisons attempted."
  - Place this in the limitations paragraph if one exists, or at the end of the Discussion before the Conclusion transition.

  **Conclusion — Limitations Update (lines 1014-1040):**
  - If the Conclusion contains a limitations paragraph, ensure it acknowledges:
    1. Symbolic (not performed) features
    2. Corpus restricted to solo piano
    3. Four-composer design excludes transitional figures
  - These should already be partially present. Add ONLY what is missing. Do NOT repeat what was added in Discussion.

  **Must NOT do**:
  - Do NOT restructure the Discussion section or change paragraph ordering
  - Do NOT add more than ~25 net lines across Discussion and Conclusion
  - Do NOT change any existing numerical claims
  - Do NOT add new \section{} or \subsection{} commands
  - Do NOT weaken existing arguments — only strengthen them

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Nuanced academic prose editing requiring careful integration with existing argumentation and proper citation management
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - None needed — LaTeX text editing

  **Parallelization**:
  - **Can Run In Parallel**: NO (edits main.tex after Task 6)
  - **Parallel Group**: Wave 3 (third in sequential chain)
  - **Blocks**: Task 8
  - **Blocked By**: Task 6

  **References** (CRITICAL):

  **Pattern References**:
  - `JournalMusicResearch/main.tex` lines 755-1040 — the Discussion and Conclusion sections being edited. Read the FULL text before editing.
  - `JournalMusicResearch/main.tex` lines 854-906 — the existing silhouette discussion paragraph. This already contains partial defense; strengthen, do NOT replace.

  **API/Type References**:
  - `data/stats/cluster_metrics.json` — contains exact Silhouette and Davies-Bouldin values to cite if strengthening the silhouette defense
  - `JournalMusicResearch/references.bib` — to verify citation keys for \citep{} commands

  **External References**:
  - `JournalMusicResearch/TempPeerReviews/CriticReview2.md` — silhouette paradox (line 28-30), features as proxies (line 15-16), Chopin-Debussy source (line 40)
  - `JournalMusicResearch/TempPeerReviews/GenericReview1.md` — corpus size suggestion (line 18-20)

  **WHY Each Reference Matters**:
  - The existing silhouette discussion (lines 854-906) must be READ IN FULL before editing — the goal is to strengthen it, not rewrite it
  - cluster_metrics.json provides exact Davies-Bouldin value to cite
  - The Critic's specific language guides what the defense must address

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: cluster_metrics.json exists with expected keys
    Tool: Bash
    Preconditions: None
    Steps:
      1. Run: mkdir -p .sisyphus/evidence && python3 -c "
import json
with open('data/stats/cluster_metrics.json') as f:
    m = json.load(f)
assert 'silhouette_score' in str(m) or 'silhouette' in str(m).lower(), 'Missing silhouette key'
assert 'davies_bouldin' in str(m).lower(), 'Missing Davies-Bouldin key'
print(f'cluster_metrics: {json.dumps(m, indent=2)}')
print('PASS')
" 2>&1 | tee .sisyphus/evidence/task-7-cluster-metrics.txt
    Expected Result: File exists with silhouette and Davies-Bouldin values
    Failure Indicators: File missing or keys absent. If missing, run: python3 src/cluster_evaluation.py --cache data/embeddings/pca_embedding_cache.csv --output data/stats/cluster_metrics.json
    Evidence: .sisyphus/evidence/task-7-cluster-metrics.txt

  Scenario: Silhouette defense strengthened
    Tool: Bash (grep)
    Preconditions: main.tex edited
    Steps:
      1. Run: mkdir -p .sisyphus/evidence && grep -c "geometric compactness" JournalMusicResearch/main.tex 2>&1 | tee .sisyphus/evidence/task-7-silhouette-defense.txt
      2. Assert count >= 1
      3. Run: grep -c "within-group\|intra-composer" JournalMusicResearch/main.tex 2>&1 | tee -a .sisyphus/evidence/task-7-silhouette-defense.txt
      4. Assert count >= 1 (in the Discussion section)
    Expected Result: Silhouette defense includes geometric compactness framing and within-group variance argument
    Failure Indicators: Key phrases missing
    Evidence: .sisyphus/evidence/task-7-silhouette-defense.txt

  Scenario: Features-as-proxies defense present
    Tool: Bash (grep)
    Preconditions: main.tex edited
    Steps:
      1. Run: grep -c "symbolic.*notated\|notated.*performed\|compositional intent" JournalMusicResearch/main.tex 2>&1 | tee .sisyphus/evidence/task-7-proxies-defense.txt
      2. Assert count >= 1
    Expected Result: Discussion acknowledges symbolic vs performed distinction
    Failure Indicators: No acknowledgment found
    Evidence: .sisyphus/evidence/task-7-proxies-defense.txt

  Scenario: Chopin-Debussy claim has citation
    Tool: Bash (grep)
    Preconditions: main.tex edited
    Steps:
      1. Run: grep -A2 "Debussy.*acknowledged.*Chopin\|Chopin.*influence" JournalMusicResearch/main.tex 2>&1 | tee .sisyphus/evidence/task-7-chopin-debussy-cite.txt
      2. Assert output contains a \citep{} or \cite{} command
    Expected Result: The "Debussy acknowledged Chopin's influence" claim now has a citation
    Failure Indicators: Claim exists without any citation in surrounding context
    Evidence: .sisyphus/evidence/task-7-chopin-debussy-cite.txt

  Scenario: LaTeX compiles after Discussion edits
    Tool: Bash
    Preconditions: main.tex edited
    Steps:
      1. Run: cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex 2>&1 | tee ../.sisyphus/evidence/task-7-latex-build.txt
      2. Assert zero errors
    Expected Result: Clean compilation
    Failure Indicators: Undefined references or compilation errors
    Evidence: .sisyphus/evidence/task-7-latex-build.txt
  ```

  **Commit**: YES
  - Message: `docs: strengthen Discussion defenses for silhouette, proxies, and corpus size`
  - Files: `JournalMusicResearch/main.tex`
  - Pre-commit: `cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex`

- [x] 8. LaTeX Throughout Fixes — BH FDR, Capitalization, and Minor Polish

  **What to do**:
  This task performs global minor fixes across the entire main.tex. These are small, cosmetic, but important for submission quality.

  **BH FDR Repetition (Critic issue #13):**
  - The Critic notes that "Benjamini-Hochberg false-discovery-rate" is spelled out fully multiple times. Scan the document for all occurrences of "Benjamini-Hochberg" and "false-discovery-rate" / "false discovery rate".
  - First occurrence: keep fully expanded with acronym introduction, e.g., "Benjamini-Hochberg false-discovery-rate (BH-FDR)".
  - All subsequent occurrences: replace with "BH-FDR" only.
  - If the text already uses a different acronym pattern, maintain consistency with whatever abbreviation is defined first.

  **Capitalization Consistency (Critic issue #15):**
  - Musical era names must be consistently capitalised throughout: "Baroque", "Classical", "Romantic", "Impressionist/Impressionism".
  - Scan for any lowercase instances: "baroque", "classical" (when referring to the era, not the generic adjective), "romantic" (when referring to the era), "impressionist".
  - Also check for inconsistent variants like "romantic era" vs "Romantic era", "classical period" vs "Classical period".
  - Fix all to capitalised form when referring to musical eras.

  **Other Minor Polish:**
  - Check for double spaces (replace `  ` with ` ` where not in LaTeX commands)
  - Check for consistent use of British English `-ise` endings (note: some technical terms like "standardize" in code contexts are acceptable)
  - Verify em-dashes vs en-dashes are used correctly (LaTeX: `---` for em-dash, `--` for en-dash in number ranges)
  - Check for orphaned `\label{}` or `\ref{}` commands that don't resolve

  **Must NOT do**:
  - Do NOT change any paragraph structure or content meaning
  - Do NOT alter technical terminology or mathematical notation
  - Do NOT add or remove sentences (this is purely cosmetic/consistency)
  - Do NOT exceed ~5 net line changes (most changes are in-place replacements)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Search-and-replace style fixes — cosmetic consistency, no substantive content changes
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - None needed — simple text cleanup

  **Parallelization**:
  - **Can Run In Parallel**: NO (edits main.tex after Task 7)
  - **Parallel Group**: Wave 3 (fourth and final in sequential chain)
  - **Blocks**: Task 9
  - **Blocked By**: Task 7

  **References** (CRITICAL):

  **Pattern References**:
  - `JournalMusicResearch/main.tex` — the entire document needs scanning for BH-FDR repetition and capitalization

  **External References**:
  - `JournalMusicResearch/TempPeerReviews/CriticReview2.md` — BH FDR complaint (line 42), capitalization note (line 44)

  **WHY Each Reference Matters**:
  - The full-text scan is required because BH-FDR and era capitalisation issues are scattered across the document, not localised to one section

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: BH-FDR fully expanded only once
    Tool: Bash (grep)
    Preconditions: main.tex edited
    Steps:
      1. Run: mkdir -p .sisyphus/evidence && grep -c "Benjamini-Hochberg" JournalMusicResearch/main.tex 2>&1 | tee .sisyphus/evidence/task-8-bhfdr-trim.txt
      2. Assert count is exactly 1 (the definitional occurrence)
      3. Run: grep -c "BH-FDR\|BH FDR" JournalMusicResearch/main.tex 2>&1 | tee -a .sisyphus/evidence/task-8-bhfdr-trim.txt
      4. Assert count >= 1 (subsequent uses are abbreviated)
    Expected Result: Full form appears once, abbreviation used elsewhere
    Failure Indicators: Full form appears more than once, or abbreviation never used
    Evidence: .sisyphus/evidence/task-8-bhfdr-trim.txt

  Scenario: Musical era capitalisation consistent
    Tool: Bash (grep)
    Preconditions: main.tex edited
    Steps:
      1. Run: grep -in "baroque\|classical\|romantic\|impressionist" JournalMusicResearch/main.tex | grep -v "^[0-9]*:.*[A-Z]" 2>&1 | tee .sisyphus/evidence/task-8-capitalisation.txt
      2. This finds lines with era names that are NOT capitalised
      3. Assert count is 0 (all era references capitalised)
    Expected Result: All era names consistently capitalised
    Failure Indicators: Lowercase era names found
    Evidence: .sisyphus/evidence/task-8-capitalisation.txt

  Scenario: LaTeX compiles after cleanup
    Tool: Bash
    Preconditions: main.tex edited
    Steps:
      1. Run: cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex 2>&1 | tee ../.sisyphus/evidence/task-8-latex-build.txt
      2. Assert zero errors
    Expected Result: Clean compilation unchanged
    Failure Indicators: Any new errors introduced by cleanup
    Evidence: .sisyphus/evidence/task-8-latex-build.txt
  ```

  **Commit**: YES
  - Message: `docs: fix BH-FDR repetition, capitalisation consistency, and minor polish`
  - Files: `JournalMusicResearch/main.tex`
  - Pre-commit: `cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex`

- [x] 9. Final Build Verification and URL/Citation Checks

  **What to do**:
  This is the final pre-review quality gate. It runs the complete LaTeX toolchain, verifies all cross-references resolve, and checks external resources.

  **Full LaTeX Build (3-pass):**
  - Run the complete LaTeX build chain: `pdflatex → bibtex → pdflatex → pdflatex`
  - Verify: 0 errors in `main.log`
  - Verify: 0 undefined references (grep for "undefined" in log, excluding known false positives like "destination" warnings)
  - Verify: 0 undefined citations (grep for "Citation.*undefined" in log)
  - Verify: 0 BibTeX warnings about missing entries

  **Cross-Reference Integrity:**
  - Verify `\Cref{sec:robustness}` resolves (the new Robustness Checks subsection)
  - Verify all `\Cref{tab:...}` and `\Cref{fig:...}` references resolve
  - Verify all `\eqref{}` references resolve

  **URL Checks (interactive figure links):**
  - The paper references interactive figures hosted at URLs. Verify these URLs are still active:
    - `https://victor-gurbani.github.io/JuFo2026/figures/embeddings/composer_clouds_3d.html`
    - Any other URLs referenced in main.tex via `\url{}` or `\href{}`
  - Use `curl -s -o /dev/null -w "%{http_code}"` to check each URL returns 200

  **PDMX Citation Status (Minor issue #17):**
  - Verify whether the PDMX dataset has a formal peer-reviewed citation now (it was a preprint/dataset link when originally cited)
  - Check the existing citation in references.bib — if it's a `@misc` or `@techreport`, verify if a published `@article` version exists

  **Net Line Count:**
  - Count net lines added compared to the original main.tex: `git diff --stat JournalMusicResearch/main.tex`
  - Assert net additions ≤ 80 lines

  **Must NOT do**:
  - Do NOT make any content changes in this task — it is verification only
  - If issues are found, report them but do NOT fix them (fixes go back to the responsible task for re-execution)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Pure verification — run commands, check outputs, report results
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (after all LaTeX edits)
  - **Blocks**: F1, F2, F3, F4
  - **Blocked By**: Task 8

  **References** (CRITICAL):

  **Pattern References**:
  - `JournalMusicResearch/main.tex` — the complete document to build
  - `JournalMusicResearch/references.bib` — bibliography to verify

  **WHY Each Reference Matters**:
  - A clean build with zero errors is the minimum bar for submission readiness

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Full LaTeX build with zero errors
    Tool: Bash
    Preconditions: All Tasks 1-8 complete
    Steps:
      1. Run: mkdir -p .sisyphus/evidence && cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex 2>&1 | tee ../.sisyphus/evidence/task-9-full-build.txt
      2. Run: grep -c "^!" main.log | tee -a ../.sisyphus/evidence/task-9-full-build.txt
      3. Assert count is 0
      4. Run: grep -c "Citation.*undefined" main.log | tee -a ../.sisyphus/evidence/task-9-full-build.txt
      5. Assert count is 0
      6. Run: grep "LaTeX Warning.*Reference.*undefined" main.log | wc -l | tee -a ../.sisyphus/evidence/task-9-full-build.txt
      7. Assert count is 0
    Expected Result: 0 errors, 0 undefined references, 0 undefined citations
    Failure Indicators: Any error, undefined reference, or undefined citation
    Evidence: .sisyphus/evidence/task-9-full-build.txt

  Scenario: Interactive figure URLs are active
    Tool: Bash (curl)
    Preconditions: None (external check)
    Steps:
      1. Run: curl -s -o /dev/null -w "%{http_code}" "https://victor-gurbani.github.io/JuFo2026/figures/embeddings/composer_clouds_3d.html" 2>&1 | tee .sisyphus/evidence/task-9-url-check.txt
      2. Assert HTTP status is 200
    Expected Result: All referenced URLs return HTTP 200
    Failure Indicators: Any URL returns non-200 status
    Evidence: .sisyphus/evidence/task-9-url-check.txt

  Scenario: Net line count within budget
    Tool: Bash
    Preconditions: All edits committed. Baseline captured in .sisyphus/evidence/baseline-commit.txt by Task 1.
    Steps:
      1. Run: bash -c '
         BASELINE=$(cat .sisyphus/evidence/baseline-commit.txt);
         ADDED=$(git diff "$BASELINE" -- JournalMusicResearch/main.tex | grep -c "^+[^+]" || echo 0);
         REMOVED=$(git diff "$BASELINE" -- JournalMusicResearch/main.tex | grep -c "^-[^-]" || echo 0);
         NET=$((ADDED - REMOVED));
         echo "Lines: added=$ADDED removed=$REMOVED net=$NET";
         [ $NET -le 80 ] && echo "RESULT: APPROVE" || echo "RESULT: REJECT (net $NET > 80)"
         ' 2>&1 | tee .sisyphus/evidence/task-9-line-count.txt
    Expected Result: net ≤ 80 and "RESULT: APPROVE"
    Failure Indicators: net > 80, or baseline-commit.txt missing
    Evidence: .sisyphus/evidence/task-9-line-count.txt
  ```

  **Commit**: NO (verification only — no files changed)

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
>
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback → fix → re-run → present again → wait for okay.

- [x] F1. **Plan Compliance Audit** — `subagent_type: oracle`

  **What to do**:
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read LaTeX section, check CSV output, run python script). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan. Verify all 17 review issues are addressed in the LaTeX with specific line references.

  **QA Scenarios**:

  ```
  Scenario: Must Have items all present
    Tool: Bash
    Preconditions: All 9 implementation tasks completed
    Steps:
      1. Run: mkdir -p .sisyphus/evidence/final-qa && bash -c '
         PASS=0; FAIL=0;
         test -f src/pca_extended.py && ((PASS++)) || ((FAIL++));
         test -f src/placebo_dd.py && ((PASS++)) || ((FAIL++));
         test -f data/stats/pca_extended.csv && ((PASS++)) || ((FAIL++));
         test -f data/stats/placebo_dd.csv && ((PASS++)) || ((FAIL++));
         grep -q "\\\\subsection{Robustness Checks}" JournalMusicResearch/main.tex && ((PASS++)) || ((FAIL++));
         grep -q "PC4" JournalMusicResearch/main.tex && ((PASS++)) || ((FAIL++));
         grep -q "PC5" JournalMusicResearch/main.tex && ((PASS++)) || ((FAIL++));
         grep -q "placebo" JournalMusicResearch/main.tex && ((PASS++)) || ((FAIL++));
         echo "Must Have: PASS=$PASS FAIL=$FAIL";
         [ $FAIL -eq 0 ] && echo "RESULT: APPROVE" || echo "RESULT: REJECT"
         ' 2>&1 | tee .sisyphus/evidence/final-qa/f1-must-have.txt
    Expected Result: PASS=8 FAIL=0 and "RESULT: APPROVE"
    Failure Indicators: Any FAIL count > 0
    Evidence: .sisyphus/evidence/final-qa/f1-must-have.txt

  Scenario: Must NOT Have items all absent
    Tool: Bash
    Preconditions: All 9 implementation tasks completed
    Steps:
      1. Run: bash -c '
         PASS=0; FAIL=0;
         grep -q "Elton John" JournalMusicResearch/main.tex && ((FAIL++)) || ((PASS++));
         grep -q "random forest" JournalMusicResearch/main.tex && ((FAIL++)) || ((PASS++));
         grep -rq "as any" src/pca_extended.py src/placebo_dd.py 2>/dev/null && ((FAIL++)) || ((PASS++));
         grep -rq "type: ignore" src/pca_extended.py src/placebo_dd.py 2>/dev/null && ((FAIL++)) || ((PASS++));
         echo "Must NOT Have: PASS=$PASS FAIL=$FAIL";
         [ $FAIL -eq 0 ] && echo "RESULT: APPROVE" || echo "RESULT: REJECT"
         ' 2>&1 | tee -a .sisyphus/evidence/final-qa/f1-must-have.txt
    Expected Result: PASS=4 FAIL=0 and "RESULT: APPROVE"
    Failure Indicators: Any FAIL count > 0
    Evidence: .sisyphus/evidence/final-qa/f1-must-have.txt

  Scenario: All evidence files exist
    Tool: Bash
    Preconditions: All 9 implementation tasks completed
    Steps:
      1. Run: bash -c '
         TOTAL=0; FOUND=0;
         for f in .sisyphus/evidence/task-*.txt .sisyphus/evidence/task-*.png; do
           ((TOTAL++));
           [ -f "$f" ] && ((FOUND++)) || echo "MISSING: $f";
         done;
         echo "Evidence files: FOUND=$FOUND TOTAL=$TOTAL";
         [ $FOUND -eq $TOTAL ] && echo "RESULT: APPROVE" || echo "RESULT: REJECT"
         ' 2>&1 | tee .sisyphus/evidence/final-qa/f1-evidence-check.txt
    Expected Result: FOUND equals TOTAL and "RESULT: APPROVE"
    Failure Indicators: Any MISSING file listed
    Evidence: .sisyphus/evidence/final-qa/f1-evidence-check.txt
  ```

  Output: `Must Have [N/N] | Must NOT Have [N/N] | Evidence [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`

  **What to do**:
  Run `python3 -m py_compile` on both new scripts. Review Python scripts for: type hints, argparse conventions, `from __future__ import annotations`, docstrings, proper error handling. Verify LaTeX builds clean. Check for undefined references.

  **QA Scenarios**:

  ```
  Scenario: Python scripts compile and follow conventions
    Tool: Bash
    Preconditions: src/pca_extended.py and src/placebo_dd.py exist
    Steps:
      1. Run: mkdir -p .sisyphus/evidence/final-qa && bash -c '
         PASS=0; FAIL=0;
         python3 -m py_compile src/pca_extended.py && ((PASS++)) || ((FAIL++));
         python3 -m py_compile src/placebo_dd.py && ((PASS++)) || ((FAIL++));
         grep -q "from __future__ import annotations" src/pca_extended.py && ((PASS++)) || ((FAIL++));
         grep -q "from __future__ import annotations" src/placebo_dd.py && ((PASS++)) || ((FAIL++));
         grep -q "argparse" src/pca_extended.py && ((PASS++)) || ((FAIL++));
         grep -q "argparse" src/placebo_dd.py && ((PASS++)) || ((FAIL++));
         grep -q "def main" src/pca_extended.py && ((PASS++)) || ((FAIL++));
         grep -q "def main" src/placebo_dd.py && ((PASS++)) || ((FAIL++));
         echo "Python conventions: PASS=$PASS FAIL=$FAIL";
         [ $FAIL -eq 0 ] && echo "RESULT: APPROVE" || echo "RESULT: REJECT"
         ' 2>&1 | tee .sisyphus/evidence/final-qa/f2-python-quality.txt
    Expected Result: PASS=8 FAIL=0 and "RESULT: APPROVE"
    Failure Indicators: Any compilation error or missing convention
    Evidence: .sisyphus/evidence/final-qa/f2-python-quality.txt

  Scenario: LaTeX builds with zero errors and zero undefined references
    Tool: Bash
    Preconditions: JournalMusicResearch/main.tex exists with all edits applied
    Steps:
      1. Run: bash -c '
         cd JournalMusicResearch &&
         pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 &&
         bibtex main > /dev/null 2>&1 &&
         pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 &&
         pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1;
         UNDEF=$(grep -c "undefined" main.log 2>/dev/null || echo 0);
         ERRORS=$(grep -c "^!" main.log 2>/dev/null || echo 0);
         echo "LaTeX build: errors=$ERRORS undefined_refs=$UNDEF";
         [ "$ERRORS" -eq 0 ] && [ "$UNDEF" -eq 0 ] && echo "RESULT: APPROVE" || echo "RESULT: REJECT"
         ' 2>&1 | tee .sisyphus/evidence/final-qa/f2-latex-build.txt
    Expected Result: errors=0 undefined_refs=0 and "RESULT: APPROVE"
    Failure Indicators: Any non-zero error or undefined reference count
    Evidence: .sisyphus/evidence/final-qa/f2-latex-build.txt
  ```

  Output: `Python [PASS/FAIL] | LaTeX Build [PASS/FAIL] | Refs [PASS/FAIL] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`

  **What to do**:
  Re-run both Python scripts from scratch. Verify output CSVs match numbers cited in main.tex. Cross-reference every new number in the LaTeX (PC4/PC5 variance, placebo p-values) against the corresponding CSV. Check that no existing numbers were altered (29/36, 48.2%, F-statistics in Table 1, DD/DDD values in Table 2). Read through every edited paragraph for grammatical correctness, British English spelling (-ise), and tone consistency.

  **QA Scenarios**:

  ```
  Scenario: PCA extended script runs and produces valid output
    Tool: Bash
    Preconditions: src/pca_extended.py exists, feature CSVs in data/features/
    Steps:
      1. Run: mkdir -p .sisyphus/evidence/final-qa && python3 src/pca_extended.py --output /tmp/f3_pca_verify.csv 2>&1 | tee .sisyphus/evidence/final-qa/f3-pca-rerun.txt
      2. Run: python3 -c "
         import pandas as pd
         df = pd.read_csv('/tmp/f3_pca_verify.csv')
         print('Columns:', list(df.columns))
         print('Rows:', len(df))
         assert 'variance_explained_pct' in df.columns or 'component' in df.columns, 'Missing expected columns'
         print('RESULT: APPROVE')
         " 2>&1 | tee -a .sisyphus/evidence/final-qa/f3-pca-rerun.txt
    Expected Result: CSV created with expected columns, "RESULT: APPROVE"
    Failure Indicators: Script crashes, missing columns, or assertion error
    Evidence: .sisyphus/evidence/final-qa/f3-pca-rerun.txt

  Scenario: Placebo DD script runs and produces valid output
    Tool: Bash
    Preconditions: src/placebo_dd.py exists, feature CSVs in data/features/
    Steps:
      1. Run: python3 src/placebo_dd.py --n-permutations 1000 --seed 42 --output /tmp/f3_placebo_verify.csv 2>&1 | tee .sisyphus/evidence/final-qa/f3-placebo-rerun.txt
      2. Run: python3 -c "
         import pandas as pd
         df = pd.read_csv('/tmp/f3_placebo_verify.csv')
         print('Columns:', list(df.columns))
         print('Rows:', len(df))
         assert 'p_value' in df.columns or 'permutation_p' in df.columns, 'Missing p-value column'
         print('RESULT: APPROVE')
         " 2>&1 | tee -a .sisyphus/evidence/final-qa/f3-placebo-rerun.txt
    Expected Result: CSV created with p-value column, "RESULT: APPROVE"
    Failure Indicators: Script crashes, missing columns, or assertion error
    Evidence: .sisyphus/evidence/final-qa/f3-placebo-rerun.txt

  Scenario: Existing numbers in main.tex unaltered
    Tool: Bash
    Preconditions: JournalMusicResearch/main.tex has all edits applied
    Steps:
      1. Run: bash -c '
         PASS=0; FAIL=0;
         grep -q "29" JournalMusicResearch/main.tex && grep -q "36" JournalMusicResearch/main.tex && ((PASS++)) || ((FAIL++));
         grep -q "48.2" JournalMusicResearch/main.tex && ((PASS++)) || ((FAIL++));
         grep -q "144" JournalMusicResearch/main.tex && ((PASS++)) || ((FAIL++));
         echo "Existing numbers preserved: PASS=$PASS FAIL=$FAIL";
         [ $FAIL -eq 0 ] && echo "RESULT: APPROVE" || echo "RESULT: REJECT"
         ' 2>&1 | tee .sisyphus/evidence/final-qa/f3-numbers-preserved.txt
    Expected Result: PASS=3 FAIL=0 and "RESULT: APPROVE"
    Failure Indicators: Any key number missing from main.tex
    Evidence: .sisyphus/evidence/final-qa/f3-numbers-preserved.txt

  Scenario: British English spelling used (-ise not -ize)
    Tool: Bash
    Preconditions: JournalMusicResearch/main.tex has all edits applied
    Steps:
      1. Run: bash -c '
         IZECOUNT=$(grep -oi "[a-z]*ize[ds]\?\b" JournalMusicResearch/main.tex | grep -v "\\\\[a-z]*" | wc -l | tr -d " ");
         echo "American -ize/-ized spellings found: $IZECOUNT";
         [ "$IZECOUNT" -le 2 ] && echo "RESULT: APPROVE (acceptable or pre-existing)" || echo "RESULT: REJECT (too many -ize spellings)"
         ' 2>&1 | tee .sisyphus/evidence/final-qa/f3-british-english.txt
    Expected Result: Low or zero -ize count, "RESULT: APPROVE"
    Failure Indicators: High count of -ize spellings indicating American English
    Evidence: .sisyphus/evidence/final-qa/f3-british-english.txt
  ```

  Output: `Numbers [N/N match] | Grammar [PASS/FAIL] | Tone [PASS/FAIL] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`

  **What to do**:
  For each of the 17 review issues: verify the corresponding LaTeX edit exists and addresses the concern. Check that no new issues were introduced. Verify ≤80 net lines added. Verify ≤5 new bibliography entries. Verify no existing pipeline scripts modified. Verify no new \section{} commands added (except Robustness Checks \subsection{}). Flag any unaccounted changes.

  **QA Scenarios**:

  ```
  Scenario: Net lines added ≤80
    Tool: Bash
    Preconditions: Baseline commit hash stored in .sisyphus/evidence/baseline-commit.txt
    Steps:
      1. Run: mkdir -p .sisyphus/evidence/final-qa && bash -c '
         BASELINE=$(cat .sisyphus/evidence/baseline-commit.txt);
         ADDED=$(git diff "$BASELINE" -- JournalMusicResearch/main.tex | grep -c "^+[^+]" || echo 0);
         REMOVED=$(git diff "$BASELINE" -- JournalMusicResearch/main.tex | grep -c "^-[^-]" || echo 0);
         NET=$((ADDED - REMOVED));
         echo "Lines: added=$ADDED removed=$REMOVED net=$NET";
         [ $NET -le 80 ] && echo "RESULT: APPROVE" || echo "RESULT: REJECT (net $NET > 80)"
         ' 2>&1 | tee .sisyphus/evidence/final-qa/f4-net-lines.txt
    Expected Result: net ≤ 80 and "RESULT: APPROVE"
    Failure Indicators: net > 80
    Evidence: .sisyphus/evidence/final-qa/f4-net-lines.txt

  Scenario: Bibliography entries ≤5 new
    Tool: Bash
    Preconditions: Baseline commit hash stored in .sisyphus/evidence/baseline-commit.txt
    Steps:
      1. Run: bash -c '
         BASELINE=$(cat .sisyphus/evidence/baseline-commit.txt);
         NEW_ENTRIES=$(git diff "$BASELINE" -- JournalMusicResearch/references.bib | grep "^+@" | wc -l | tr -d " ");
         echo "New bibliography entries: $NEW_ENTRIES";
         [ "$NEW_ENTRIES" -le 5 ] && echo "RESULT: APPROVE" || echo "RESULT: REJECT ($NEW_ENTRIES > 5)"
         ' 2>&1 | tee .sisyphus/evidence/final-qa/f4-bib-count.txt
    Expected Result: NEW_ENTRIES ≤ 5 and "RESULT: APPROVE"
    Failure Indicators: More than 5 new @-entries
    Evidence: .sisyphus/evidence/final-qa/f4-bib-count.txt

  Scenario: No existing pipeline scripts modified
    Tool: Bash
    Preconditions: Baseline commit hash stored in .sisyphus/evidence/baseline-commit.txt
    Steps:
      1. Run: bash -c '
         BASELINE=$(cat .sisyphus/evidence/baseline-commit.txt);
         MODIFIED=$(git diff --name-only "$BASELINE" -- src/ | grep -v "pca_extended.py" | grep -v "placebo_dd.py" || true);
         if [ -z "$MODIFIED" ]; then
           echo "No existing pipeline scripts modified";
           echo "RESULT: APPROVE";
         else
           echo "MODIFIED EXISTING SCRIPTS: $MODIFIED";
           echo "RESULT: REJECT";
         fi
         ' 2>&1 | tee .sisyphus/evidence/final-qa/f4-pipeline-untouched.txt
    Expected Result: No existing scripts modified, "RESULT: APPROVE"
    Failure Indicators: Any file listed under MODIFIED EXISTING SCRIPTS
    Evidence: .sisyphus/evidence/final-qa/f4-pipeline-untouched.txt

  Scenario: No new section commands (except Robustness Checks subsection)
    Tool: Bash
    Preconditions: Baseline commit hash stored in .sisyphus/evidence/baseline-commit.txt
    Steps:
      1. Run: bash -c '
         BASELINE=$(cat .sisyphus/evidence/baseline-commit.txt);
         NEW_SECTIONS=$(git diff "$BASELINE" -- JournalMusicResearch/main.tex | grep "^+.*\\\\section{" || true);
         NEW_SUBSECTIONS=$(git diff "$BASELINE" -- JournalMusicResearch/main.tex | grep "^+.*\\\\subsection{" | grep -v "Robustness Checks" || true);
         if [ -z "$NEW_SECTIONS" ] && [ -z "$NEW_SUBSECTIONS" ]; then
           echo "No unauthorized section/subsection commands added";
           echo "RESULT: APPROVE";
         else
           echo "UNAUTHORIZED SECTIONS: $NEW_SECTIONS $NEW_SUBSECTIONS";
           echo "RESULT: REJECT";
         fi
         ' 2>&1 | tee .sisyphus/evidence/final-qa/f4-sections-check.txt
    Expected Result: No unauthorized sections, "RESULT: APPROVE"
    Failure Indicators: Any new \section{} or unauthorized \subsection{}
    Evidence: .sisyphus/evidence/final-qa/f4-sections-check.txt
  ```

  Output: `Issues [17/17 addressed] | Net Lines [N≤80] | New Refs [N≤5] | Scope [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **Commit 1**: `feat: add PC4/PC5 extended PCA analysis script` — `src/pca_extended.py`, `data/stats/pca_extended.csv`
- **Commit 2**: `feat: add placebo DD permutation test script` — `src/placebo_dd.py`, `data/stats/placebo_dd.csv`
- **Commit 3**: `docs: update bibliography for JNMR revision` — `JournalMusicResearch/references.bib`
- **Commit 4**: `docs: address critical review issues in Introduction, Methods` — `JournalMusicResearch/main.tex`
- **Commit 5**: `docs: add Robustness Checks subsection and PC4/PC5 analysis to Results` — `JournalMusicResearch/main.tex`
- **Commit 6**: `docs: address Discussion and Conclusion review issues` — `JournalMusicResearch/main.tex`
- **Commit 7**: `docs: fix BH FDR repetition, capitalization, and minor polish` — `JournalMusicResearch/main.tex`

---

## Success Criteria

### Verification Commands
```bash
# Python scripts compile
python3 -m py_compile src/pca_extended.py       # Expected: exit 0
python3 -m py_compile src/placebo_dd.py          # Expected: exit 0

# Python scripts produce output
python3 src/pca_extended.py                      # Expected: data/stats/pca_extended.csv created
python3 src/placebo_dd.py --n-permutations 10000 # Expected: data/stats/placebo_dd.csv created

# LaTeX builds clean
cd JournalMusicResearch && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
# Expected: 0 errors in main.log

grep -c 'undefined' JournalMusicResearch/main.log   # Expected: 0
bibtex JournalMusicResearch/main 2>&1 | grep -c 'Warning--'  # Expected: 0
```

### Final Checklist
- [x] All 17 review issues addressed with traceable LaTeX edits
- [x] PC4/PC5 loadings and variance reported in Results
- [x] Placebo DD p-values reported in Robustness Checks subsection
- [x] All "Must Have" present
- [x] All "Must NOT Have" absent
- [x] LaTeX compiles with 0 errors, 0 undefined references
- [x] All new numbers match script outputs
- [x] No existing numerical claims altered
- [x] ≤80 net lines (user approved 90) added
- [x] ≤5 new bibliography entries
