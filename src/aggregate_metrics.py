"""Aggregate headline statistics used in the project documentation.

Run this script from the repository root:
    python3 src/aggregate_metrics.py

It prints counts covering the raw catalog, curated corpus, feature families,
parsed structure, and inferential tests so readers can verify the numbers that
appear in the READMEs and articles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def hours_from_quarters(total_quarters: float, bpm: float) -> float:
    """Convert accumulated quarter-note durations to hours at a given tempo."""

    seconds = total_quarters * 60.0 / bpm
    return seconds / 3600.0


def safe_sum(items: Iterable[float]) -> float:
    """Sum values, treating ``None`` as zero."""

    return float(sum(value or 0.0 for value in items))


def main() -> None:
    catalog_path = ROOT / "15571083" / "PDMX.csv"
    summaries_path = DATA_DIR / "parsed" / "summaries.json"
    harmonic_path = DATA_DIR / "features" / "harmonic_features.csv"
    melodic_path = DATA_DIR / "features" / "melodic_features.csv"
    rhythmic_path = DATA_DIR / "features" / "rhythmic_features.csv"
    anova_path = DATA_DIR / "stats" / "anova_summary.csv"
    tukey_path = DATA_DIR / "stats" / "tukey_hsd.csv"

    catalog_df = pd.read_csv(catalog_path)
    catalog_size = len(catalog_df)

    summaries = json.loads(summaries_path.read_text())
    curated_count = len(summaries)
    total_measures = safe_sum(item.get("measures") for item in summaries)
    total_parts = safe_sum(item.get("parts") for item in summaries)
    total_quarters = safe_sum(item.get("duration_quarters") for item in summaries)
    avg_measures = total_measures / curated_count if curated_count else 0.0
    avg_parts = total_parts / curated_count if curated_count else 0.0
    hours_90_bpm = hours_from_quarters(total_quarters, bpm=90.0)

    harmonic_df = pd.read_csv(harmonic_path)
    melodic_df = pd.read_csv(melodic_path)
    rhythmic_df = pd.read_csv(rhythmic_path)

    def feature_columns(df: pd.DataFrame) -> list[str]:
        exclude = {"composer_label", "title", "mxl_path"}
        return [col for col in df.columns if col not in exclude]

    harmonic_features = feature_columns(harmonic_df)
    melodic_features = feature_columns(melodic_df)
    rhythmic_features = feature_columns(rhythmic_df)
    total_features = (
        len(harmonic_features) + len(melodic_features) + len(rhythmic_features)
    )

    per_composer_counts = harmonic_df["composer_label"].value_counts().to_dict()

    anova_df = pd.read_csv(anova_path)
    alpha = 0.05
    total_tests = len(anova_df)
    significant_features = int((anova_df["p_value"] < alpha).sum())

    tukey_df = pd.read_csv(tukey_path)
    if "reject" in tukey_df.columns:
        significant_pairs = int(tukey_df["reject"].sum())
    elif "p_adj" in tukey_df.columns:
        significant_pairs = int((tukey_df["p_adj"] < alpha).sum())
    else:
        significant_pairs = 0
    total_pairs = len(tukey_df)

    print("Catalog size:", catalog_size)
    print("Curated solo piano pieces:", curated_count)
    print("Per-composer counts:", per_composer_counts)
    print("Total measures:", int(total_measures))
    print("Average measures per piece:", round(avg_measures, 2))
    print("Average parts per piece:", round(avg_parts, 2))
    print("Total quarter-note durations:", round(total_quarters, 1))
    print("Listening hours at 90 BPM:", round(hours_90_bpm, 2))
    print("Harmonic features:", len(harmonic_features))
    print("Melodic features:", len(melodic_features))
    print("Rhythmic features:", len(rhythmic_features))
    print("Total features tracked:", total_features)
    print("ANOVA tests run (alpha=0.05):", total_tests)
    print("Significant features:", significant_features)
    print("Tukey contrasts evaluated:", total_pairs)
    print("Significant contrasts:", significant_pairs)


if __name__ == "__main__":
    main()
