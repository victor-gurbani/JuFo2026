import pandas as pd

harm = pd.read_csv("data/features/harmonic_features.csv")
mel = pd.read_csv("data/features/melodic_features.csv")
rhy = pd.read_csv("data/features/rhythmic_features.csv")


def show(df: pd.DataFrame, cols: list[str]) -> None:
    grouped = df.groupby("composer_label")[cols].mean(numeric_only=True)
    grouped = grouped.round(3)
    print(grouped.to_string())


print("melodic means:")
show(
    mel,
    [
        "pitch_range_semitones",
        "avg_melodic_interval",
        "melodic_interval_std",
        "conjunct_motion_ratio",
        "melodic_leap_ratio",
        "pitch_class_entropy",
        "voice_independence_index",
        "contrary_motion_ratio",
        "parallel_motion_ratio",
        "oblique_motion_ratio",
    ],
)

print("\nrhythmic means:")
show(
    rhy,
    [
        "avg_note_duration",
        "std_note_duration",
        "notes_per_beat",
        "downbeat_emphasis_ratio",
        "syncopation_ratio",
        "rhythmic_pattern_entropy",
        "micro_rhythmic_density",
        "cross_rhythm_ratio",
    ],
)

print("\nharmonic means:")
show(
    harm,
    [
        "dissonance_ratio",
        "harmonic_density_mean",
        "modal_interchange_ratio",
        "passing_tone_ratio",
        "appoggiatura_ratio",
        "other_dissonance_ratio",
        "deceptive_cadence_ratio",
        "chord_quality_other_pct",
        "chord_quality_augmented_pct",
    ],
)
