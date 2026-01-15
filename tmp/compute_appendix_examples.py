from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "features"

FILES = {
    "harmonic": FEATURE_DIR / "harmonic_features.csv",
    "melodic": FEATURE_DIR / "melodic_features.csv",
    "rhythmic": FEATURE_DIR / "rhythmic_features.csv",
}

CANDIDATES: dict[str, list[str]] = {
    "harmonic": [
        "chord_event_count",
        "chord_quality_other_pct",
        "harmonic_density_mean",
        "dissonance_ratio",
        "passing_tone_ratio",
        "appoggiatura_ratio",
        "deceptive_cadence_ratio",
        "modal_interchange_ratio",
    ],
    "melodic": [
        "note_count",
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
    "rhythmic": [
        "note_event_count",
        "avg_note_duration",
        "notes_per_beat",
        "downbeat_emphasis_ratio",
        "syncopation_ratio",
        "rhythmic_pattern_entropy",
        "micro_rhythmic_density",
        "cross_rhythm_ratio",
    ],
}


def pick_extremes(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series] | None:
    if col not in df.columns:
        return None

    s = pd.to_numeric(df[col], errors="coerce")
    ok = s.notna() & np.isfinite(s)
    if not ok.any():
        return None

    # Column names in this repo's CSVs.
    meta = ["composer_label", "title", col]
    sdf = df.loc[ok, meta].copy()
    sdf[col] = pd.to_numeric(sdf[col], errors="coerce")

    lo = sdf.nsmallest(1, col).iloc[0]
    hi = sdf.nlargest(1, col).iloc[0]
    return lo, hi


def pick_high_for_composer(df: pd.DataFrame, col: str, composer: str) -> pd.Series | None:
    if col not in df.columns:
        return None
    if "composer_label" not in df.columns:
        return None

    sdf = df.loc[df["composer_label"].astype(str) == composer].copy()
    if sdf.empty:
        return None
    s = pd.to_numeric(sdf[col], errors="coerce")
    ok = s.notna() & np.isfinite(s)
    if not ok.any():
        return None
    meta = ["composer_label", "title", col]
    sdf = sdf.loc[ok, meta].copy()
    sdf[col] = pd.to_numeric(sdf[col], errors="coerce")
    return sdf.nlargest(1, col).iloc[0]


def fmt(row: pd.Series, col: str) -> str:
    composer = str(row["composer_label"]).strip()
    title = str(row["title"]).strip()
    val = float(row[col])

    # Heuristic formatting: percentages/ratios get 3 decimals, counts get ints.
    if col.endswith("_count") or col.endswith("_event_count") or col.endswith("_chord_count"):
        v = f"{int(round(val)):,}".replace(",", "\u202f")  # narrow no-break space
    elif abs(val) >= 10 and abs(val - round(val)) < 1e-9:
        v = f"{int(val)}"
    else:
        v = f"{val:.3f}".rstrip("0").rstrip(".")

    return f"{composer} — {title} ({col}={v})"


def main() -> None:
    for name, path in FILES.items():
        df = pd.read_csv(path)
        print(f"\n== {name} ({path.name}) ==")
        for col in CANDIDATES[name]:
            ext = pick_extremes(df, col)
            if ext is None:
                continue
            lo, hi = ext
            print(f"{col}:")
            print(f"  low:  {fmt(lo, col)}")
            print(f"  high: {fmt(hi, col)}")
            for composer in ("Bach", "Mozart", "Chopin", "Debussy"):
                row = pick_high_for_composer(df, col, composer)
                if row is None:
                    continue
                print(f"  {composer} high: {fmt(row, col)}")


if __name__ == "__main__":
    main()
