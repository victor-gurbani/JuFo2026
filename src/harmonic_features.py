"""Extract harmonic features from the curated solo-piano corpus."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from music21 import chord, interval, note, roman, stream

# Use a non-interactive backend to support headless environments
matplotlib.use("Agg")

from score_parser import parse_score  # type: ignore

DEFAULT_CORPUS = Path("data/curated/solo_piano_corpus.csv")
DEFAULT_PATHS = Path("data/curated/solo_piano_mxl_paths.txt")
DEFAULT_OUTPUT = Path("data/features/harmonic_features.csv")
DEFAULT_FIGURE_DIR = Path("figures/harmonic")

CHORD_MATCH_TOLERANCE = 1e-3

QUALITY_TARGETS = ("major", "minor", "diminished", "augmented")


@dataclass
class HarmonicFeatureResult:
    composer_label: str
    title: Optional[str]
    mxl_path: Path
    chord_event_count: int
    chord_quality_total: int
    chord_quality_major_pct: float
    chord_quality_minor_pct: float
    chord_quality_diminished_pct: float
    chord_quality_augmented_pct: float
    chord_quality_other_pct: float
    harmonic_density_mean: Optional[float]
    dissonance_ratio: Optional[float]
    dissonant_note_count: int
    passing_tone_ratio: Optional[float]
    appoggiatura_ratio: Optional[float]
    other_dissonance_ratio: Optional[float]
    roman_chord_count: int
    deceptive_cadence_ratio: Optional[float]
    modal_interchange_ratio: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "composer_label": self.composer_label,
            "title": self.title,
            "mxl_path": str(self.mxl_path),
            "chord_event_count": self.chord_event_count,
            "chord_quality_total": self.chord_quality_total,
            "chord_quality_major_pct": self.chord_quality_major_pct,
            "chord_quality_minor_pct": self.chord_quality_minor_pct,
            "chord_quality_diminished_pct": self.chord_quality_diminished_pct,
            "chord_quality_augmented_pct": self.chord_quality_augmented_pct,
            "chord_quality_other_pct": self.chord_quality_other_pct,
            "harmonic_density_mean": self.harmonic_density_mean,
            "dissonance_ratio": self.dissonance_ratio,
            "dissonant_note_count": self.dissonant_note_count,
            "passing_tone_ratio": self.passing_tone_ratio,
            "appoggiatura_ratio": self.appoggiatura_ratio,
            "other_dissonance_ratio": self.other_dissonance_ratio,
            "roman_chord_count": self.roman_chord_count,
            "deceptive_cadence_ratio": self.deceptive_cadence_ratio,
            "modal_interchange_ratio": self.modal_interchange_ratio,
        }


def load_corpus(csv_path: Path, paths_path: Optional[Path] = None) -> pd.DataFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "mxl_abs_path" not in df.columns:
            raise ValueError(f"Missing 'mxl_abs_path' column in {csv_path}")
        return df
    if paths_path and paths_path.exists():
        paths = [line.strip() for line in paths_path.read_text().splitlines() if line.strip()]
        return pd.DataFrame({"mxl_abs_path": paths})
    raise FileNotFoundError("No curated corpus CSV or path list found.")


def iter_score_records(df: pd.DataFrame) -> Iterable[Tuple[str, Optional[str], Path]]:
    for _, row in df.iterrows():
        composer = str(row.get("composer_label") or "").strip()
        title = row.get("title") if isinstance(row.get("title"), str) else None
        path_value = row.get("mxl_abs_path") or row.get("mxl_path")
        if not isinstance(path_value, str):
            continue
        yield composer, title, Path(path_value).expanduser().resolve()


def chordify_score(score: stream.Score) -> stream.Stream:
    ch = score.chordify()
    return ch.flatten()


def extract_chords(chordified: stream.Stream) -> List[chord.Chord]:
    chords = []
    for ch in chordified.recurse().getElementsByClass(chord.Chord):
        if not ch.pitches:
            continue
        if ch.duration.quarterLength is None or ch.duration.quarterLength <= 0:
            continue
        chords.append(ch)
    return chords


def chord_quality_metrics(chords: Sequence[chord.Chord]) -> Tuple[int, Dict[str, float]]:
    counts = {quality: 0 for quality in QUALITY_TARGETS}
    other = 0
    for ch in chords:
        quality = ch.quality or ""
        if quality in counts:
            counts[quality] += 1
        else:
            other += 1
    total = sum(counts.values()) + other
    if total == 0:
        return 0, {key: 0.0 for key in (*QUALITY_TARGETS, "other")}
    percentages = {f"{key}_pct": (counts[key] / total) * 100.0 for key in QUALITY_TARGETS}
    percentages["other_pct"] = (other / total) * 100.0
    return total, percentages


def harmonic_density(chords: Sequence[chord.Chord]) -> Optional[float]:
    if not chords:
        return None
    densities = [len(set(ch.pitchClasses)) for ch in chords if ch.pitches]
    if not densities:
        return None
    return float(sum(densities) / len(densities))


def dissonance_ratio(chords: Sequence[chord.Chord]) -> Optional[float]:
    if not chords:
        return None
    dissonant = 0
    evaluated = 0
    for ch in chords:
        try:
            consonant = ch.isConsonant()
        except Exception:
            continue
        evaluated += 1
        if not consonant:
            dissonant += 1
    if evaluated == 0:
        return None
    return float(dissonant / evaluated)


def build_chord_index(chords: Sequence[chord.Chord]) -> List[Tuple[float, float, chord.Chord]]:
    index: List[Tuple[float, float, chord.Chord]] = []
    for ch in chords:
        start = float(ch.offset or 0.0)
        ql = float(ch.duration.quarterLength or 0.0)
        end = start + max(ql, CHORD_MATCH_TOLERANCE)
        index.append((start, end, ch))
    return index


def find_chord_for_offset(index: Sequence[Tuple[float, float, chord.Chord]], offset: float) -> Optional[chord.Chord]:
    for start, end, ch in index:
        if start - CHORD_MATCH_TOLERANCE <= offset < end + CHORD_MATCH_TOLERANCE:
            return ch
    return None


def remove_pitch_from_chord(src_chord: chord.Chord, target_pitch_class: Optional[int]) -> Optional[chord.Chord]:
    """Strip a single occurrence of the melodic pitch class to inspect the support."""
    if target_pitch_class is None:
        return None
    remaining = []
    removed = False
    for chord_pitch in src_chord.pitches:
        if not removed and chord_pitch.pitchClass == target_pitch_class:
            removed = True
            continue
        remaining.append(chord_pitch)
    if not remaining:
        return None
    stripped = chord.Chord(remaining)
    stripped.duration = src_chord.duration
    return stripped


def classify_dissonant_note(
    target_note: note.Note,
    prev_note: Optional[note.Note],
    next_note: Optional[note.Note],
    current_chord: Optional[chord.Chord],
    next_chord: Optional[chord.Chord],
) -> Optional[str]:
    if current_chord is None:
        return None
    if len(current_chord.pitches) < 2:
        return None
    target_pitch = getattr(target_note, "pitch", None)
    target_pitch_class = getattr(target_pitch, "pitchClass", None)
    if target_pitch is None or target_pitch_class is None:
        return None

    accompaniment = remove_pitch_from_chord(current_chord, target_pitch_class)
    if accompaniment is None:
        return None

    try:
        bass_pitch = min(accompaniment.pitches, key=lambda p: p.midi)
    except Exception:
        return None
    if any(p.pitchClass == target_pitch_class for p in accompaniment.pitches):
        return None

    try:
        melodic_interval = interval.Interval(bass_pitch, target_note)
    except Exception:
        return None
    if melodic_interval.isConsonant():
        return None

    dissonance_type = "other_dissonance"

    prev_interval = interval.Interval(prev_note, target_note) if prev_note else None
    next_interval = interval.Interval(target_note, next_note) if next_note else None

    next_support = next_chord or current_chord
    next_is_chord_tone = False
    if next_note and next_support:
        next_pitch = getattr(next_note, "pitch", None)
        next_pitch_class = getattr(next_pitch, "pitchClass", None)
        support_without_note = remove_pitch_from_chord(next_support, next_pitch_class)
        if support_without_note and support_without_note.pitches:
            try:
                next_bass = min(support_without_note.pitches, key=lambda p: p.midi)
                resolution_interval = interval.Interval(next_bass, next_note)
                next_is_chord_tone = resolution_interval.isConsonant()
            except Exception:
                next_is_chord_tone = False

    if prev_interval is not None and next_interval is not None:
        prev_step = abs(prev_interval.semitones) <= 2
        next_step = abs(next_interval.semitones) <= 2
        prev_dir = getattr(prev_interval, "direction", 0)
        next_dir = getattr(next_interval, "direction", 0)
        same_direction = prev_dir == next_dir != 0
        if prev_step and next_step and same_direction and next_is_chord_tone:
            return "passing_tone"

    if prev_interval is not None and next_interval is not None:
        prev_leap = abs(prev_interval.semitones) >= 3
        next_step = abs(next_interval.semitones) <= 2
        strong_beat = getattr(target_note, "beatStrength", 0.0) >= 0.5
        if strong_beat and prev_leap and next_step and next_is_chord_tone:
            return "appoggiatura"

    return dissonance_type


def dissonance_profile(score: stream.Score, chord_index: Sequence[Tuple[float, float, chord.Chord]]) -> Tuple[int, Dict[str, Optional[float]]]:
    if not chord_index:
        return 0, {
            "passing_tone_ratio": None,
            "appoggiatura_ratio": None,
            "other_dissonance_ratio": None,
        }

    parts = list(score.parts)
    if not parts:
        return 0, {
            "passing_tone_ratio": None,
            "appoggiatura_ratio": None,
            "other_dissonance_ratio": None,
        }

    melodic_stream = parts[0].flatten()
    melodic_notes = [n for n in melodic_stream.notes if isinstance(n, note.Note)]
    if len(melodic_notes) < 3:
        return 0, {
            "passing_tone_ratio": None,
            "appoggiatura_ratio": None,
            "other_dissonance_ratio": None,
        }

    dissonant_counts = {"passing_tone": 0, "appoggiatura": 0, "other_dissonance": 0}
    dissonant_total = 0

    for idx, current in enumerate(melodic_notes):
        prev_note = melodic_notes[idx - 1] if idx > 0 else None
        next_note = melodic_notes[idx + 1] if idx + 1 < len(melodic_notes) else None
        if not prev_note or not next_note:
            continue
        if current.tie and current.tie.type in {"stop", "continue"}:
            continue
        offset = float(current.offset or 0.0)
        chord_now = find_chord_for_offset(chord_index, offset)
        if chord_now is None:
            continue
        next_offset = float(next_note.offset or 0.0)
        chord_next = find_chord_for_offset(chord_index, next_offset)
        classification = classify_dissonant_note(current, prev_note, next_note, chord_now, chord_next)
        if classification is None:
            continue
        dissonant_total += 1
        dissonant_counts[classification] += 1

    if dissonant_total == 0:
        return 0, {
            "passing_tone_ratio": None,
            "appoggiatura_ratio": None,
            "other_dissonance_ratio": None,
        }

    return dissonant_total, {
        "passing_tone_ratio": dissonant_counts["passing_tone"] / dissonant_total,
        "appoggiatura_ratio": dissonant_counts["appoggiatura"] / dissonant_total,
        "other_dissonance_ratio": dissonant_counts["other_dissonance"] / dissonant_total,
    }


def roman_analysis(chords: Sequence[chord.Chord], key_obj) -> Tuple[int, Optional[float], Optional[float]]:
    if not chords or key_obj is None:
        return 0, None, None

    romans: List[roman.RomanNumeral] = []
    for ch in chords:
        try:
            rn = roman.romanNumeralFromChord(ch, key_obj)
        except Exception:
            continue
        if romans and rn.figure == romans[-1].figure:
            continue
        romans.append(rn)

    if len(romans) < 2:
        mixture_ratio = None
        return len(romans), None, mixture_ratio

    transitions = 0
    deceptive = 0
    for first, second in zip(romans, romans[1:]):
        transitions += 1
        try:
            if first.scaleDegree == 5 and second.scaleDegree == 6:
                deceptive += 1
        except Exception:
            continue

    deceptive_ratio = deceptive / transitions if transitions else None

    mixture_hits = 0
    evaluated = 0
    for rn in romans:
        func = getattr(rn, "isMixture", None)
        if callable(func):
            evaluated += 1
            if func():
                mixture_hits += 1
    mixture_ratio = (mixture_hits / evaluated) if evaluated else None

    return len(romans), deceptive_ratio, mixture_ratio


def compute_harmonic_features(score: stream.Score) -> Dict[str, object]:
    chordified = chordify_score(score)
    chords = extract_chords(chordified)

    quality_total, quality_percentages = chord_quality_metrics(chords)
    density = harmonic_density(chords)
    diss_ratio = dissonance_ratio(chords)

    chord_index = build_chord_index(chords)
    dissonant_total, dissonance_ratios = dissonance_profile(score, chord_index)

    key_obj = None
    try:
        key_obj = score.analyze("key")
    except Exception:
        key_obj = None

    roman_count, deceptive_ratio, mixture_ratio = roman_analysis(chords, key_obj)

    result: Dict[str, object] = {
        "chord_event_count": len(chords),
        "chord_quality_total": quality_total,
        "harmonic_density_mean": density,
        "dissonance_ratio": diss_ratio,
        "dissonant_note_count": dissonant_total,
        "roman_chord_count": roman_count,
        "deceptive_cadence_ratio": deceptive_ratio,
        "modal_interchange_ratio": mixture_ratio,
    }
    for key, value in quality_percentages.items():
        result[f"chord_quality_{key}"] = value
    for key, value in dissonance_ratios.items():
        result[key] = value
    return result


def run_feature_extraction(
    df: pd.DataFrame,
    limit: Optional[int] = None,
    skip_errors: bool = True,
) -> List[HarmonicFeatureResult]:
    results: List[HarmonicFeatureResult] = []
    for idx, (composer, title, path) in enumerate(iter_score_records(df)):
        if limit is not None and idx >= limit:
            break
        try:
            score = parse_score(path)
            metrics = cast(Dict[str, Any], compute_harmonic_features(score))
        except Exception as exc:
            message = f"[warn] Failed to extract harmonic features from {path}: {exc}"
            if skip_errors:
                print(message, file=sys.stderr)
                continue
            raise
        results.append(
            HarmonicFeatureResult(
                composer_label=composer,
                title=title,
                mxl_path=path,
                chord_event_count=_get_int(metrics, "chord_event_count"),
                chord_quality_total=_get_int(metrics, "chord_quality_total"),
                chord_quality_major_pct=_get_float(metrics, "chord_quality_major_pct"),
                chord_quality_minor_pct=_get_float(metrics, "chord_quality_minor_pct"),
                chord_quality_diminished_pct=_get_float(metrics, "chord_quality_diminished_pct"),
                chord_quality_augmented_pct=_get_float(metrics, "chord_quality_augmented_pct"),
                chord_quality_other_pct=_get_float(metrics, "chord_quality_other_pct"),
                harmonic_density_mean=_get_optional_float(metrics, "harmonic_density_mean"),
                dissonance_ratio=_get_optional_float(metrics, "dissonance_ratio"),
                dissonant_note_count=_get_int(metrics, "dissonant_note_count"),
                passing_tone_ratio=_get_optional_float(metrics, "passing_tone_ratio"),
                appoggiatura_ratio=_get_optional_float(metrics, "appoggiatura_ratio"),
                other_dissonance_ratio=_get_optional_float(metrics, "other_dissonance_ratio"),
                roman_chord_count=_get_int(metrics, "roman_chord_count"),
                deceptive_cadence_ratio=_get_optional_float(metrics, "deceptive_cadence_ratio"),
                modal_interchange_ratio=_get_optional_float(metrics, "modal_interchange_ratio"),
            )
        )
    return results


def write_features(results: Sequence[HarmonicFeatureResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([result.to_dict() for result in results])
    df.to_csv(output_csv, index=False)


def load_features(features_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    return pd.read_csv(features_path)


def _feature_columns() -> List[str]:
    return [
        "chord_quality_major_pct",
        "chord_quality_minor_pct",
        "chord_quality_diminished_pct",
        "chord_quality_augmented_pct",
        "chord_quality_other_pct",
        "harmonic_density_mean",
        "dissonance_ratio",
        "passing_tone_ratio",
        "appoggiatura_ratio",
        "other_dissonance_ratio",
        "deceptive_cadence_ratio",
        "modal_interchange_ratio",
    ]


def plot_features(df: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    features = [col for col in _feature_columns() if col in df.columns]
    for column in features:
        filtered = df[["composer_label", column]].dropna()
        if filtered.empty:
            continue
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=filtered, x="composer_label", y=column)
        plt.title(column.replace("_", " ").title())
        plt.xlabel("Composer")
        plt.ylabel(column)
        plt.tight_layout()
        output_path = figure_dir / f"boxplot_{column}.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    group = df.groupby("composer_label")
    stats = group[_feature_columns()].mean().round(4)
    counts = group.size().rename("piece_count")
    return stats.join(counts)


def _get_int(metrics: Dict[str, Any], key: str, default: int = 0) -> int:
    value = metrics.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_float(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_optional_float(metrics: Dict[str, Any], key: str) -> Optional[float]:
    if key not in metrics or metrics[key] is None:
        return None
    try:
        return float(metrics[key])
    except (TypeError, ValueError):
        return None


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract harmonic features from the curated corpus.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CORPUS, help="Path to curated corpus CSV.")
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS, help="Fallback path list if CSV absent.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT, help="Destination for harmonic features CSV.")
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR, help="Directory for saving plots.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of files processed (for dry runs).")
    parser.add_argument("--no-skip-errors", action="store_true", help="Abort on the first extraction error instead of skipping.")
    parser.add_argument("--skip-plots", action="store_true", help="Do not generate plots after extraction.")
    parser.add_argument("--features-from", type=Path, default=None, help="Load an existing features CSV instead of recomputing.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)

    if args.features_from is not None:
        features_df = load_features(args.features_from)
        print(descriptive_stats(features_df))
        if not args.skip_plots:
            plot_features(features_df, args.figure_dir)
        return 0

    df = load_corpus(args.csv, args.paths)
    results = run_feature_extraction(df, limit=args.limit, skip_errors=not args.no_skip_errors)
    if not results:
        raise RuntimeError("No harmonic features could be extracted.")

    write_features(results, args.output_csv)
    features_df = pd.DataFrame([result.to_dict() for result in results])
    stats = descriptive_stats(features_df)
    print(stats)

    if not args.skip_plots:
        plot_features(features_df, args.figure_dir)
        print(f"Saved plots to {args.figure_dir}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
