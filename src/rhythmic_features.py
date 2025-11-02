"""Extract rhythmic features from the curated solo-piano corpus."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from music21 import chord, meter, note, stream

# Use a non-interactive backend to support headless environments
matplotlib.use("Agg")

from score_parser import parse_score  # type: ignore

DEFAULT_CORPUS = Path("data/curated/solo_piano_corpus.csv")
DEFAULT_PATHS = Path("data/curated/solo_piano_mxl_paths.txt")
DEFAULT_OUTPUT = Path("data/features/rhythmic_features.csv")
DEFAULT_FIGURE_DIR = Path("figures/rhythmic")

STRONG_BEAT_THRESHOLD = 0.75
SYNC_WEAK_THRESHOLD = 0.5
FAST_NOTE_THRESHOLD = 0.25  # quarter-length value (<= sixteenth note)
MICRO_WINDOW = 4


@dataclass
class RhythmicFeatureResult:
    composer_label: str
    title: Optional[str]
    mxl_path: Path
    note_event_count: int
    avg_note_duration: Optional[float]
    std_note_duration: Optional[float]
    notes_per_beat: Optional[float]
    downbeat_emphasis_ratio: Optional[float]
    syncopation_ratio: Optional[float]
    rhythmic_pattern_entropy: Optional[float]
    micro_rhythmic_density: Optional[float]
    cross_rhythm_ratio: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "composer_label": self.composer_label,
            "title": self.title,
            "mxl_path": str(self.mxl_path),
            "note_event_count": self.note_event_count,
            "avg_note_duration": self.avg_note_duration,
            "std_note_duration": self.std_note_duration,
            "notes_per_beat": self.notes_per_beat,
            "downbeat_emphasis_ratio": self.downbeat_emphasis_ratio,
            "syncopation_ratio": self.syncopation_ratio,
            "rhythmic_pattern_entropy": self.rhythmic_pattern_entropy,
            "micro_rhythmic_density": self.micro_rhythmic_density,
            "cross_rhythm_ratio": self.cross_rhythm_ratio,
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


def _collect_rhythmic_events(score: stream.Score) -> List[Tuple[float, float, float]]:
    events: List[Tuple[float, float, float]] = []
    flat = score.flatten().notes
    for element in flat:
        if isinstance(element, note.Note):
            duration = float(element.duration.quarterLength or 0.0)
            if duration <= 0:
                continue
            events.append((float(element.offset or 0.0), duration, float(element.beatStrength or 0.0)))
        elif isinstance(element, chord.Chord):
            duration = float(element.duration.quarterLength or 0.0)
            if duration <= 0:
                continue
            events.append((float(element.offset or 0.0), duration, float(element.beatStrength or 0.0)))
    events.sort(key=lambda item: item[0])
    return events


def _primary_time_signature(score: stream.Score) -> Optional[Any]:
    ts = score.recurse().getElementsByClass("TimeSignature")
    for candidate in ts:
        return candidate
    return None


def _notes_per_beat(events: Sequence[Tuple[float, float, float]], beat_length: float, highest_time: float) -> Optional[float]:
    if not events or beat_length <= 0:
        return None
    total_beats = highest_time / beat_length if highest_time > 0 else 0.0
    if total_beats <= 0:
        return None
    return float(len(events) / total_beats)


def _downbeat_emphasis(events: Sequence[Tuple[float, float, float]]) -> Optional[float]:
    if not events:
        return None
    strong = sum(1 for _, _, strength in events if strength >= STRONG_BEAT_THRESHOLD)
    return float(strong / len(events))


def _is_syncopated(offset: float, duration: float, beat_length: float, beat_strength: float) -> bool:
    if beat_length <= 0:
        return False
    if beat_strength >= SYNC_WEAK_THRESHOLD:
        return False
    beat_position = offset % beat_length
    remaining = beat_length - beat_position
    return duration > remaining + 1e-6


def _syncopation_ratio(events: Sequence[Tuple[float, float, float]], beat_length: float) -> Optional[float]:
    if not events or beat_length <= 0:
        return None
    syncopated = sum(1 for offset, duration, strength in events if _is_syncopated(offset, duration, beat_length, strength))
    return float(syncopated / len(events))


def _rhythmic_pattern_entropy(events: Sequence[Tuple[float, float, float]], n: int = 3) -> Optional[float]:
    if len(events) < n:
        return None
    tokens = [f"{duration:.3f}" for _, duration, _ in events]
    total_windows = len(tokens) - n + 1
    counter: Counter[str] = Counter()
    for idx in range(total_windows):
        window = "|".join(tokens[idx : idx + n])
        counter[window] += 1
    if not counter:
        return None
    entropy = 0.0
    for count in counter.values():
        p = count / total_windows
        entropy -= p * np.log2(p)
    return float(entropy)


def _micro_rhythmic_density(events: Sequence[Tuple[float, float, float]], window: int = MICRO_WINDOW) -> Optional[float]:
    if len(events) < window:
        return None
    hits = 0
    total = len(events) - window + 1
    for idx in range(total):
        window_durations = [duration for _, duration, _ in events[idx : idx + window]]
        short_notes = sum(1 for duration in window_durations if duration <= FAST_NOTE_THRESHOLD)
        if short_notes >= window - 1:
            hits += 1
    if total == 0:
        return None
    return float(hits / total)


def _measure_denominators(measure: stream.Measure) -> List[int]:
    denominators: List[int] = []
    for element in measure.notesAndRests:
        if isinstance(element, note.Rest):
            continue
        duration = float(element.duration.quarterLength or 0.0)
        if duration <= 0:
            continue
        frac = Fraction(duration).limit_denominator(64)
        denominators.append(frac.denominator)
    return denominators


def _cross_rhythm_ratio(score: stream.Score) -> Optional[float]:
    parts = list(score.parts)
    if len(parts) < 2:
        return None
    upper_measures = list(parts[0].getElementsByClass(stream.Measure))
    lower_measures = list(parts[-1].getElementsByClass(stream.Measure))
    total_measures = min(len(upper_measures), len(lower_measures))
    if total_measures == 0:
        return None
    cross = 0
    evaluated = 0
    for idx in range(total_measures):
        upper = upper_measures[idx]
        lower = lower_measures[idx]
        if upper is None or lower is None:
            continue
        upper_den = _measure_denominators(upper)
        lower_den = _measure_denominators(lower)
        if not upper_den or not lower_den:
            continue
        upper_set = {min(den, 64) for den in upper_den}
        lower_set = {min(den, 64) for den in lower_den}
        evaluated += 1
        if upper_set != lower_set:
            cross += 1
    if evaluated == 0:
        return None
    return float(cross / evaluated)


def compute_rhythmic_features(score: stream.Score) -> Dict[str, object]:
    events = _collect_rhythmic_events(score)
    note_event_count = len(events)

    if note_event_count == 0:
        return {
            "note_event_count": 0,
            "avg_note_duration": None,
            "std_note_duration": None,
            "notes_per_beat": None,
            "downbeat_emphasis_ratio": None,
            "syncopation_ratio": None,
            "rhythmic_pattern_entropy": None,
            "micro_rhythmic_density": None,
            "cross_rhythm_ratio": None,
        }

    durations = [duration for _, duration, _ in events]
    avg_duration = float(np.mean(durations))
    std_duration = float(np.std(durations, ddof=0)) if len(durations) > 1 else 0.0

    time_signature = _primary_time_signature(score)
    if time_signature and hasattr(time_signature, "beatDuration"):
        beat_length = float(time_signature.beatDuration.quarterLength)
    else:
        beat_length = 1.0

    notes_per_beat = _notes_per_beat(events, beat_length, float(score.highestTime or 0.0))
    downbeat_ratio = _downbeat_emphasis(events)
    syncopation = _syncopation_ratio(events, beat_length)
    pattern_entropy = _rhythmic_pattern_entropy(events)
    micro_density = _micro_rhythmic_density(events)
    cross_ratio = _cross_rhythm_ratio(score)

    return {
        "note_event_count": note_event_count,
        "avg_note_duration": avg_duration,
        "std_note_duration": std_duration,
        "notes_per_beat": notes_per_beat,
        "downbeat_emphasis_ratio": downbeat_ratio,
        "syncopation_ratio": syncopation,
        "rhythmic_pattern_entropy": pattern_entropy,
        "micro_rhythmic_density": micro_density,
        "cross_rhythm_ratio": cross_ratio,
    }


def run_feature_extraction(
    df: pd.DataFrame,
    limit: Optional[int] = None,
    skip_errors: bool = True,
) -> List[RhythmicFeatureResult]:
    results: List[RhythmicFeatureResult] = []
    for idx, (composer, title, path) in enumerate(iter_score_records(df)):
        if limit is not None and idx >= limit:
            break
        try:
            score = parse_score(path)
            metrics = cast(Dict[str, Any], compute_rhythmic_features(score))
        except Exception as exc:
            message = f"[warn] Failed to extract rhythmic features from {path}: {exc}"
            if skip_errors:
                print(message, file=sys.stderr)
                continue
            raise
        results.append(
            RhythmicFeatureResult(
                composer_label=composer,
                title=title,
                mxl_path=path,
                note_event_count=_get_int(metrics, "note_event_count"),
                avg_note_duration=_get_optional_float(metrics, "avg_note_duration"),
                std_note_duration=_get_optional_float(metrics, "std_note_duration"),
                notes_per_beat=_get_optional_float(metrics, "notes_per_beat"),
                downbeat_emphasis_ratio=_get_optional_float(metrics, "downbeat_emphasis_ratio"),
                syncopation_ratio=_get_optional_float(metrics, "syncopation_ratio"),
                rhythmic_pattern_entropy=_get_optional_float(metrics, "rhythmic_pattern_entropy"),
                micro_rhythmic_density=_get_optional_float(metrics, "micro_rhythmic_density"),
                cross_rhythm_ratio=_get_optional_float(metrics, "cross_rhythm_ratio"),
            )
        )
    return results


def write_features(results: Sequence[RhythmicFeatureResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([result.to_dict() for result in results])
    df.to_csv(output_csv, index=False)


def load_features(features_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    return pd.read_csv(features_path)


def _feature_columns() -> List[str]:
    return [
        "avg_note_duration",
        "std_note_duration",
        "notes_per_beat",
        "downbeat_emphasis_ratio",
        "syncopation_ratio",
        "rhythmic_pattern_entropy",
        "micro_rhythmic_density",
        "cross_rhythm_ratio",
    ]


def plot_features(df: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for column in _feature_columns():
        if column not in df.columns:
            continue
        filtered = df[["composer_label", column]].dropna()
        if filtered.empty:
            continue
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=filtered, x="composer_label", y=column)
        plt.title(column.replace("_", " ").title())
        plt.xlabel("Composer")
        plt.ylabel(column)
        plt.tight_layout()
        plt.savefig(figure_dir / f"boxplot_{column}.png", dpi=200, bbox_inches="tight")
        plt.close()


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    group = df.groupby("composer_label")
    stats = group[["note_event_count"] + _feature_columns()].mean().round(4)
    counts = group.size().rename("piece_count")
    return stats.join(counts)


def _get_int(metrics: Dict[str, Any], key: str, default: int = 0) -> int:
    value = metrics.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_optional_float(metrics: Dict[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract rhythmic features from the curated corpus.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CORPUS, help="Path to curated corpus CSV.")
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS, help="Fallback path list if CSV absent.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT, help="Destination for rhythmic features CSV.")
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
        raise RuntimeError("No rhythmic features could be extracted.")

    write_features(results, args.output_csv)
    features_df = pd.DataFrame([result.to_dict() for result in results])
    stats = descriptive_stats(features_df)
    print(stats)

    if not args.skip_plots:
        plot_features(features_df, args.figure_dir)
        print(f"Saved plots to {args.figure_dir}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
