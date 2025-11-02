"""Extract melodic features from the curated solo-piano corpus."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from music21 import chord, note, stream

OFFSET_TOLERANCE = 1e-3

# Use a non-interactive backend to support headless environments
matplotlib.use("Agg")

from score_parser import parse_score  # type: ignore

DEFAULT_CORPUS = Path("data/curated/solo_piano_corpus.csv")
DEFAULT_PATHS = Path("data/curated/solo_piano_mxl_paths.txt")
DEFAULT_OUTPUT = Path("data/features/melodic_features.csv")
DEFAULT_FIGURE_DIR = Path("figures/melodic")


@dataclass
class MelodicFeatureResult:
    composer_label: str
    title: Optional[str]
    mxl_path: Path
    note_count: int
    pitch_range_semitones: Optional[float]
    avg_melodic_interval: Optional[float]
    conjunct_motion_ratio: Optional[float]
    pitch_class_entropy: Optional[float]
    melodic_interval_std: Optional[float]
    melodic_leap_ratio: Optional[float]
    voice_independence_index: Optional[float]
    contrary_motion_ratio: Optional[float]
    parallel_motion_ratio: Optional[float]
    oblique_motion_ratio: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "composer_label": self.composer_label,
            "title": self.title,
            "mxl_path": str(self.mxl_path),
            "note_count": self.note_count,
            "pitch_range_semitones": self.pitch_range_semitones,
            "avg_melodic_interval": self.avg_melodic_interval,
            "conjunct_motion_ratio": self.conjunct_motion_ratio,
            "pitch_class_entropy": self.pitch_class_entropy,
            "melodic_interval_std": self.melodic_interval_std,
            "melodic_leap_ratio": self.melodic_leap_ratio,
            "voice_independence_index": self.voice_independence_index,
            "contrary_motion_ratio": self.contrary_motion_ratio,
            "parallel_motion_ratio": self.parallel_motion_ratio,
            "oblique_motion_ratio": self.oblique_motion_ratio,
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


def _collect_notes(stream_obj: stream.Stream) -> List[note.Note]:
    collected: List[note.Note] = []
    for element in stream_obj.flatten().notes:
        if isinstance(element, note.Note):
            collected.append(element)
        elif isinstance(element, chord.Chord):
            # Take each pitch in the chord as a separate note event for pitch statistics.
            for ch_pitch in element.pitches:
                proxy = note.Note(ch_pitch)
                proxy.offset = element.offset
                proxy.duration = element.duration
                collected.append(proxy)
    return collected


def _pitch_range_semitones(notes: Sequence[note.Note]) -> Optional[float]:
    if len(notes) < 2:
        return None
    try:
        midis = [n.pitch.midi for n in notes if n.pitch is not None]
    except Exception:
        return None
    if not midis:
        return None
    return float(max(midis) - min(midis))


def _melodic_intervals(notes: Sequence[note.Note]) -> Tuple[List[float], List[int]]:
    intervals: List[float] = []
    directions: List[int] = []
    previous: Optional[note.Note] = None
    for current in notes:
        if previous is None:
            previous = current
            continue
        if previous.pitch is None or current.pitch is None:
            previous = current
            continue
        diff = current.pitch.midi - previous.pitch.midi
        intervals.append(abs(diff))
        if diff > 0:
            directions.append(1)
        elif diff < 0:
            directions.append(-1)
        else:
            directions.append(0)
        previous = current
    return intervals, directions


def _avg_melodic_interval(intervals: Sequence[float]) -> Optional[float]:
    if not intervals:
        return None
    return float(sum(intervals) / len(intervals))


def _conjunct_motion_ratio(intervals: Sequence[float]) -> Optional[float]:
    if not intervals:
        return None
    steps = sum(1 for value in intervals if 0 < value <= 2)
    return float(steps / len(intervals))


def _pitch_class_entropy(notes: Sequence[note.Note]) -> Optional[float]:
    counts = [0] * 12
    total = 0
    for n in notes:
        if n.pitch is None:
            continue
        counts[n.pitch.pitchClass] += 1
        total += 1
    if total == 0:
        return None
    entropy = 0.0
    for count in counts:
        if count == 0:
            continue
        p = count / total
        entropy -= p * np.log2(p)
    return float(entropy)


def _melodic_interval_std(intervals: Sequence[float]) -> Optional[float]:
    if not intervals:
        return None
    return float(np.std(intervals, ddof=0))


def _melodic_leap_ratio(intervals: Sequence[float]) -> Optional[float]:
    if not intervals:
        return None
    leaps = sum(1 for value in intervals if value > 2)
    return float(leaps / len(intervals))


def _voice_note_events(part: stream.Part, mode: str) -> List[Tuple[float, float]]:
    events: List[Tuple[float, float]] = []
    for element in part.recurse().notes:
        offset = float(element.offset or 0.0)
        if isinstance(element, note.Note):
            if element.tie and element.tie.type in {"stop", "continue"}:
                continue
            if element.pitch is None:
                continue
            events.append((offset, float(element.pitch.midi)))
        elif isinstance(element, chord.Chord):
            if element.tie and element.tie.type in {"stop", "continue"}:
                continue
            if not element.pitches:
                continue
            chosen = max(element.pitches, key=lambda p: p.midi) if mode == "highest" else min(
                element.pitches, key=lambda p: p.midi
            )
            events.append((offset, float(chosen.midi)))
    events.sort(key=lambda item: item[0])
    return events


def _voice_motion_metrics(score: stream.Score) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    parts = list(score.parts)
    if len(parts) < 2:
        return (None, None, None, None)

    upper_events = _voice_note_events(parts[0], "highest")
    lower_events = _voice_note_events(parts[-1], "lowest")
    if len(upper_events) < 2 or len(lower_events) < 2:
        return (None, None, None, None)

    offsets = sorted({round(off / OFFSET_TOLERANCE) * OFFSET_TOLERANCE for off, _ in upper_events} | {
        round(off / OFFSET_TOLERANCE) * OFFSET_TOLERANCE for off, _ in lower_events
    })

    upper_idx = 0
    lower_idx = 0
    current_upper: Optional[float] = None
    current_lower: Optional[float] = None
    prev_upper: Optional[float] = None
    prev_lower: Optional[float] = None

    contrary = 0
    parallel = 0
    oblique = 0

    for offset in offsets:
        while upper_idx < len(upper_events) and abs(upper_events[upper_idx][0] - offset) <= OFFSET_TOLERANCE:
            current_upper = upper_events[upper_idx][1]
            upper_idx += 1
        while lower_idx < len(lower_events) and abs(lower_events[lower_idx][0] - offset) <= OFFSET_TOLERANCE:
            current_lower = lower_events[lower_idx][1]
            lower_idx += 1

        if current_upper is None or current_lower is None:
            continue
        if prev_upper is None or prev_lower is None:
            prev_upper = current_upper
            prev_lower = current_lower
            continue

        delta_upper = current_upper - prev_upper
        delta_lower = current_lower - prev_lower
        if abs(delta_upper) <= 1e-6 and abs(delta_lower) <= 1e-6:
            continue

        sign_upper = 1 if delta_upper > 0 else -1 if delta_upper < 0 else 0
        sign_lower = 1 if delta_lower > 0 else -1 if delta_lower < 0 else 0

        if sign_upper == 0 or sign_lower == 0:
            oblique += 1
        elif sign_upper == sign_lower:
            parallel += 1
        else:
            contrary += 1

        prev_upper = current_upper
        prev_lower = current_lower

    total_events = contrary + parallel + oblique
    if total_events == 0:
        return (None, None, None, None)

    independence = float((contrary - parallel) / total_events)
    contrary_ratio = float(contrary / total_events) if total_events else None
    parallel_ratio = float(parallel / total_events) if total_events else None
    oblique_ratio = float(oblique / total_events) if total_events else None
    return independence, contrary_ratio, parallel_ratio, oblique_ratio


def compute_melodic_features(score: stream.Score) -> Dict[str, object]:
    all_notes = _collect_notes(score)
    note_count = len(all_notes)

    pitch_range = _pitch_range_semitones(all_notes)
    entropy = _pitch_class_entropy(all_notes)

    melodic_source = score
    parts = list(score.parts)
    if parts:
        melodic_source = parts[0]
    melodic_notes = _collect_notes(melodic_source)
    intervals, directions = _melodic_intervals(melodic_notes)

    avg_interval = _avg_melodic_interval(intervals)
    conjunct_ratio = _conjunct_motion_ratio(intervals)

    voice_index, contrary_ratio, parallel_ratio, oblique_ratio = _voice_motion_metrics(score)

    interval_std = _melodic_interval_std(intervals)
    leap_ratio = _melodic_leap_ratio(intervals)

    return {
        "note_count": note_count,
        "pitch_range_semitones": pitch_range,
        "avg_melodic_interval": avg_interval,
        "conjunct_motion_ratio": conjunct_ratio,
        "pitch_class_entropy": entropy,
        "melodic_interval_std": interval_std,
        "melodic_leap_ratio": leap_ratio,
        "voice_independence_index": voice_index,
        "contrary_motion_ratio": contrary_ratio,
        "parallel_motion_ratio": parallel_ratio,
        "oblique_motion_ratio": oblique_ratio,
    }


def run_feature_extraction(
    df: pd.DataFrame,
    limit: Optional[int] = None,
    skip_errors: bool = True,
) -> List[MelodicFeatureResult]:
    results: List[MelodicFeatureResult] = []
    for idx, (composer, title, path) in enumerate(iter_score_records(df)):
        if limit is not None and idx >= limit:
            break
        try:
            score = parse_score(path)
            metrics = cast(Dict[str, Any], compute_melodic_features(score))
        except Exception as exc:
            message = f"[warn] Failed to extract melodic features from {path}: {exc}"
            if skip_errors:
                print(message, file=sys.stderr)
                continue
            raise
        results.append(
            MelodicFeatureResult(
                composer_label=composer,
                title=title,
                mxl_path=path,
                note_count=_get_int(metrics, "note_count"),
                pitch_range_semitones=_get_optional_float(metrics, "pitch_range_semitones"),
                avg_melodic_interval=_get_optional_float(metrics, "avg_melodic_interval"),
                conjunct_motion_ratio=_get_optional_float(metrics, "conjunct_motion_ratio"),
                pitch_class_entropy=_get_optional_float(metrics, "pitch_class_entropy"),
                melodic_interval_std=_get_optional_float(metrics, "melodic_interval_std"),
                melodic_leap_ratio=_get_optional_float(metrics, "melodic_leap_ratio"),
                voice_independence_index=_get_optional_float(metrics, "voice_independence_index"),
                contrary_motion_ratio=_get_optional_float(metrics, "contrary_motion_ratio"),
                parallel_motion_ratio=_get_optional_float(metrics, "parallel_motion_ratio"),
                oblique_motion_ratio=_get_optional_float(metrics, "oblique_motion_ratio"),
            )
        )
    return results


def write_features(results: Sequence[MelodicFeatureResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([result.to_dict() for result in results])
    df.to_csv(output_csv, index=False)


def load_features(features_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    return pd.read_csv(features_path)


def _feature_columns() -> List[str]:
    return [
        "pitch_range_semitones",
        "avg_melodic_interval",
        "conjunct_motion_ratio",
        "pitch_class_entropy",
        "melodic_interval_std",
        "melodic_leap_ratio",
        "voice_independence_index",
        "contrary_motion_ratio",
        "parallel_motion_ratio",
        "oblique_motion_ratio",
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
        output_path = figure_dir / f"boxplot_{column}.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    group = df.groupby("composer_label")
    stats = group[["note_count"] + _feature_columns()].mean().round(4)
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
    parser = argparse.ArgumentParser(description="Extract melodic features from the curated corpus.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CORPUS, help="Path to curated corpus CSV.")
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS, help="Fallback path list if CSV absent.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT, help="Destination for melodic features CSV.")
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
        raise RuntimeError("No melodic features could be extracted.")

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
