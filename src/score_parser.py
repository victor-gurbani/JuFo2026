"""Parse curated MusicXML scores with music21 and record basic summaries."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from music21 import converter, stream


DEFAULT_CORPUS = Path("data/curated/solo_piano_corpus.csv")
DEFAULT_PATHS = Path("data/curated/solo_piano_mxl_paths.txt")


@dataclass
class ParseResult:
    composer_label: str
    title: Optional[str]
    mxl_path: Path
    measures: Optional[int]
    parts: int
    duration_quarters: Optional[float]

    def to_dict(self) -> dict:
        return {
            "composer_label": self.composer_label,
            "title": self.title,
            "mxl_path": str(self.mxl_path),
            "measures": self.measures,
            "parts": self.parts,
            "duration_quarters": self.duration_quarters,
        }


def read_mxl_paths(csv_path: Path, paths_path: Optional[Path] = None) -> pd.DataFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "mxl_abs_path" not in df.columns:
            raise ValueError(f"Missing 'mxl_abs_path' column in {csv_path}")
        return df
    if paths_path and paths_path.exists():
        paths = [line.strip() for line in paths_path.read_text().splitlines() if line.strip()]
        return pd.DataFrame({"mxl_abs_path": paths})
    raise FileNotFoundError("No curated corpus CSV or path list found.")


def iter_score_paths(df: pd.DataFrame) -> Iterable[tuple[str, Optional[str], Path]]:
    for _, row in df.iterrows():
        composer = str(row.get("composer_label") or "").strip()
        title = row.get("title") if isinstance(row.get("title"), str) else None
        path_value = row.get("mxl_abs_path") or row.get("mxl_path")
        if not isinstance(path_value, str):
            continue
        path = Path(path_value).expanduser().resolve()
        yield composer, title, path


def summarize_score(score: stream.Score) -> tuple[Optional[int], int, Optional[float]]:
    measures = None
    try:
        measures = len(list(score.parts[0].getElementsByClass(stream.Measure))) if score.parts else None
    except Exception:
        measures = None
    parts = len(score.parts) if hasattr(score, "parts") else 0
    duration = None
    try:
        quarter_length = score.duration.quarterLength
        duration = float(quarter_length) if quarter_length is not None else None
    except Exception:
        duration = None
    return measures, parts, duration


def parse_score(path: Path) -> stream.Score:
    parsed = converter.parse(path)
    if isinstance(parsed, stream.Score):
        return parsed
    if isinstance(parsed, stream.Part):
        score = stream.Score()
        score.insert(0, parsed)
        return score
    if isinstance(parsed, stream.Opus):
        # Join all contained scores into a single Score
        score = stream.Score()
        for sub in parsed:
            if isinstance(sub, stream.Score):
                for part in sub.parts:
                    score.insert(0, part)
            elif isinstance(sub, stream.Part):
                score.insert(0, sub)
        return score
    raise TypeError(f"Unsupported parsed object type for {path}: {type(parsed)!r}")


def parse_corpus(
    df: pd.DataFrame,
    limit: Optional[int] = None,
    skip_errors: bool = True,
) -> List[ParseResult]:
    results: List[ParseResult] = []
    for idx, (composer, title, mxl_path) in enumerate(iter_score_paths(df)):
        if limit is not None and idx >= limit:
            break
        try:
            score = parse_score(mxl_path)
            measures, parts, duration = summarize_score(score)
            results.append(ParseResult(composer, title, mxl_path, measures, parts, duration))
        except Exception as exc:
            if skip_errors:
                print(f"[warn] Failed to parse {mxl_path}: {exc}", file=sys.stderr)
                continue
            raise
    return results


def write_results(results: List[ParseResult], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.to_dict() for result in results]
    output_json.write_text(json.dumps(payload, indent=2))


def results_to_dataframe(results: List[ParseResult]) -> pd.DataFrame:
    return pd.DataFrame([result.to_dict() for result in results]) if results else pd.DataFrame()


def load_summaries(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Unexpected JSON format in {path}")
    return pd.DataFrame(data)


def print_stats(df: pd.DataFrame) -> None:
    if df.empty:
        print("No parsed scores available for statistics.")
        return
    required = ["composer_label", "measures", "parts", "duration_quarters"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing column '{column}' in summaries data")
    stats_df = df.copy()
    numeric_columns = ["measures", "parts", "duration_quarters"]
    for column in numeric_columns:
        stats_df[column] = pd.to_numeric(stats_df[column], errors="coerce")
    grouped = stats_df.groupby("composer_label")
    averages = grouped[numeric_columns].mean().round(2)
    counts = grouped.size()
    display = averages.rename(columns={"duration_quarters": "duration"})
    print("\nAverage measures/parts/duration by composer:")
    print(display)
    print("\nPiece counts:")
    print(counts)


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse curated MusicXML scores with music21.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CORPUS, help="Path to curated corpus CSV.")
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS, help="Fallback path list if CSV absent.")
    parser.add_argument("--output", type=Path, default=Path("data/parsed/summaries.json"), help="Path to write JSON summaries.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of files to parse.")
    parser.add_argument("--no-skip-errors", action="store_true", help="Abort on the first parse failure.")
    parser.add_argument("--stats-from", type=Path, default=None, help="Load an existing summaries JSON and print statistics without parsing.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_arguments(argv)
    if args.stats_from is not None:
        summaries = load_summaries(args.stats_from)
        print_stats(summaries)
        return 0
    df = read_mxl_paths(args.csv, args.paths)
    results = parse_corpus(df, limit=args.limit, skip_errors=not args.no_skip_errors)
    write_results(results, args.output)
    print(f"Parsed {len(results)} scores -> {args.output}")
    stats_df = results_to_dataframe(results)
    print_stats(stats_df)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
