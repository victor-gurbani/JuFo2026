#!/usr/bin/env python3
"""Interactive helper to annotate scores from the curated corpus.

This script lets you search the curated solo-piano corpus, pick a piece,
run the existing `annotate_musicxml.annotate_score` pipeline, and deposit the
annotated MusicXML into your `~/Documents` folder (inside an `AnnotatedScores`
subdirectory so things stay tidy).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    from tqdm import tqdm  # type: ignore[import]
except ImportError:  # pragma: no cover - fallback when tqdm missing
    tqdm = None  # type: ignore

# Ensure we can import the annotation utilities that live under src/
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from annotate_musicxml import annotate_score  # type: ignore  # pylint: disable=wrong-import-position


@dataclass
class CorpusEntry:
    composer: str
    title: str
    mxl_path: Path
    composer_norm: str
    title_norm: str
    rating: float | None
    instrument: str

    @classmethod
    def from_row(cls, row: dict, csv_dir: Path) -> "CorpusEntry":
        composer = str(row.get("composer_name") or row.get("composer_label") or "?").strip()
        title = str(row.get("title") or row.get("song_name") or "Untitled").strip()
        mxl_path = resolve_musicxml_path(row, csv_dir)
        return cls(
            composer=composer,
            title=title,
            mxl_path=mxl_path,
            composer_norm=normalise_text(composer),
            title_norm=normalise_text(title),
            rating=parse_rating(row.get("rating")),
            instrument=parse_instrument(row),
        )


def parse_rating(raw_value: object) -> float | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_instrument(row: dict) -> str:
    candidate_keys: Sequence[str] = (
        "instrument_names",
        "instrument",
        "instruments",
        "instrumentation_tags",
    )
    for key in candidate_keys:
        value = row.get(key)
        if not value:
            continue
        text = str(value).strip()
        if text:
            return text
    return "Unknown"


def resolve_musicxml_path(row: dict, csv_dir: Path) -> Path:
    """Resolve a MusicXML path from a corpus row, tolerating relative forms."""

    candidate_keys: Sequence[str] = (
        "mxl_abs_path",
        "mxl_path",
        "mxl",
        "musicxml_path",
        "mxl_rel_path",
    )

    attempted: list[str] = []
    for key in candidate_keys:
        raw_value = row.get(key)
        candidate = coerce_musicxml_path(raw_value, csv_dir)
        if candidate is None:
            if raw_value:
                attempted.append(f"{key}={raw_value}")
            continue
        if candidate.exists():
            return candidate
        attempted.append(f"{key}={candidate}")

    if attempted:
        raise ValueError(
            "Unable to locate MusicXML path. Tried: " + "; ".join(attempted)
        )
    raise ValueError("Missing MusicXML path in corpus row")


def coerce_musicxml_path(raw_value: object, csv_dir: Path) -> Path | None:
    """Convert a raw CSV field into an absolute MusicXML path."""

    if not raw_value:
        return None

    raw_text = str(raw_value).strip()
    if not raw_text:
        return None
    if not raw_text.lower().endswith(".mxl"):
        return None

    raw_path = Path(raw_text)

    # Immediate absolute or user-relative paths.
    if raw_path.is_absolute():
        return raw_path.resolve(strict=False)
    if raw_text.startswith("~"):
        expanded = Path(raw_text).expanduser()
        if expanded.is_absolute():
            return expanded.resolve(strict=False)

    candidate_bases = [csv_dir, REPO_ROOT]
    dataset_root = REPO_ROOT / "15571083"
    if dataset_root not in candidate_bases:
        candidate_bases.append(dataset_root)

    for base in candidate_bases:
        candidate = (base / raw_path).resolve(strict=False)
        if candidate.exists():
            return candidate

    # If nothing existed, still return the best guess so callers can report it.
    return (candidate_bases[0] / raw_path).resolve(strict=False)


def load_corpus(csv_path: Path) -> List[CorpusEntry]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Corpus CSV not found: {csv_path}")

    entries: List[CorpusEntry] = []
    missing_paths = 0
    csv_dir = csv_path.parent

    total_rows: int | None = None
    if tqdm is not None:
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                total_rows = sum(1 for _ in handle) - 1  # subtract header row
                if total_rows < 0:
                    total_rows = 0
        except Exception:  # pragma: no cover - best-effort only
            total_rows = None

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        iterator = reader
        progress_iter = iterator
        if tqdm is not None:
            progress_iter = tqdm(
                iterator,
                desc="Loading corpus",
                unit="row",
                total=total_rows,
                dynamic_ncols=True,
                smoothing=0.1,
            )
        for row in progress_iter:
            try:
                entry = CorpusEntry.from_row(row, csv_dir)
            except Exception as exc:  # pragma: no cover - defensive logging only
                print(f"[warn] Skipping row due to error: {exc}")
                continue
            if entry.mxl_path.exists():
                entries.append(entry)
            else:  # pragma: no cover - only hit when dataset is incomplete
                missing_paths += 1
    if tqdm is not None and hasattr(progress_iter, "close"):
        try:
            progress_iter.close()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - best-effort cleanup
            pass

    if missing_paths:
        print(f"[warn] {missing_paths} entries referenced missing MusicXML files and were skipped.")
    if not entries:
        raise RuntimeError("No valid entries were loaded from the corpus CSV.")
    return entries


def normalise_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return " ".join(text.lower().split())


def normalise_query(query: str) -> str:
    pieces = query.strip().split()
    if not pieces:
        return ""
    normalized = [normalise_text(piece) for piece in pieces if piece.strip()]
    return ".*".join(re.escape(part) for part in normalized if part)


def query_matches(entry: CorpusEntry, query: str) -> bool:
    pattern = normalise_query(query)
    if not pattern:
        return False
    regex = re.compile(pattern)
    return bool(regex.search(entry.title_norm) or regex.search(entry.composer_norm))


def find_matches(entries: Iterable[CorpusEntry], query: str) -> List[CorpusEntry]:
    matches = [entry for entry in entries if query_matches(entry, query)]
    return sorted(
        matches,
        key=lambda entry: entry.rating if entry.rating is not None else float("-inf"),
        reverse=True,
    )


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return slug or "annotated_score"


def choose_entry(matches: List[CorpusEntry]) -> CorpusEntry | None:
    if not matches:
        print("No matches found. Try a different search.")
        return None

    print("\nMatches:")
    for idx, entry in enumerate(matches, start=1):
        rating_text = f"{entry.rating:.2f}" if entry.rating is not None else "N/A"
        instrument_text = entry.instrument or "Unknown"
        print(
            f"  {idx:2d}. {entry.composer} — {entry.title} "
            f"(rating {rating_text}, instrument {instrument_text})"
        )

    while True:
        selection = input("Select a number to annotate (blank to cancel): ").strip()
        if selection == "":
            return None
        if not selection.isdigit():
            print("Please enter a number from the list or press Enter to cancel.")
            continue
        index = int(selection)
        if 1 <= index <= len(matches):
            return matches[index - 1]
        print("Selection out of range. Try again.")


def annotate_entry(entry: CorpusEntry, documents_dir: Path) -> None:
    target_dir = documents_dir / "AnnotatedScores"
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = slugify(f"{entry.composer}_{entry.title}")[:120]
    output_path = target_dir / f"{safe_name}.musicxml"

    print(f"\nAnnotating '{entry.title}' by {entry.composer}…")
    annotate_score(entry.mxl_path, output_path, hide_dissonant_label=True)
    print(f"Done. Annotated score saved to {output_path}")


def interactive_loop(entries: List[CorpusEntry]) -> None:
    documents_dir = Path.home() / "Documents"
    if not documents_dir.exists():
        print(f"[warn] {documents_dir} does not exist. Creating it now.")
        documents_dir.mkdir(parents=True, exist_ok=True)

    print("Interactive corpus annotator")
    print("Type part of a title or composer, or 'quit' to exit.\n")

    while True:
        query = input("Search query: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            return
        matches = find_matches(entries, query)
        chosen = choose_entry(matches)
        if chosen is None:
            continue
        try:
            annotate_entry(chosen, documents_dir)
        except Exception as exc:
            print(f"[error] Annotation failed: {exc}")
        print("")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search and annotate scores from solo piano corpora.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Search the entire PDMX catalogue instead of the curated solo-piano subset.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional explicit path to a corpus CSV (overrides curated/full defaults).",
    )
    return parser.parse_args()


def default_csv(full_catalogue: bool) -> Path:
    if full_catalogue:
        return REPO_ROOT / "15571083" / "PDMX.csv"
    return REPO_ROOT / "data" / "curated" / "solo_piano_corpus.csv"


def main() -> int:
    args = parse_arguments()
    corpus_csv = args.csv or default_csv(args.full)
    try:
        entries = load_corpus(corpus_csv)
    except Exception as exc:
        print(f"Failed to load corpus: {exc}")
        return 1

    try:
        interactive_loop(entries)
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
