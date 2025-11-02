"""Tools for curating a balanced solo-piano corpus from the PDMX dataset."""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


TARGET_COMPOSERS = ("bach", "mozart", "chopin", "debussy")


@dataclass(frozen=True)
class ComposerRule:
    label: str
    required: Sequence[str]
    any_groups: Sequence[Sequence[str]] = ()
    exclude: Sequence[str] = ()


COMPOSER_RULES: Dict[str, ComposerRule] = {
    "bach": ComposerRule(
        label="Bach",
        required=("bach",),
        any_groups=(("johann", "j s", "js", "sebastian", "joh"),),
        exclude=("christian", "wilhelm", "friedemann", "cpe", "emanuel", "carl", "anna", "wf"),
    ),
    "mozart": ComposerRule(
        label="Mozart",
        required=("mozart",),
        exclude=("leopold", "franz", "xaver", "anna", "holzer", "simrock", "suessmayr", "sussmayr"),
    ),
    "chopin": ComposerRule(
        label="Chopin",
        required=("chopin",),
        exclude=("bellini", "mussorgsky", "liszt", "guitar"),
    ),
    "debussy": ComposerRule(
        label="Debussy",
        required=("debussy",),
        any_groups=(("claude", "achille"),),
    ),
}

ARRANGEMENT_KEYWORDS = {
    "arr",
    "arranged",
    "arranger",
    "arrg",
    "transcription",
    "transcribed",
    "transcr",
    "transposed",
    "cover",
    "setting",
}

EXCLUDED_SYMBOLS = ("/", "&")


def normalize_text(value: str) -> str:
    """Return a lowercase ASCII string without punctuation for matching."""
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_arrangement(text: str) -> bool:
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]
    return any(tok in ARRANGEMENT_KEYWORDS for tok in tokens)


def resolve_composer(raw_name: Optional[str]) -> Optional[str]:
    if not isinstance(raw_name, str):
        return None
    if any(symbol in raw_name for symbol in EXCLUDED_SYMBOLS):
        return None
    if is_arrangement(raw_name):
        return None
    normalized = normalize_text(raw_name)
    for key, rule in COMPOSER_RULES.items():
        if not all(token in normalized for token in rule.required):
            continue
        if rule.any_groups and not all(any(option in normalized for option in group) for group in rule.any_groups):
            continue
        if any(token in normalized for token in rule.exclude):
            continue
        return rule.label
    return None


def load_score_metadata(base_path: Path, relative_path: str, cache: Dict[str, dict]) -> Optional[dict]:
    if not isinstance(relative_path, str):
        return None
    if relative_path in cache:
        return cache[relative_path]
    metadata_path = (base_path / relative_path.lstrip("./")).resolve()
    if not metadata_path.exists():
        return None
    cache[relative_path] = json.loads(metadata_path.read_text())
    return cache[relative_path]


def extract_instrument_names(score: dict) -> List[str]:
    instruments = score.get("instruments") or []
    names = []
    for entry in instruments:
        name = entry.get("name") if isinstance(entry, dict) else None
        if isinstance(name, str):
            names.append(name)
    return names


def extract_instrumentation_tags(score: dict) -> List[str]:
    instrumentations = score.get("instrumentations") or []
    names = []
    for entry in instrumentations:
        name = entry.get("name") if isinstance(entry, dict) else None
        if isinstance(name, str):
            names.append(name)
    return names


def is_solo_piano(score: dict) -> bool:
    instruments = [name.lower() for name in extract_instrument_names(score)]
    evidence = False
    if instruments:
        if not any("piano" in name for name in instruments):
            return False
        disallowed = ("choir", "voice", "ensemble", "orchestra", "vocal")
        if any(bad in inst for inst in instruments for bad in disallowed):
            return False
        unique = {name for name in instruments}
        if len(unique) > 1 and not all("piano" in name for name in unique):
            return False
        evidence = True
    tags = [tag.lower() for tag in extract_instrumentation_tags(score)]
    if tags:
        if not any("solo" in tag or "piano" in tag for tag in tags):
            return False
        evidence = True
    instrumentation = score.get("instrumentation_text")
    if isinstance(instrumentation, str) and instrumentation.strip():
        normalized = normalize_text(instrumentation)
        if "piano" in normalized and all(tok not in normalized for tok in ("violin", "voice", "organ", "string", "orchestra")):
            evidence = True
        elif any(tok in normalized for tok in ("violin", "voice", "organ", "string", "orchestra")):
            return False
    parts = score.get("parts")
    if isinstance(parts, int) and parts > 2:
        return False
    return evidence


def curate_corpus(
    csv_path: Path,
    dataset_root: Path,
    min_rating: float,
    require_no_license_conflict: bool,
    require_deduplicated: bool,
    require_best_unique: bool,
    max_per_composer: Optional[int],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    numeric_columns = ("rating", "n_ratings")
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    mask = df["rating"].fillna(0) >= min_rating
    if require_no_license_conflict and "subset:no_license_conflict" in df.columns:
        mask &= df["subset:no_license_conflict"] == True
    if require_deduplicated and "subset:rated_deduplicated" in df.columns:
        mask &= df["subset:rated_deduplicated"] == True
    if require_best_unique and "is_best_unique_arrangement" in df.columns:
        mask &= df["is_best_unique_arrangement"] == True

    filtered = df[mask].copy()

    metadata_cache: Dict[str, dict] = {}
    accepted_rows: List[dict] = []

    for _, row in filtered.iterrows():
        composer_label = resolve_composer(row.get("composer_name"))
        if composer_label is None:
            continue
        metadata_rel = row.get("metadata")
        if not isinstance(metadata_rel, str) or not metadata_rel.strip():
            continue
        metadata_entry = load_score_metadata(dataset_root, metadata_rel.strip(), metadata_cache)
        if not metadata_entry:
            continue
        score_data = metadata_entry.get("data", {}).get("score", {})
        if not is_solo_piano(score_data):
            continue
        mxl_path = row.get("mxl")
        if not isinstance(mxl_path, str):
            continue
        mxl_full = (dataset_root / mxl_path.lstrip("./")).resolve()
        if not mxl_full.exists():
            continue
        accepted_rows.append(
            {
                "composer_label": composer_label,
                "composer_name": row.get("composer_name"),
                "title": row.get("title") or row.get("song_name"),
                "rating": row.get("rating"),
                "n_ratings": row.get("n_ratings"),
                "mxl_rel_path": str(Path(dataset_root.name) / mxl_path.lstrip("./")),
                "mxl_abs_path": str(mxl_full),
                "metadata_rel_path": row.get("metadata"),
                "instrument_names": "; ".join(extract_instrument_names(score_data)),
                "instrumentation_tags": "; ".join(extract_instrumentation_tags(score_data)),
                "parts": score_data.get("parts"),
            }
        )

    if not accepted_rows:
        raise RuntimeError("No scores matched the requested filters.")

    curated = pd.DataFrame(accepted_rows).drop_duplicates("mxl_abs_path")
    curated.sort_values(by=["composer_label", "rating", "n_ratings"], ascending=[True, False, False], inplace=True)

    per_composer = max_per_composer
    if per_composer is None:
        per_composer = curated.groupby("composer_label").size().min()

    balanced = (
        curated.groupby("composer_label", group_keys=False)
        .apply(lambda group: group.head(per_composer))
        .reset_index(drop=True)
    )

    return balanced


def write_outputs(curated: pd.DataFrame, output_csv: Path, output_paths: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_paths.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(output_csv, index=False)
    with output_paths.open("w", encoding="utf-8") as stream:
        for path in curated["mxl_abs_path"]:
            stream.write(f"{path}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Curate a balanced solo-piano corpus from PDMX metadata.")
    parser.add_argument("--dataset-root", type=Path, default=Path("15571083"), help="Path to the extracted PDMX dataset directory.")
    parser.add_argument("--csv", type=Path, default=None, help="Path to PDMX.csv (defaults to <dataset-root>/PDMX.csv).")
    parser.add_argument("--min-rating", type=float, default=4.0, help="Minimum average rating to keep a score.")
    parser.add_argument("--max-per-composer", type=int, default=None, help="Maximum item count per composer (defaults to the smallest available count).")
    parser.add_argument("--skip-license-filter", action="store_true", help="Allow entries with potential license conflicts.")
    parser.add_argument("--skip-deduplicated-filter", action="store_true", help="Ignore the rated_deduplicated subset flag.")
    parser.add_argument("--skip-unique-filter", action="store_true", help="Ignore the best_unique_arrangement flag.")
    parser.add_argument("--output-csv", type=Path, default=Path("data/curated/solo_piano_corpus.csv"), help="Destination CSV for curated metadata.")
    parser.add_argument("--output-paths", type=Path, default=Path("data/curated/solo_piano_mxl_paths.txt"), help="Destination text file listing absolute MusicXML paths.")

    args = parser.parse_args(argv)

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        parser.error(f"Dataset root not found: {dataset_root}")

    csv_path = args.csv.resolve() if args.csv else dataset_root / "PDMX.csv"
    if not csv_path.exists():
        parser.error(f"Metadata CSV not found: {csv_path}")

    curated = curate_corpus(
        csv_path=csv_path,
        dataset_root=dataset_root,
        min_rating=args.min_rating,
        require_no_license_conflict=not args.skip_license_filter,
        require_deduplicated=not args.skip_deduplicated_filter,
        require_best_unique=not args.skip_unique_filter,
        max_per_composer=args.max_per_composer,
    )

    write_outputs(curated, args.output_csv.resolve(), args.output_paths.resolve())

    summary = (
        curated.groupby("composer_label").size().rename("count").sort_index().to_frame()
        .assign(total=len(curated))
    )
    print(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
