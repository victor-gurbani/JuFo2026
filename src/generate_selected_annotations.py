"""Generate annotated MusicXML files for the curated recital set."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from annotate_musicxml import COLOR_CATEGORY_CHOICES, annotate_score

DEFAULT_CORPUS = Path("data/curated/solo_piano_corpus.csv")


@dataclass(frozen=True)
class Selection:
    label: str
    composer_label: str
    title_substring: str
    output_path: Path


SELECTIONS: List[Selection] = [
    Selection(
        label="Bach - Fugue in C Major BWV 846",
        composer_label="Bach",
        title_substring="Fugue in C Major BWV 846",
        output_path=Path("figures/annotated/Bach_Fugue_in_C_Major_BWV_846_annotated.mxl"),
    ),
    Selection(
        label="Bach - Toccata and Fugue in D Minor",
        composer_label="Bach",
        title_substring="Toccata and Fugue in D Minor",
        output_path=Path("figures/annotated/Bach_Toccata_and_Fugue_in_D_Minor_annotated.mxl"),
    ),
    Selection(
        label="Mozart - Lacrimosa (Requiem)",
        composer_label="Mozart",
        title_substring="Lacrimosa",
        output_path=Path("figures/annotated/Mozart_Lacrimosa_Requiem_annotated.mxl"),
    ),
    Selection(
        label="Mozart - Sonata K545 (3rd movement)",
        composer_label="Mozart",
        title_substring="K 545 3rd",
        output_path=Path("figures/annotated/Mozart_Sonata_K545_3rd_Movement_annotated.mxl"),
    ),
    Selection(
        label="Chopin - Ballade No. 1",
        composer_label="Chopin",
        title_substring="Ballade No. 1",
        output_path=Path("figures/annotated/Chopin_Ballade_No1_annotated.mxl"),
    ),
    Selection(
        label="Chopin - Etude Op.25 No.11",
        composer_label="Chopin",
        title_substring="Op.25 No.11",
        output_path=Path("figures/annotated/Chopin_Etude_Op25_No11_annotated.mxl"),
    ),
    Selection(
        label="Debussy - Clair de Lune",
        composer_label="Debussy",
        title_substring="Clair de Lune",
        output_path=Path("figures/annotated/Debussy_Clair_de_Lune_annotated.mxl"),
    ),
    Selection(
        label="Debussy - La cathedrale engloutie",
        composer_label="Debussy",
        title_substring="engloutie",
        output_path=Path("figures/annotated/La_cathedrale_engloutie_annotated.mxl"),
    ),
]


def _find_score_path(df: pd.DataFrame, selection: Selection) -> Path:
    composer_series = df.get("composer_label")
    title_series = df.get("title")
    path_series = df.get("mxl_abs_path")

    if composer_series is None or title_series is None or path_series is None:
        raise ValueError("Corpus CSV must contain composer_label, title, and mxl_abs_path columns.")

    composer_mask = composer_series.astype(str).str.lower() == selection.composer_label.lower()
    title_mask = title_series.fillna(" ").str.contains(selection.title_substring, case=False, na=False)
    matches = df[composer_mask & title_mask]
    if matches.empty:
        raise ValueError(f"Could not find score for selection '{selection.label}'.")

    mxl_path = matches.iloc[0]["mxl_abs_path"]
    if not isinstance(mxl_path, str):
        raise ValueError(f"Invalid path for selection '{selection.label}'.")
    return Path(mxl_path).expanduser().resolve()


def generate_annotations(
    corpus_csv: Path,
    selections: Iterable[Selection],
    renderer_template: str | None,
    render_format: str,
    hide_dissonant_label: bool = True,
    hidden_color_categories: Iterable[str] | None = None,
) -> List[Path]:
    df = pd.read_csv(corpus_csv)
    outputs: List[Path] = []
    hidden_colors = set(hidden_color_categories or [])
    for selection in selections:
        mxl_path = _find_score_path(df, selection)
        output_path = selection.output_path
        annotate_score(
            mxl_path,
            output_path,
            renderer_template=renderer_template,
            render_format=render_format,
            hide_dissonant_label=hide_dissonant_label,
            hidden_color_categories=hidden_colors,
        )
        outputs.append(output_path)
        print(f"Annotated {selection.label} -> {output_path}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate the shared recital playlist scores.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Path to the curated corpus CSV (default: data/curated/solo_piano_corpus.csv).",
    )
    parser.add_argument(
        "--renderer-template",
        type=str,
        default=None,
        help=(
            "Optional command template for external rendering. Use {input}, {output}, and {format} placeholders, "
            "e.g. 'mscore -o {output} {input}'."
        ),
    )
    parser.add_argument(
        "--render-format",
        type=str,
        default="pdf",
        choices=["pdf", "png"],
        help="Output format when --renderer-template is supplied (default: pdf).",
    )
    parser.add_argument(
        "--show-dissonant-label",
        action="store_true",
        help="Include the 'dissonant-chord' lyric tag (hidden by default).",
    )
    parser.add_argument(
        "--hide-color",
        action="append",
        dest="hidden_colors",
        metavar="CATEGORY",
        choices=COLOR_CATEGORY_CHOICES,
        help="Skip coloring for the given category. May be passed multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hide_dissonant_label = not args.show_dissonant_label
    hidden_colors = args.hidden_colors or []
    outputs = generate_annotations(
        args.corpus,
        SELECTIONS,
        renderer_template=args.renderer_template,
        render_format=args.render_format,
        hide_dissonant_label=hide_dissonant_label,
        hidden_color_categories=hidden_colors,
    )
    print(f"Generated {len(outputs)} annotated scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
