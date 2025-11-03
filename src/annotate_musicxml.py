"""Annotate MusicXML scores with harmonic coloration for dissonant material."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
from typing import Dict, List, Optional, Tuple

from music21 import chord as m21_chord
from music21 import note as m21_note
from music21 import stream
from music21.base import Music21Object

import harmonic_features as hf
from score_parser import parse_score

DISSONANT_COLORS = {
    "passing_tone": "#FFA500",  # orange
    "appoggiatura": "#8A2BE2",  # blue-violet
    "other_dissonance": "#FF0000",  # red
    "dissonant_chord": "#FF6666",  # light red
}


def _classify_melodic_dissonances(score: stream.Score, chord_index) -> List[Tuple[m21_note.Note, str]]:
    parts = list(score.parts)
    if not parts:
        return []

    melodic_notes = [n for n in parts[0].flatten().notes if isinstance(n, m21_note.Note)]
    if len(melodic_notes) < 3:
        return []

    classifications: List[Tuple[m21_note.Note, str]] = []
    for idx, current in enumerate(melodic_notes):
        prev_note = melodic_notes[idx - 1] if idx > 0 else None
        next_note = melodic_notes[idx + 1] if idx + 1 < len(melodic_notes) else None
        if not prev_note or not next_note:
            continue
        if current.tie and current.tie.type in {"stop", "continue"}:
            continue
        offset = float(current.offset or 0.0)
        chord_now = hf.find_chord_for_offset(chord_index, offset)
        if chord_now is None:
            continue
        next_offset = float(next_note.offset or 0.0)
        chord_next = hf.find_chord_for_offset(chord_index, next_offset)
        label = hf.classify_dissonant_note(current, prev_note, next_note, chord_now, chord_next)
        if label is None:
            continue
        classifications.append((current, label))
    return classifications


def _color_element(element: Music21Object, color: str) -> None:
    style = getattr(element, "style", None)
    if style is not None:
        style.color = color


def annotate_score(
    mxl_path: Path,
    output_path: Path,
    renderer_template: Optional[str] = None,
    render_format: str = "pdf",
) -> None:
    score = parse_score(mxl_path)
    chordified = hf.chordify_score(score)
    chords = hf.extract_chords(chordified)
    chord_index = hf.build_chord_index(chords)

    melodic_entries = _classify_melodic_dissonances(score, chord_index)
    melodic_map = {id(note): label for note, label in melodic_entries}

    for element in score.recurse().notes:
        offset = float(element.offset or 0.0)
        harmony = hf.find_chord_for_offset(chord_index, offset)
        dissonant = False
        if harmony is not None:
            try:
                dissonant = not harmony.isConsonant()
            except Exception:
                dissonant = False
        if isinstance(element, m21_note.Note):
            note_id = id(element)
            if note_id in melodic_map:
                label = melodic_map[note_id]
                _color_element(element, DISSONANT_COLORS[label])
                element.addLyric(label)
                continue
            if dissonant:
                _color_element(element, DISSONANT_COLORS["dissonant_chord"])
                element.addLyric("dissonant-chord")
        elif isinstance(element, m21_chord.Chord):
            if dissonant:
                _color_element(element, DISSONANT_COLORS["dissonant_chord"])
                for component in element.notes:
                    _color_element(component, DISSONANT_COLORS["dissonant_chord"])
                    component.addLyric("dissonant-chord")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(output_path))

    if renderer_template:
        render_path = output_path.with_suffix(f".{render_format}")
        command = renderer_template.format(
            input=str(output_path),
            output=str(render_path),
            format=render_format,
        )
        try:
            subprocess.run(
                shlex.split(command),
                check=True,
            )
            print(f"Rendered {render_format.upper()} via renderer -> {render_path}")
        except subprocess.CalledProcessError as exc:
            print(f"[warn] Renderer failed ({exc}). Annotated MusicXML still available at {output_path}.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Color dissonant events inside a MusicXML score.")
    parser.add_argument("--mxl", type=Path, required=True, help="Input MusicXML file to annotate.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the annotated MusicXML file.",
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
        help="Output format when --renderer is supplied (default: pdf).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    annotate_score(
        args.mxl,
        args.output,
        renderer_template=args.renderer_template,
        render_format=args.render_format,
    )
    print(f"Annotated score written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
