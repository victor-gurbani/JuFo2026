"""Annotate MusicXML scores with harmonic coloration for dissonant material."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import shlex
import subprocess
from typing import Dict, List, Optional, Tuple

from music21 import chord as m21_chord
from music21 import expressions
from music21 import harmony
from music21 import note as m21_note
from music21 import roman
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
CHROMATIC_CHORD_COLOR = "#00CED1"
CHROMATIC_LYRIC = "chromatic-chord"
OFFSET_ROUND_DIGITS = 6


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


def _color_element(element: Music21Object, color: str, *, force: bool = True) -> None:
    style = getattr(element, "style", None)
    if style is None:
        return
    current_color = getattr(style, "color", None)
    if not force and current_color:
        return
    style.color = color


def _add_unique_lyric(element: Music21Object, text: str) -> None:
    lyrics = getattr(element, "lyrics", None)
    if lyrics:
        if any(getattr(item, "text", None) == text for item in lyrics):
            return
    adder = getattr(element, "addLyric", None)
    if callable(adder):
        adder(text)


def _quantize_offset(value: float) -> float:
    return round(float(value), OFFSET_ROUND_DIGITS)


def _collect_elements_by_offset(score: stream.Score) -> Dict[float, List[Music21Object]]:
    buckets: Dict[float, List[Music21Object]] = defaultdict(list)
    for element in score.recurse().getElementsByClass((m21_note.Note, m21_chord.Chord)):
        offset = _quantize_offset(float(element.offset or 0.0))
        buckets[offset].append(element)
    return buckets


def _describe_chord(ch: m21_chord.Chord, key_obj) -> Tuple[str, Optional[str], Optional[roman.RomanNumeral]]:
    roman_label: Optional[str] = None
    roman_obj: Optional[roman.RomanNumeral] = None
    if key_obj is not None:
        try:
            roman_obj = roman.romanNumeralFromChord(ch, key_obj)
            roman_label = roman_obj.figure
        except Exception:
            pass
    common = None
    try:
        common = ch.commonName
    except Exception:
        common = None
    if not common:
        try:
            common = ch.pitchedCommonName
        except Exception:
            common = None
    label = roman_label or common or "?"
    return label, roman_label, roman_obj


def _is_interesting_chord(
    ch: m21_chord.Chord,
    diatonic_pitch_classes: Optional[set[int]],
    label: str,
    roman_label: Optional[str],
) -> bool:
    pitch_classes = {
        pitch.pitchClass
        for pitch in getattr(ch, "pitches", [])
        if getattr(pitch, "pitchClass", None) is not None
    }
    if diatonic_pitch_classes is not None and pitch_classes:
        if not pitch_classes.issubset(diatonic_pitch_classes):
            return True
    if roman_label is None:
        return True
    return label == "?"


def _attach_text_expression(
    score: stream.Score,
    offset: float,
    text: str,
    measure_number: Optional[int],
) -> None:
    if not text:
        return
    target = score.parts[0] if score.parts else score
    if target is None:
        return
    measure_obj = None
    if measure_number is not None:
        try:
            measure_obj = target.measure(int(measure_number))
        except Exception:
            measure_obj = None
    container = measure_obj if measure_obj is not None else target
    expression = expressions.TextExpression(text)
    local_offset = max(0.0, offset - float(container.offset or 0.0))
    container.insert(local_offset, expression)


def _insert_chord_symbol(
    score: stream.Score,
    offset: float,
    source_chord: m21_chord.Chord,
    fallback_text: Optional[str],
) -> None:
    symbol = None
    try:
        symbol = harmony.chordSymbolFromChord(source_chord)
    except Exception:
        symbol = None
    fallback = fallback_text or ""
    needs_fallback = symbol is None or getattr(symbol, "figure", None) in {None, "N.C."}
    if needs_fallback and fallback:
        try:
            symbol = harmony.ChordSymbol(figure=fallback)
        except Exception:
            symbol = None
    if symbol is None:
        return
    if fallback and getattr(symbol, "figure", None) in {None, "N.C."}:
        try:
            symbol.figure = fallback
        except Exception:
            pass
    if fallback:
        _add_unique_lyric(symbol, fallback)
    target = score.parts[0] if score.parts else score
    measure_number = getattr(source_chord, "measureNumber", None)
    measure_obj = None
    if measure_number is not None and target is not None:
        try:
            measure_obj = target.measure(int(measure_number))
        except Exception:
            measure_obj = None
    container = measure_obj if measure_obj is not None else target
    if container is None:
        return
    local_offset = max(0.0, offset - float(container.offset or 0.0))
    container.insert(local_offset, symbol)


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

    key_obj = None
    try:
        key_obj = score.analyze("key")
    except Exception:
        key_obj = None

    diatonic_pitch_classes: Optional[set[int]] = None
    if key_obj is not None:
        try:
            pitch_classes = getattr(key_obj, "pitchClasses", None)
            if pitch_classes:
                diatonic_pitch_classes = {
                    int(pc) for pc in pitch_classes if pc is not None
                }
        except Exception:
            diatonic_pitch_classes = None

    elements_by_offset = _collect_elements_by_offset(score)
    seen_labels: set[Tuple[float, str]] = set()
    interesting_offsets: set[float] = set()

    for ch in chords:
        raw_offset = float(ch.offset or 0.0)
        quant_offset = _quantize_offset(raw_offset)
        label, roman_label, _ = _describe_chord(ch, key_obj)
        text_label = roman_label or label
        if text_label and (quant_offset, text_label) not in seen_labels:
            _attach_text_expression(score, raw_offset, text_label, getattr(ch, "measureNumber", None))
            _insert_chord_symbol(score, raw_offset, ch, text_label)
            seen_labels.add((quant_offset, text_label))
        if _is_interesting_chord(ch, diatonic_pitch_classes, label, roman_label):
            interesting_offsets.add(quant_offset)

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

    for offset in interesting_offsets:
        for element in elements_by_offset.get(offset, []):
            _color_element(element, CHROMATIC_CHORD_COLOR, force=False)
            _add_unique_lyric(element, CHROMATIC_LYRIC)
            if isinstance(element, m21_chord.Chord):
                for component in element.notes:
                    _color_element(component, CHROMATIC_CHORD_COLOR, force=False)
                    _add_unique_lyric(component, CHROMATIC_LYRIC)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(output_path), makeNotation=False)

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
