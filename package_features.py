#!/usr/bin/env python3
"""Create a zip archive with combined feature documentation PDF and all figures.

This utility converts every ``*_Features.md`` markdown file in the repository
root into a single PDF (using pandoc) and bundles that PDF together with all
files under the ``figures/`` directory into a zip archive. The resulting package
is convenient for sharing documentation alongside the accompanying visuals.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List
import zipfile

REPO_ROOT = Path(__file__).resolve().parent
FEATURE_GLOB = "*_Features.md"
FIGURE_DIRNAME = "figures"
DEFAULT_PDF_NAME = "Feature_Documentation.pdf"
DEFAULT_ZIP_NAME = "feature_bundle.zip"
DEFAULT_PDF_ENGINE = "xelatex"
DEFAULT_MAIN_FONT = "Libertinus Serif"
DEFAULT_SANS_FONT = "Libertinus Sans"
DEFAULT_MONO_FONT = "Menlo"
DEFAULT_MATH_FONT = "XITS Math"


def find_feature_markdown_files(directory: Path) -> List[Path]:
    """Locate all *_Features.md files in the given directory."""

    files = sorted(directory.glob(FEATURE_GLOB))
    if not files:
        raise FileNotFoundError(
            f"Did not find any markdown files matching {FEATURE_GLOB} in {directory}"
        )
    return files


def ensure_pandoc_available() -> Path:
    """Locate the pandoc executable or raise an informative error."""

    pandoc_path = shutil.which("pandoc")
    if pandoc_path is None:
        raise RuntimeError(
            "pandoc is required to convert markdown to PDF. Install it first: "
            "https://pandoc.org/installing.html"
        )
    return Path(pandoc_path)


def convert_markdown_to_pdf(
    markdown_files: Iterable[Path],
    output_pdf: Path,
    *,
    pdf_engine: str = DEFAULT_PDF_ENGINE,
    paper_size: str = "letter",
    margin: str = "1in",
    main_font: str | None = DEFAULT_MAIN_FONT,
    sans_font: str | None = DEFAULT_SANS_FONT,
    mono_font: str | None = DEFAULT_MONO_FONT,
    math_font: str | None = DEFAULT_MATH_FONT,
) -> None:
    """Run pandoc to concatenate markdown files into a single PDF."""

    pandoc_path = ensure_pandoc_available()
    command = [str(pandoc_path), "-s", "-o", str(output_pdf)]
    if pdf_engine:
        command.extend(["--pdf-engine", pdf_engine])
    if paper_size:
        command.extend(["-V", f"papersize={paper_size}"])
    if margin:
        command.extend(["-V", f"geometry:margin={margin}"])
    
    # Add header configuration with fancyhdr
    command.extend(["-V", "header-includes=\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhf{}\\fancyhead[L]{Victor Gurbani}\\fancyhead[R]{2025-2026}\\renewcommand{\\headrulewidth}{0.4pt}"])
    
    resolved_main = resolve_font_path(main_font)
    resolved_sans = resolve_font_path(sans_font)
    resolved_mono = resolve_font_path(mono_font)
    resolved_math = resolve_font_path(math_font)
    if resolved_main:
        command.extend(["-V", f"mainfont={resolved_main}"])
    if resolved_sans:
        command.extend(["-V", f"sansfont={resolved_sans}"])
    if resolved_mono:
        command.extend(["-V", f"monofont={resolved_mono}"])
    if resolved_math:
        command.extend(["-V", f"mathfont={resolved_math}"])
    command.extend(str(path) for path in markdown_files)
    subprocess.run(command, check=True)


def iter_figure_files(figure_root: Path) -> Iterable[Path]:
    """Yield all files under the figures directory recursively."""

    if not figure_root.exists():
        return []
    if not figure_root.is_dir():
        raise NotADirectoryError(f"Expected {figure_root} to be a directory of figures")
    return (path for path in figure_root.rglob("*") if path.is_file())


def build_zip_archive(pdf_path: Path, figure_files: Iterable[Path], output_zip: Path) -> None:
    """Create a zip archive containing the supplied PDF and figure files."""

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(pdf_path, arcname=pdf_path.name)
        for figure in figure_files:
            relative = figure.relative_to(REPO_ROOT)
            archive.write(figure, arcname=str(relative))


def resolve_font_path(font_spec: str | None) -> str | None:
    """Resolve a font specification to an absolute path when possible."""

    if not font_spec:
        return None

    if font_spec.startswith("[") and font_spec.endswith("]"):
        return font_spec

    candidate = Path(font_spec).expanduser()
    if candidate.exists():
        resolved_path = candidate.resolve().as_posix()
        return f"\\detokenize{{{resolved_path}}}"

    try:
        result = subprocess.run(
            ["kpsewhich", font_spec],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return font_spec

    resolved = result.stdout.strip()
    if result.returncode == 0 and resolved:
        resolved_path = Path(resolved).resolve().as_posix()
        return f"\\detokenize{{{resolved_path}}}"
    return font_spec


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=REPO_ROOT / DEFAULT_PDF_NAME,
        help=f"Destination path for the combined features PDF (default: {DEFAULT_PDF_NAME}).",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=REPO_ROOT / DEFAULT_ZIP_NAME,
        help=f"Destination path for the output archive (default: {DEFAULT_ZIP_NAME}).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=REPO_ROOT / FIGURE_DIRNAME,
        help="Directory containing figures to include in the archive (default: figures/).",
    )
    parser.add_argument(
        "--pdf-engine",
        type=str,
        default=DEFAULT_PDF_ENGINE,
        help=(
            "Pandoc PDF engine to use (default: xelatex). "
            "Set to an empty string to omit --pdf-engine."
        ),
    )
    parser.add_argument(
        "--paper-size",
        type=str,
        default="letter",
        help="Paper size to pass to pandoc (default: letter).",
    )
    parser.add_argument(
        "--margin",
        type=str,
        default="1in",
        help="Page margin (via geometry) to pass to pandoc (default: 1in).",
    )
    parser.add_argument(
        "--main-font",
        type=str,
        default=DEFAULT_MAIN_FONT,
        help="Primary serif font for the PDF (default: Libertinus Serif).",
    )
    parser.add_argument(
        "--sans-font",
        type=str,
        default=DEFAULT_SANS_FONT,
        help="Sans-serif font for the PDF (default: Libertinus Sans).",
    )
    parser.add_argument(
        "--mono-font",
        type=str,
        default=DEFAULT_MONO_FONT,
        help="Monospaced font for the PDF (default: Libertinus Mono).",
    )
    parser.add_argument(
        "--math-font",
        type=str,
        default=DEFAULT_MATH_FONT,
        help="Math font for the PDF (default: XITS Math).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    try:
        markdown_files = find_feature_markdown_files(REPO_ROOT)
        convert_markdown_to_pdf(
            markdown_files,
            args.pdf,
            pdf_engine=args.pdf_engine,
            paper_size=args.paper_size,
            margin=args.margin,
            main_font=args.main_font,
            sans_font=args.sans_font,
            mono_font=args.mono_font,
            math_font=args.math_font,
        )
        figure_files = list(iter_figure_files(args.figure_dir))
        build_zip_archive(args.pdf, figure_files, args.zip)
    except subprocess.CalledProcessError as exc:
        print(f"[error] pandoc failed with exit code {exc.returncode}")
        return exc.returncode or 1
    except Exception as exc:  # pragma: no cover - command line feedback only
        print(f"[error] {exc}")
        return 1

    print(f"Created PDF at {args.pdf}")
    print(f"Created archive at {args.zip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
