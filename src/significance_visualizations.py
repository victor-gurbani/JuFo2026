"""Generate visual summaries for Phase 2 significance testing outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors

DEFAULT_ANOVA = Path("data/stats/anova_summary.csv")
DEFAULT_TUKEY = Path("data/stats/tukey_hsd.csv")
DEFAULT_FIGURE_DIR = Path("figures/significance")
DEFAULT_TOP_N = 15
DEFAULT_EXCLUDE = [
    "note_count",
    "note_event_count",
    "chord_event_count",
    "chord_quality_total",
    "roman_chord_count",
    "dissonant_note_count",
]

BANDS = ["Bach", "Mozart", "Chopin", "Debussy"]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"File {path} is empty; run significance_tests first.")
    return df


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in DataFrame.")


def _format_pair(a: str, b: str) -> str:
    return f"{a} vs {b}"


def _canonical_pair(a: str, b: str) -> str:
    try:
        idx_a = BANDS.index(a)
    except ValueError:
        idx_a = len(BANDS)
    try:
        idx_b = BANDS.index(b)
    except ValueError:
        idx_b = len(BANDS)
    if idx_a <= idx_b:
        return _format_pair(a, b)
    return _format_pair(b, a)


def _ordered_pair_labels() -> List[str]:
    labels: List[str] = []
    for idx, left in enumerate(BANDS):
        for right in BANDS[idx + 1 :]:
            labels.append(_format_pair(left, right))
    return labels


def plot_top_anova(anova: pd.DataFrame, figure_dir: Path, top_n: int) -> Path:
    _ensure_columns(anova, ["feature", "source", "p_value"])
    filtered = anova.dropna(subset=["p_value"]).copy()
    filtered["neg_log_p"] = -np.log10(filtered["p_value"].clip(lower=float(np.finfo(float).tiny)))
    top = filtered.sort_values("p_value").head(top_n)
    if top.empty:
        raise ValueError("No ANOVA rows available to plot.")

    plt.figure(figsize=(8, max(4, top_n * 0.35)))
    sns.barplot(data=top, y="feature", x="neg_log_p", hue="source", dodge=False)
    plt.xlabel(r"$-\log_{10}(p)$")
    plt.ylabel("Feature")
    plt.title("Top ANOVA Features by Significance")
    plt.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "top_anova_bar.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def aggregate_pair_significance(tukey: pd.DataFrame) -> pd.DataFrame:
    required = ["feature", "group_1", "group_2", "p_adj", "reject", "meandiff"]
    _ensure_columns(tukey, required)
    tukey = tukey.copy()
    tukey["pair"] = tukey.apply(lambda row: _canonical_pair(str(row["group_1"]), str(row["group_2"])), axis=1)
    tukey["reject"] = tukey["reject"].astype(bool)

    subset = tukey[tukey["reject"]]
    if subset.empty:
        return pd.DataFrame()
    features = subset.groupby(["feature", "pair"])["meandiff"].mean().unstack(fill_value=0)
    return features


def _pair_matrix(values: pd.Series) -> pd.DataFrame:
    matrix = pd.DataFrame(0.0, index=BANDS, columns=BANDS)
    for pair, value in values.items():
        a, b = str(pair).split(" vs ")
        matrix.loc[a, b] = float(value)
        matrix.loc[b, a] = float(value)
    return matrix


def plot_pair_heatmap(counts: pd.Series, figure_dir: Path) -> Path:
    matrix = _pair_matrix(counts)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Significant Features"},
    )
    plt.title("Composer Pair Differences (count)")
    plt.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "tukey_pair_heatmap.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_pair_effect_heatmap(effect: pd.Series, figure_dir: Path, name: str, cmap: str, slug: str) -> Path:
    matrix = _pair_matrix(effect)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        center=0 if matrix.values.min() < 0 else None,
        cbar_kws={"label": name},
    )
    plt.title(f"Composer Pair Differences ({name})")
    plt.tight_layout()
    output_path = figure_dir / f"tukey_pair_{slug}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def _apply_exclusions(feature_means: pd.DataFrame, patterns: Sequence[str]) -> pd.DataFrame:
    if not patterns:
        return feature_means
    include_mask = pd.Series(True, index=feature_means.index)
    index_series = feature_means.index.to_series().astype(str)
    for pattern in patterns:
        include_mask &= ~index_series.str.contains(pattern, case=False, regex=False)
    return feature_means.loc[include_mask]


def plot_feature_pair_matrix(
    feature_means: pd.DataFrame,
    figure_dir: Path,
    top_n: int,
    symlog: bool = False,
) -> Path:
    if feature_means.empty:
        raise ValueError("No significant Tukey comparisons available for feature heatmap.")
    ordered = feature_means.copy()
    available_pairs = ordered.columns.tolist()
    pair_labels = [label for label in _ordered_pair_labels() if label in available_pairs]
    ordered = ordered[pair_labels]

    # Select top features based on max absolute mean difference
    magnitudes = ordered.abs().max(axis=1).sort_values(ascending=False)
    top = ordered.loc[magnitudes.head(top_n).index]

    plt.figure(figsize=(len(pair_labels) * 1.4 + 2, max(4, top_n * 0.45)))
    norm = colors.SymLogNorm(linthresh=0.05, linscale=1.0, base=10) if symlog else None
    sns.heatmap(
        top,
        cmap="RdBu_r",
        center=0,
        norm=norm,
        annot=False,
        cbar_kws={"label": "Mean Difference (group1 - group2)"},
    )
    plt.title("Mean Difference Heatmap for Significant Comparisons")
    plt.ylabel("Feature")
    plt.xlabel("Composer Pair")
    plt.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "tukey_feature_heatmap.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_normalized_heatmap(
    feature_means: pd.DataFrame,
    figure_dir: Path,
    top_n: int,
) -> Path:
    if feature_means.empty:
        raise ValueError("No data available for normalized heatmap.")
    std = feature_means.std(axis=1)
    normalized = feature_means.divide(std.replace(0, np.nan), axis=0)
    normalized = normalized.dropna(how="all")
    if normalized.empty:
        raise ValueError("Normalization removed all features.")
    magnitudes = normalized.abs().max(axis=1).sort_values(ascending=False)
    top = normalized.loc[magnitudes.head(top_n).index]

    plt.figure(figsize=(len(normalized.columns) * 1.4 + 2, max(4, top_n * 0.45)))
    sns.heatmap(
        top,
        cmap="RdBu_r",
        center=0,
        annot=False,
        cbar_kws={"label": "Std-Normalized Mean Difference"},
    )
    plt.title("Normalized Mean Difference Heatmap")
    plt.ylabel("Feature")
    plt.xlabel("Composer Pair")
    plt.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "tukey_feature_heatmap_normalized.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate significance testing visualizations.")
    parser.add_argument("--anova", type=Path, default=DEFAULT_ANOVA, help="Path to ANOVA summary CSV.")
    parser.add_argument("--tukey", type=Path, default=DEFAULT_TUKEY, help="Path to Tukey HSD CSV.")
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR, help="Directory for output figures.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Number of top features to visualize (default 15).")
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=list(DEFAULT_EXCLUDE),
        help="Substring to exclude from feature names in pair heatmaps (can repeat).",
    )
    parser.add_argument(
        "--no-symlog",
        action="store_true",
        help="Disable symmetric log scaling for mean-difference heatmap.",
    )
    parser.add_argument(
        "--skip-normalized",
        action="store_true",
        help="Skip generation of the normalized mean-difference heatmap.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)

    anova_df = load_csv(args.anova)
    tukey_df = load_csv(args.tukey)

    if args.top_n <= 0:
        raise ValueError("--top-n must be positive.")

    figure_dir = args.figure_dir
    outputs: Dict[str, Path] = {}

    outputs["top_anova"] = plot_top_anova(anova_df, figure_dir, args.top_n)

    feature_means = aggregate_pair_significance(tukey_df)
    filtered = _apply_exclusions(feature_means, args.exclude_pattern or [])

    if not filtered.empty:
        counts_series = (filtered.values != 0).sum(axis=0)
        counts_series = pd.Series(counts_series, index=filtered.columns)
        counts_series = counts_series.reindex(_ordered_pair_labels(), fill_value=0)
        outputs["pair_heatmap"] = plot_pair_heatmap(counts_series, figure_dir)

        mean_diff = filtered.mean(axis=0).reindex(_ordered_pair_labels(), fill_value=0.0)
        mean_abs = filtered.abs().mean(axis=0).reindex(_ordered_pair_labels(), fill_value=0.0)
        outputs["pair_heatmap_effect"] = plot_pair_effect_heatmap(
            mean_diff, figure_dir, "Mean Difference", "PuOr", "mean_difference"
        )
        outputs["pair_heatmap_effect_abs"] = plot_pair_effect_heatmap(
            mean_abs, figure_dir, "Mean |Difference|", "YlGnBu", "mean_abs_difference"
        )
    if not feature_means.empty:
        filtered = _apply_exclusions(feature_means, args.exclude_pattern or [])
        if not filtered.empty:
            outputs["feature_heatmap"] = plot_feature_pair_matrix(
                filtered,
                figure_dir,
                min(args.top_n, filtered.shape[0]),
                symlog=not args.no_symlog,
            )
            if not args.skip_normalized:
                outputs["feature_heatmap_normalized"] = plot_normalized_heatmap(
                    filtered,
                    figure_dir,
                    min(args.top_n, filtered.shape[0]),
                )

    print("Generated visualizations:")
    for name, path in outputs.items():
        print(f" - {name}: {path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
