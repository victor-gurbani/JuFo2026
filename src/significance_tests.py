"""Run ANOVA and Tukey HSD significance tests across extracted feature tables."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:  # statsmodels is optional but preferred for Tukey HSD
    from statsmodels.stats.multicomp import pairwise_tukeyhsd  # type: ignore
except Exception:  # pragma: no cover - narrow environments may omit statsmodels
    pairwise_tukeyhsd = None  # type: ignore

try:  # statsmodels also provides convenient FDR correction utilities
    from statsmodels.stats.multitest import fdrcorrection  # type: ignore
except Exception:  # pragma: no cover
    fdrcorrection = None  # type: ignore

DEFAULT_HARMONIC_FEATURES = Path("data/features/harmonic_features.csv")
DEFAULT_MELODIC_FEATURES = Path("data/features/melodic_features.csv")
DEFAULT_RHYTHMIC_FEATURES = Path("data/features/rhythmic_features.csv")
DEFAULT_ANOVA_OUTPUT = Path("data/stats/anova_summary.csv")
DEFAULT_TUKEY_OUTPUT = Path("data/stats/tukey_hsd.csv")

ALPHA = 0.05
MIN_GROUP_SIZE = 3
MIN_GROUPS = 3


@dataclass
class AnovaResult:
    feature: str
    source: str
    groups_tested: int
    min_group_size: int
    max_group_size: int
    f_statistic: Optional[float]
    p_value: Optional[float]
    significant: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature": self.feature,
            "source": self.source,
            "groups_tested": self.groups_tested,
            "min_group_size": self.min_group_size,
            "max_group_size": self.max_group_size,
            "f_statistic": self.f_statistic,
            "p_value": self.p_value,
            "significant": self.significant,
        }


def _load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    df = pd.read_csv(path)
    if "composer_label" not in df.columns:
        raise ValueError(f"Missing composer_label column in {path}")
    if "mxl_path" not in df.columns:
        raise ValueError(f"Missing mxl_path column in {path}")
    # Normalise composer labels to strings for grouping stability.
    df["composer_label"] = df["composer_label"].astype(str)
    return df


def _numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    blacklist = {"index"}
    return [col for col in numeric_cols if col not in blacklist]


def _collect_groups(df: pd.DataFrame, feature: str) -> List[Tuple[str, np.ndarray]]:
    grouped: List[Tuple[str, np.ndarray]] = []
    for label, group in df[["composer_label", feature]].dropna().groupby("composer_label"):
        values = group[feature].to_numpy(dtype=float)
        if values.size >= MIN_GROUP_SIZE:
            grouped.append((str(label), values))
    return grouped


def _anova_for_feature(df: pd.DataFrame, feature: str, source: str) -> Optional[AnovaResult]:
    groups = _collect_groups(df, feature)
    if len(groups) < MIN_GROUPS:
        return None
    min_size = int(min(len(values) for _, values in groups))
    max_size = int(max(len(values) for _, values in groups))
    if min_size < MIN_GROUP_SIZE:
        return None
    try:
        f_stat, p_value = stats.f_oneway(*[values for _, values in groups])
    except ValueError:
        return None
    significant = bool(p_value <= ALPHA)
    return AnovaResult(
        feature=feature,
        source=source,
        groups_tested=len(groups),
        min_group_size=min_size,
        max_group_size=max_size,
        f_statistic=float(f_stat),
        p_value=float(p_value),
        significant=significant,
    )


def _tukey_for_feature(df: pd.DataFrame, feature: str, source: str, alpha: float) -> Optional[pd.DataFrame]:
    groups = _collect_groups(df, feature)
    if len(groups) < MIN_GROUPS:
        return None

    labels = [label for label, _ in groups]
    arrays = [values for _, values in groups]

    if pairwise_tukeyhsd is not None:
        subset = df[["composer_label", feature]].dropna()
        if subset.groupby("composer_label").size().min() < MIN_GROUP_SIZE:
            return None
        try:
            tukey = pairwise_tukeyhsd(
                endog=subset[feature].to_numpy(dtype=float),
                groups=subset["composer_label"].to_numpy(),
                alpha=alpha,
            )
        except Exception:
            tukey = None
        if tukey is not None:
            summary = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
            summary.insert(0, "feature", feature)
            summary.insert(1, "source", source)
            summary = summary.rename(
                columns={
                    "group1": "group_1",
                    "group2": "group_2",
                    "p-adj": "p_adj",
                    "lower": "ci_lower",
                    "upper": "ci_upper",
                }
            )
            return summary

    if hasattr(stats, "tukey_hsd"):
        try:
            tukey_res = stats.tukey_hsd(*arrays)
        except Exception:
            return None
        ci_level = 1.0 - alpha if 0.0 < alpha < 1.0 else 0.95
        ci = tukey_res.confidence_interval(confidence_level=ci_level)
        rows: List[Dict[str, object]] = []
        for idx in range(len(labels)):
            for jdx in range(idx + 1, len(labels)):
                mean_diff = float(np.mean(arrays[idx]) - np.mean(arrays[jdx]))
                p_val = float(tukey_res.pvalue[idx, jdx])
                rows.append(
                    {
                        "feature": feature,
                        "source": source,
                        "group_1": labels[idx],
                        "group_2": labels[jdx],
                        "meandiff": mean_diff,
                        "p_adj": p_val,
                        "ci_lower": float(ci.low[idx, jdx]),
                        "ci_upper": float(ci.high[idx, jdx]),
                        "reject": bool(p_val <= alpha),
                    }
                )
        return pd.DataFrame(rows)

    return None


def _bh_fdr(p_values: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction with statsmodels fallback."""

    if fdrcorrection is not None:
        reject, pvals_corr = fdrcorrection(p_values, alpha=alpha)
        return reject, pvals_corr

    m = p_values.size
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(p_values)
    ranked = np.empty_like(p_values, dtype=float)
    cumulative = (np.arange(1, m + 1) / m) * alpha
    ranked_vals = p_values[order]
    below = ranked_vals <= cumulative
    reject = np.zeros(m, dtype=bool)
    if below.any():
        max_idx = np.where(below)[0].max()
        reject[order[: max_idx + 1]] = True

    adjusted = np.empty_like(p_values, dtype=float)
    prev = 1.0
    for idx in reversed(range(m)):
        rank = idx + 1
        value = min(prev, (p_values[order[idx]] * m) / rank)
        prev = value
        adjusted[order[idx]] = value

    return reject, np.clip(adjusted, 0.0, 1.0)


def run_significance_tests(feature_paths: Dict[str, Path], alpha: float, run_tukey: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    anova_records: List[Dict[str, object]] = []
    tukey_frames: List[pd.DataFrame] = []
    dataframes: Dict[str, pd.DataFrame] = {}

    for source, path in feature_paths.items():
        if path is None:
            continue
        df = _load_features(path)
        dataframes[source] = df
        columns = _numeric_feature_columns(df)
        for column in columns:
            if column in {"composer_label"}:
                continue
            result = _anova_for_feature(df, column, source)
            if result is None:
                continue
            anova_records.append(result.to_dict())

    anova_df = pd.DataFrame(anova_records)
    if not anova_df.empty:
        anova_df = anova_df.sort_values("p_value", ascending=True)

        anova_df["alpha_threshold"] = alpha
        anova_df["significant_alpha"] = anova_df["p_value"] <= alpha
        anova_df["significant"] = anova_df["significant_alpha"]

        num_tests = max(len(anova_df), 1)
        bonf_threshold = alpha / num_tests
        anova_df["bonferroni_threshold"] = bonf_threshold
        anova_df["p_value_bonferroni"] = np.clip(anova_df["p_value"] * num_tests, 0.0, 1.0)
        anova_df["significant_bonferroni"] = anova_df["p_value"] <= bonf_threshold

        reject_fdr, pvals_fdr = _bh_fdr(anova_df["p_value"].to_numpy(dtype=float), alpha=alpha)
        anova_df["p_value_fdr"] = pvals_fdr
        anova_df["significant_fdr"] = reject_fdr

        if run_tukey:
            mask = anova_df["significant_bonferroni"] | anova_df["significant_fdr"]
            selected = anova_df.loc[mask, ["feature", "source"]]
            for feature, source in selected.itertuples(index=False):
                df = dataframes.get(source)
                if df is None:
                    continue
                tukey_df = _tukey_for_feature(df, feature, source, alpha)
                if tukey_df is not None:
                    tukey_frames.append(tukey_df)

    tukey_df = pd.concat(tukey_frames, ignore_index=True) if tukey_frames else pd.DataFrame()
    return anova_df, tukey_df


def write_results(anova_df: pd.DataFrame, tukey_df: pd.DataFrame, anova_output: Path, tukey_output: Path) -> None:
    anova_output.parent.mkdir(parents=True, exist_ok=True)
    tukey_output.parent.mkdir(parents=True, exist_ok=True)
    anova_df.to_csv(anova_output, index=False)
    if not tukey_df.empty:
        tukey_df.to_csv(tukey_output, index=False)
    else:
        # Write an empty file with headers for reproducibility.
        pd.DataFrame(columns=["feature", "source", "group_1", "group_2", "meandiff", "p_adj", "ci_lower", "ci_upper", "reject"]).to_csv(tukey_output, index=False)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANOVA and Tukey HSD tests across feature tables.")
    parser.add_argument("--harmonic-csv", type=Path, default=DEFAULT_HARMONIC_FEATURES, help="Harmonic feature table.")
    parser.add_argument("--melodic-csv", type=Path, default=DEFAULT_MELODIC_FEATURES, help="Melodic feature table.")
    parser.add_argument("--rhythmic-csv", type=Path, default=DEFAULT_RHYTHMIC_FEATURES, help="Rhythmic feature table.")
    parser.add_argument("--anova-output", type=Path, default=DEFAULT_ANOVA_OUTPUT, help="Destination CSV for ANOVA summary results.")
    parser.add_argument("--tukey-output", type=Path, default=DEFAULT_TUKEY_OUTPUT, help="Destination CSV for Tukey HSD post-hoc comparisons.")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Significance level for hypothesis testing (default: 0.05).")
    parser.add_argument("--no-tukey", action="store_true", help="Skip Tukey HSD even when ANOVA is significant.")
    parser.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE, help="Minimum samples per composer required for inclusion (default: 3).")
    parser.add_argument("--min-groups", type=int, default=MIN_GROUPS, help="Minimum number of composer groups required (default: 3).")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)

    global ALPHA, MIN_GROUP_SIZE, MIN_GROUPS  # configure thresholds based on CLI args
    ALPHA = float(args.alpha)
    MIN_GROUP_SIZE = int(max(2, args.min_group_size))
    MIN_GROUPS = int(max(2, args.min_groups))

    feature_paths = {
        "harmonic": args.harmonic_csv,
        "melodic": args.melodic_csv,
        "rhythmic": args.rhythmic_csv,
    }

    anova_df, tukey_df = run_significance_tests(feature_paths, alpha=ALPHA, run_tukey=not args.no_tukey)
    write_results(anova_df, tukey_df if not args.no_tukey else pd.DataFrame(), args.anova_output, args.tukey_output)

    if anova_df.empty:
        print("[warn] No ANOVA results were produced. Check group sizes and feature coverage.")
    else:
        preview_cols = [
            "feature",
            "source",
            "f_statistic",
            "p_value",
            "significant_alpha",
            "p_value_bonferroni",
            "significant_bonferroni",
            "p_value_fdr",
            "significant_fdr",
        ]
        display_df = anova_df[preview_cols].head(20)
        print(display_df.to_string(index=False))
        total_tests = len(anova_df)
        alpha_hits = int(anova_df["significant_alpha"].sum())
        bonf_hits = int(anova_df["significant_bonferroni"].sum())
        fdr_hits = int(anova_df["significant_fdr"].sum())
        bonf_threshold = float(anova_df["bonferroni_threshold"].iloc[0]) if "bonferroni_threshold" in anova_df.columns else args.alpha
        print(f"[info] Raw α={args.alpha:.3g}: {alpha_hits}/{total_tests} significant")
        print(f"[info] Bonferroni threshold α/{total_tests}≈{bonf_threshold:.3g}: {bonf_hits}/{total_tests} significant")
        print(f"[info] Benjamini–Hochberg FDR q<{args.alpha:.3g}: {fdr_hits}/{total_tests} significant")
        print(f"Saved ANOVA summary to {args.anova_output}")

    if not args.no_tukey:
        if tukey_df.empty:
            if pairwise_tukeyhsd is None:
                print("[warn] statsmodels is unavailable; Tukey HSD results were skipped.")
            else:
                print("[warn] No Tukey HSD results met significance thresholds.")
        else:
            print(f"Saved Tukey HSD comparisons to {args.tukey_output}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
