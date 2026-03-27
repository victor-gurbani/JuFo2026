"""
Train a Neural Network (MLP) and compare its performance head-to-head against
the previously trained Random Forest classifier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from feature_embedding import _merge_feature_tables, _prepare_feature_matrix

DEFAULT_HARMONIC = Path("data/features/harmonic_features.csv")
DEFAULT_MELODIC = Path("data/features/melodic_features.csv")
DEFAULT_RHYTHMIC = Path("data/features/rhythmic_features.csv")
DEFAULT_FIG_DIR = Path("figures/random_forest")
DEFAULT_MODEL_DIR = Path("data/models")
RF_MODEL_PATH = DEFAULT_MODEL_DIR / "random_forest_composer.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Neural Network and compare with Random Forest."
    )
    parser.add_argument(
        "--harmonic",
        type=Path,
        default=DEFAULT_HARMONIC,
        help="Path to harmonic features CSV.",
    )
    parser.add_argument(
        "--melodic",
        type=Path,
        default=DEFAULT_MELODIC,
        help="Path to melodic features CSV.",
    )
    parser.add_argument(
        "--rhythmic",
        type=Path,
        default=DEFAULT_RHYTHMIC,
        help="Path to rhythmic features CSV.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    return parser.parse_args()


def _print_comparison_table(
    rf_acc: float,
    nn_acc: float,
    rf_f1: float,
    nn_f1: float,
) -> None:
    """Print head-to-head comparison table between RF and NN."""
    print("\n" + "=" * 80)
    print("NEURAL NETWORK vs RANDOM FOREST COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<25} {'Random Forest':<25} {'Neural Network':<25} {'Winner':<10}")
    print("-" * 80)

    acc_winner = "RF" if rf_acc > nn_acc else ("NN" if nn_acc > rf_acc else "Tie")
    print(
        f"{'5-Fold CV Accuracy':<25} {rf_acc:<25.4f} {nn_acc:<25.4f} {acc_winner:<10}"
    )

    f1_winner = "RF" if rf_f1 > nn_f1 else ("NN" if nn_f1 > rf_f1 else "Tie")
    print(f"{'Macro F1 Score':<25} {rf_f1:<25.4f} {nn_f1:<25.4f} {f1_winner:<10}")
    print("=" * 80)


def generate_bar_chart(
    rf_cr: dict, nn_cr: dict, classes: np.ndarray, output_path: Path
):
    """Generate a grouped bar chart showing Precision and Recall side-by-side."""
    x = np.arange(len(classes))
    width = 0.2

    rf_precisions = [rf_cr[cls]["precision"] for cls in classes if cls in rf_cr]
    rf_recalls = [rf_cr[cls]["recall"] for cls in classes if cls in rf_cr]

    nn_precisions = [nn_cr[cls]["precision"] for cls in classes if cls in nn_cr]
    nn_recalls = [nn_cr[cls]["recall"] for cls in classes if cls in nn_cr]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot RF
    ax.bar(
        x - width * 1.5,
        rf_precisions,
        width,
        label="RF Precision",
        color="#1f77b4",
        edgecolor="black",
    )
    ax.bar(
        x - width * 0.5,
        rf_recalls,
        width,
        label="RF Recall",
        color="#aec7e8",
        edgecolor="black",
    )

    # Plot NN
    ax.bar(
        x + width * 0.5,
        nn_precisions,
        width,
        label="NN Precision",
        color="#ff7f0e",
        edgecolor="black",
    )
    ax.bar(
        x + width * 1.5,
        nn_recalls,
        width,
        label="NN Recall",
        color="#ffbb78",
        edgecolor="black",
    )

    ax.set_ylabel("Score")
    ax.set_title("Random Forest vs Neural Network: Precision and Recall by Composer")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Comparison bar chart saved to {output_path}")


def main() -> int:
    args = parse_args()
    DEFAULT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading feature tables...")
    try:
        harm = pd.read_csv(args.harmonic)
        mel = pd.read_csv(args.melodic)
        rhy = pd.read_csv(args.rhythmic)
    except FileNotFoundError as e:
        print(f"Error loading features: {e}", file=sys.stderr)
        return 1

    combined = _merge_feature_tables(harm, mel, rhy)
    print(f"Combined dataset shape: {combined.shape}")

    # For NN, we use the FULL scaled feature matrix (no manual collinearity filtering)
    matrix, feature_cols, scaler = _prepare_feature_matrix(combined)
    X_full = pd.DataFrame(matrix, columns=feature_cols)

    COMPOSER_ORDER = ["Bach", "Mozart", "Chopin", "Debussy"]
    # Ensure chronological order for labels
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(COMPOSER_ORDER)
    y = (
        combined["composer_label"]
        .map({c: i for i, c in enumerate(COMPOSER_ORDER)})
        .values
    )

    print(f"Full feature matrix shape for NN: {X_full.shape}")

    print(f"Loading previously trained Random Forest model from {RF_MODEL_PATH}...")
    try:
        artifacts = joblib.load(RF_MODEL_PATH)
        rf_model = artifacts["model"]
        rf_feature_cols = artifacts["feature_cols"]
        print(f"Loaded RF model with {len(rf_feature_cols)} filtered features.")
    except FileNotFoundError:
        print(
            f"Error: RF model not found at {RF_MODEL_PATH}. Run classify_rf.py first.",
            file=sys.stderr,
        )
        return 1

    print("\nOptimizing Neural Network hyperparameters...")
    mlp = MLPClassifier(max_iter=1000, random_state=args.seed)

    param_distributions = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "activation": ["relu", "tanh", "logistic"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate_init": [0.001, 0.01],
    }

    # We use RandomizedSearchCV to optimize the NN using the FULL feature matrix
    search = RandomizedSearchCV(
        mlp,
        param_distributions=param_distributions,
        n_iter=15,
        cv=5,
        scoring="accuracy",
        random_state=args.seed,
        n_jobs=-1,
    )

    search.fit(X_full, y)
    best_nn = search.best_estimator_
    print(f"Best NN parameters found: {search.best_params_}")

    print("\nEvaluating BOTH models using 5-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    # 1. RF Cross-validation (MUST use the filtered features it was trained on)
    X_rf = X_full[rf_feature_cols]
    cv_rf_pred = cross_val_predict(rf_model, X_rf, y, cv=cv, n_jobs=-1)

    # 2. NN Cross-validation (Uses full feature matrix)
    cv_nn_pred = cross_val_predict(best_nn, X_full, y, cv=cv, n_jobs=-1)

    # Calculate metrics
    rf_acc = accuracy_score(y, cv_rf_pred)
    nn_acc = accuracy_score(y, cv_nn_pred)

    rf_f1 = f1_score(y, cv_rf_pred, average="macro")
    nn_f1 = f1_score(y, cv_nn_pred, average="macro")

    _print_comparison_table(rf_acc, nn_acc, rf_f1, nn_f1)

    # Classification reports for charting
    rf_cr = classification_report(
        y, cv_rf_pred, target_names=label_encoder.classes_, output_dict=True
    )
    nn_cr = classification_report(
        y, cv_nn_pred, target_names=label_encoder.classes_, output_dict=True
    )

    print("\nNeural Network Classification Report (Aggregated 5-Fold CV):")
    print(classification_report(y, cv_nn_pred, target_names=label_encoder.classes_))

    chart_path = DEFAULT_FIG_DIR / "nn_vs_rf_comparison.png"
    print("Generating side-by-side bar chart of Precision and Recall...")
    generate_bar_chart(rf_cr, nn_cr, label_encoder.classes_, chart_path)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
