"""
Build a high-accuracy, highly interpretable Random Forest classifier based on our extracted features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import joblib

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as hc

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree, export_text

from feature_embedding import _merge_feature_tables, _prepare_feature_matrix

DEFAULT_HARMONIC = Path("data/features/harmonic_features.csv")
DEFAULT_MELODIC = Path("data/features/melodic_features.csv")
DEFAULT_RHYTHMIC = Path("data/features/rhythmic_features.csv")
DEFAULT_FIG_DIR = Path("figures/random_forest")
DEFAULT_MODEL_DIR = Path("data/models")


def _compute_tree_metrics(forest: RandomForestClassifier) -> dict:
    """
    Compute average tree depth and total nodes across all estimators in the forest.
    """
    avg_depth = np.mean([tree.get_depth() for tree in forest.estimators_])
    total_nodes = np.sum([tree.tree_.node_count for tree in forest.estimators_])
    return {"avg_depth": avg_depth, "total_nodes": int(total_nodes)}


def _print_comparison_table(
    baseline_acc: float,
    best_acc: float,
    baseline_metrics: dict,
    best_metrics: dict,
) -> None:
    """
    Print a comparison table between baseline unpruned and best pruned Random Forest.
    """
    print("\n" + "=" * 80)
    print("BASELINE vs BEST PRUNED RANDOM FOREST COMPARISON")
    print("=" * 80)
    print(
        f"{'Metric':<25} {'Baseline Unpruned':<25} {'Best Pruned':<25} {'Change':<10}"
    )
    print("-" * 80)

    # Accuracy
    acc_change = best_acc - baseline_acc
    acc_pct_change = (acc_change / baseline_acc * 100) if baseline_acc > 0 else 0
    print(
        f"{'5-Fold CV Accuracy':<25} {baseline_acc:<25.4f} {best_acc:<25.4f} "
        f"{acc_change:+.4f} ({acc_pct_change:+.1f}%)"
    )

    # Average Tree Depth
    depth_change = best_metrics["avg_depth"] - baseline_metrics["avg_depth"]
    depth_pct_change = (
        (depth_change / baseline_metrics["avg_depth"] * 100)
        if baseline_metrics["avg_depth"] > 0
        else 0
    )
    print(
        f"{'Avg Tree Depth':<25} {baseline_metrics['avg_depth']:<25.2f} "
        f"{best_metrics['avg_depth']:<25.2f} {depth_change:+.2f} ({depth_pct_change:+.1f}%)"
    )

    # Total Nodes
    nodes_change = best_metrics["total_nodes"] - baseline_metrics["total_nodes"]
    nodes_pct_change = (
        (nodes_change / baseline_metrics["total_nodes"] * 100)
        if baseline_metrics["total_nodes"] > 0
        else 0
    )
    print(
        f"{'Total Nodes (all trees)':<25} {baseline_metrics['total_nodes']:<25} "
        f"{best_metrics['total_nodes']:<25} {nodes_change:+} ({nodes_pct_change:+.1f}%)"
    )
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest classifier on musical features."
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


def main() -> int:
    args = parse_args()

    DEFAULT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

    matrix, feature_cols, scaler = _prepare_feature_matrix(combined)
    X = pd.DataFrame(matrix, columns=feature_cols)

    COMPOSER_ORDER = ["Bach", "Mozart", "Chopin", "Debussy"]
    # Ensure chronological order for labels
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(COMPOSER_ORDER)
    y = combined["composer_label"].map({c: i for i, c in enumerate(COMPOSER_ORDER)}).values

    print(f"Original features count: {len(feature_cols)}")
    print("Computing Spearman rank correlation matrix...")
    corr_matrix = spearmanr(X).correlation

    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1)

    dist_matrix = 1 - np.abs(corr_matrix)

    print("Performing hierarchical clustering to remove collinearity...")
    condensed_dist = squareform(dist_matrix, checks=False)
    linkage = hc.linkage(condensed_dist, method="average")

    cluster_ids = hc.fcluster(linkage, t=0.25, criterion="distance")

    cluster_dict = defaultdict(list)
    for i, cluster_id in enumerate(cluster_ids):
        cluster_dict[cluster_id].append(i)

    selected_features = []
    for cluster_id, indices in cluster_dict.items():
        selected_features.append(feature_cols[indices[0]])

    X = X[selected_features]
    feature_cols = selected_features
    print(f"Selected uncorrelated features count: {len(feature_cols)}")

    print("Splitting data into train/test sets for final model building...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    print("Training baseline unpruned Random Forest for comparison...")
    baseline_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        ccp_alpha=0.0,
        random_state=args.seed,
        n_jobs=-1,
    )
    baseline_rf.fit(X_train, y_train)
    baseline_metrics = _compute_tree_metrics(baseline_rf)

    print("\nOptimizing Random Forest hyperparameters...")
    rf = RandomForestClassifier(random_state=args.seed)

    # FORCED PRUNING: We remove ccp_alpha=0.0 to strictly enforce tree simplification.
    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, 20],
        "min_samples_leaf": [2, 3, 4],
        "ccp_alpha": [0.005, 0.01, 0.015, 0.02, 0.03],
    }

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring="accuracy",
        random_state=args.seed,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    print(f"Best parameters found: {search.best_params_}")
    best_metrics = _compute_tree_metrics(best_rf)

    print("\nEvaluating model using 5-Fold Cross-Validation on the ENTIRE dataset...")
    # This evaluates on 5 different test splits and aggregates them,
    # ensuring every single piece in the 144 dataset is tested exactly once as an "unseen" holdout.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_baseline_pred = cross_val_predict(baseline_rf, X, y, cv=cv, n_jobs=-1)
    cv_best_pred = cross_val_predict(best_rf, X, y, cv=cv, n_jobs=-1)

    baseline_acc = accuracy_score(y, cv_baseline_pred)
    best_acc = accuracy_score(y, cv_best_pred)

    _print_comparison_table(baseline_acc, best_acc, baseline_metrics, best_metrics)

    print("\nClassification Report (Aggregated 5-Fold CV):")
    cr_text = classification_report(y, cv_best_pred, target_names=label_encoder.classes_)
    print(cr_text)

    print("Confusion Matrix (Aggregated 5-Fold CV):")
    cm = confusion_matrix(y, cv_best_pred)
    print(cm)
    
    print("Generating color-coded Confusion Matrix plot...")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Random Forest Confusion Matrix (5-Fold CV)")
    cm_path = DEFAULT_FIG_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion Matrix plot saved to {cm_path}")

    print("Generating Precision vs Recall scatter plot...")
    cr_dict = classification_report(y, cv_best_pred, target_names=label_encoder.classes_, output_dict=True)
    
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Distinct colors for the 4 composers
    for idx, cls in enumerate(label_encoder.classes_):
        if cls in cr_dict:
            plt.scatter(cr_dict[cls]['recall'], cr_dict[cls]['precision'], 
                        label=cls, s=150, color=colors[idx], edgecolor='black', zorder=5)
            plt.annotate(cls, (cr_dict[cls]['recall'], cr_dict[cls]['precision']), 
                         xytext=(10, 5), textcoords='offset points', fontweight='bold')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, zorder=1) # Diagonal line
    plt.xlabel('Recall (How many actual pieces were found?)')
    plt.ylabel('Precision (How many predicted pieces were correct?)')
    plt.title('Classification Performance by Composer (5-Fold CV)')
    plt.xlim(0.3, 1.05)
    plt.ylim(0.3, 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower left')
    
    scatter_path = DEFAULT_FIG_DIR / "classification_scatter.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Classification scatter plot saved to {scatter_path}")

    print("Finding the single most accurate tree in the forest for visualization...")
    # Evaluate all trees in the forest on the test set to find the best representative tree
    tree_accuracies = []
    for tree in best_rf.estimators_:
        tree_pred = tree.predict(X_test.values if hasattr(X_test, "values") else X_test)
        tree_accuracies.append(accuracy_score(y_test, tree_pred))
    
    best_tree_idx = np.argmax(tree_accuracies)
    best_single_tree = best_rf.estimators_[best_tree_idx]
    print(f"  -> Selected Tree #{best_tree_idx} (Single-tree accuracy on holdout: {tree_accuracies[best_tree_idx]:.4f})")

    print("Exporting best single tree visualizations...")
    first_tree = best_single_tree
    tree_png_path = DEFAULT_FIG_DIR / "rf_sample_tree.png"
    tree_txt_path = DEFAULT_FIG_DIR / "rf_sample_tree.txt"

    plt.figure(figsize=(20, 15))
    plot_tree(
        first_tree,
        feature_names=feature_cols,
        class_names=label_encoder.classes_,
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(tree_png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Tree visualization (PNG) saved to {tree_png_path}")

    tree_text = export_text(first_tree, feature_names=feature_cols)
    with open(tree_txt_path, "w") as f:
        f.write(tree_text)
    print(f"Tree text export saved to {tree_txt_path}")

    print("Generating SHAP values and summary plot...")
    explainer = shap.TreeExplainer(best_rf)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_cols,
        class_names=label_encoder.classes_,
        show=False,
    )

    shap_fig_path = DEFAULT_FIG_DIR / "rf_shap_summary.png"
    plt.tight_layout()
    plt.savefig(shap_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {shap_fig_path}")

    model_path = DEFAULT_MODEL_DIR / "random_forest_composer.pkl"
    print(f"Saving model and artifacts to {model_path}...")

    artifacts = {
        "model": best_rf,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }
    joblib.dump(artifacts, model_path)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
