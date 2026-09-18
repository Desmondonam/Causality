"""Matplotlib figure generation for the pipeline report."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ml.modeling import ModelResult, best_model_name, roc_curve_points


def plot_model_comparison(results: dict[str, ModelResult], y_test: pd.Series, path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    names = list(results.keys())
    accuracies = [results[n].accuracy for n in names]
    axes[0, 0].bar(names, accuracies, color="#3b82f6")
    axes[0, 0].set_title("Model Accuracy Comparison", fontweight="bold")
    axes[0, 0].set_ylim([min(accuracies) - 0.02, 1.0])
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontweight="bold")

    roc_aucs = [results[n].roc_auc for n in names]
    axes[0, 1].bar(names, roc_aucs, color="#10b981")
    axes[0, 1].set_title("Model ROC-AUC Comparison", fontweight="bold")
    axes[0, 1].set_ylim([min(roc_aucs) - 0.02, 1.0])
    for i, v in enumerate(roc_aucs):
        axes[0, 1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontweight="bold")

    for name, (fpr, tpr) in roc_curve_points(y_test, results).items():
        axes[1, 0].plot(fpr, tpr, label=f"{name} (AUC={results[name].roc_auc:.4f})")
    axes[1, 0].plot([0, 1], [0, 1], "k--", label="Random", alpha=0.5)
    axes[1, 0].set_xlabel("False Positive Rate")
    axes[1, 0].set_ylabel("True Positive Rate")
    axes[1, 0].set_title("ROC Curves", fontweight="bold")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    best = best_model_name(results)
    cm = confusion_matrix(y_test, results[best].y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 1])
    axes[1, 1].set_title(f"Confusion Matrix - {best}", fontweight="bold")
    axes[1, 1].set_ylabel("True Label")
    axes[1, 1].set_xlabel("Predicted Label")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(causal_scores: pd.DataFrame, shap_importance: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    top_causal = causal_scores.head(10)
    axes[0].barh(range(len(top_causal)), top_causal["Causal_Score"], color="#8b5cf6")
    axes[0].set_yticks(range(len(top_causal)))
    axes[0].set_yticklabels(top_causal.index)
    axes[0].set_xlabel("Composite Causal Score")
    axes[0].set_title("Top 10 Features by Statistical Causal Score", fontweight="bold")
    axes[0].invert_yaxis()

    top_shap = shap_importance.head(10)
    axes[1].barh(range(len(top_shap)), top_shap["SHAP_Importance"], color="#ef4444")
    axes[1].set_yticks(range(len(top_shap)))
    axes[1].set_yticklabels(top_shap["Feature"])
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].set_title("Top 10 Features by SHAP Importance", fontweight="bold")
    axes[1].invert_yaxis()

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
