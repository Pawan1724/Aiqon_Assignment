"""
evaluation.py  –  Model evaluation metrics, plots, and reports.
"""
from __future__ import annotations

import os
import json
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src import CONFIG, logger


# ------------------------------------------------------------------ #
#  Regression evaluation                                              #
# ------------------------------------------------------------------ #

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> dict[str, float]:
    """Compute MAE, RMSE, R² for a regression model."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    n = len(y_true)

    metrics = {
        "model": model_name,
        "n_samples": n,
        "mae_minutes": round(mae, 2),
        "rmse_minutes": round(rmse, 2),
        "r_squared": round(r2, 4),
    }
    logger.info(
        "%s  MAE=%.2f min  RMSE=%.2f min  R²=%.4f  (n=%d)",
        model_name, mae, rmse, r2, n,
    )
    return metrics


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Delay",
    save_path: str | None = None,
) -> str | None:
    """Scatter plot of actual vs predicted."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="white", s=50, color="#4C72B0")
    lims = [
        min(min(y_true), min(y_pred)) - 10,
        max(max(y_true), max(y_pred)) + 10,
    ]
    ax.plot(lims, lims, "r--", alpha=0.7, label="Perfect prediction")
    ax.set_xlabel("Actual Delay (minutes)")
    ax.set_ylabel("Predicted Delay (minutes)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    plt.close(fig)
    return None


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Error Distribution",
    save_path: str | None = None,
) -> str | None:
    """Histogram of residuals."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=25, color="#55A868", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Residual (Actual − Predicted, minutes)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    plt.close(fig)
    return None


# ------------------------------------------------------------------ #
#  Classification evaluation                                          #
# ------------------------------------------------------------------ #

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    model_name: str = "",
) -> dict[str, Any]:
    """Compute precision, recall, F1, accuracy, ROC-AUC."""
    metrics: dict[str, Any] = {
        "model": model_name,
        "n_samples": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)

    logger.info(
        "%s  Acc=%.3f  P=%.3f  R=%.3f  F1=%.3f",
        model_name,
        metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"],
    )
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> str | None:
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = labels or ["On-Time", "Delayed"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    plt.close(fig)
    return None


# ------------------------------------------------------------------ #
#  Report writer                                                       #
# ------------------------------------------------------------------ #

def write_evaluation_report(
    reg_metrics: dict,
    baseline_metrics: dict,
    clf_metrics: dict | None = None,
    save_dir: str | None = None,
) -> str:
    """Write a markdown evaluation report."""
    save_dir = save_dir or CONFIG["paths"]["reports_dir"]
    if not os.path.isabs(save_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, "model_evaluation.md")
    lines = [
        "# Model Evaluation Report",
        "",
        "## ETA Regression",
        "",
        "| Metric | Baseline (Mean) | Best Model |",
        "|--------|----------------|------------|",
        f"| MAE (min) | {baseline_metrics.get('mae_minutes', 'N/A')} | {reg_metrics.get('mae_minutes', 'N/A')} |",
        f"| RMSE (min) | {baseline_metrics.get('rmse_minutes', 'N/A')} | {reg_metrics.get('rmse_minutes', 'N/A')} |",
        f"| R² | {baseline_metrics.get('r_squared', 'N/A')} | {reg_metrics.get('r_squared', 'N/A')} |",
        f"| Test samples | {baseline_metrics.get('n_samples', 'N/A')} | {reg_metrics.get('n_samples', 'N/A')} |",
        "",
    ]

    if clf_metrics:
        lines.extend([
            "## Delay Classification",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Accuracy | {clf_metrics.get('accuracy', 'N/A')} |",
            f"| Precision | {clf_metrics.get('precision', 'N/A')} |",
            f"| Recall | {clf_metrics.get('recall', 'N/A')} |",
            f"| F1 | {clf_metrics.get('f1', 'N/A')} |",
            f"| ROC-AUC | {clf_metrics.get('roc_auc', 'N/A')} |",
            f"| Test samples | {clf_metrics.get('n_samples', 'N/A')} |",
            "",
        ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also save as JSON
    json_path = os.path.join(save_dir, "metrics.json")
    all_metrics = {
        "regression_baseline": baseline_metrics,
        "regression_best": reg_metrics,
    }
    if clf_metrics:
        all_metrics["classification"] = clf_metrics
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    logger.info("Evaluation report written to %s", path)
    return path
