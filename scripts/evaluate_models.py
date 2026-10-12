"""
scripts/evaluate_models.py  –  Evaluate saved models on the held-out test set.

Usage:
    python -m scripts.evaluate_models
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import CONFIG, logger
from src.data_loader import load_raw_data
from src.preprocessing import preprocess
from src.feature_engineering import build_feature_matrix, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.eta_model import load_model as load_eta
from src.delay_classifier import load_classifier
from src.evaluation import (
    evaluate_regression, evaluate_classification,
    plot_actual_vs_predicted, plot_residuals, plot_confusion_matrix,
    write_evaluation_report,
)
from src.explainability import (
    compute_permutation_importance, plot_feature_importance,
    compute_shap_values, plot_shap_summary,
)
from src.eta_model import BaselineMeanRegressor


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, CONFIG["paths"]["models_dir"])
    reports_dir = os.path.join(project_root, CONFIG["paths"]["reports_dir"])
    figures_dir = os.path.join(project_root, CONFIG["paths"]["figures_dir"])
    os.makedirs(figures_dir, exist_ok=True)

    # ---- 1. Load data and reconstruct split ----
    raw = load_raw_data()
    df = preprocess(raw)
    df_with_outcome = df[df["has_outcome"] == True].copy()

    split_path = os.path.join(models_dir, "split_info.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError("Run train_models.py first to generate split info.")

    with open(split_path, "r") as f:
        split_info = json.load(f)

    test_indices = split_info["test_indices"]
    train_indices = split_info["train_indices"]

    X_all, num_cols, cat_cols = build_feature_matrix(df_with_outcome)
    y_reg = df_with_outcome["delay_target_minutes"]
    y_clf = df_with_outcome["is_delayed"].astype(int)

    X_test = X_all.loc[X_all.index.isin(test_indices)]
    y_reg_test = y_reg.loc[y_reg.index.isin(test_indices)]
    y_clf_test = y_clf.loc[y_clf.index.isin(test_indices)]
    X_train = X_all.loc[X_all.index.isin(train_indices)]
    y_reg_train = y_reg.loc[y_reg.index.isin(train_indices)]

    logger.info("Test set: %d samples", len(X_test))

    # ---- 2. ETA regression evaluation ----
    eta_model, eta_meta = load_eta(models_dir)
    y_pred_reg = eta_model.predict(X_test)

    reg_metrics = evaluate_regression(
        y_reg_test.values, y_pred_reg,
        model_name=eta_meta.get("model_name", "best"),
    )

    # Baseline comparison
    baseline_pred = np.full(len(y_reg_test), y_reg_train.mean())
    baseline_metrics = evaluate_regression(
        y_reg_test.values, baseline_pred, model_name="baseline_mean",
    )

    # Plots
    plot_actual_vs_predicted(
        y_reg_test.values, y_pred_reg,
        title=f"Actual vs Predicted Delay ({eta_meta.get('model_name', '')})",
        save_path=os.path.join(figures_dir, "actual_vs_predicted.png"),
    )
    plot_residuals(
        y_reg_test.values, y_pred_reg,
        save_path=os.path.join(figures_dir, "residuals.png"),
    )

    # ---- 3. Feature importance ----
    # Get preprocessor output feature names
    preprocessor = eta_model.named_steps.get("preprocessor")
    if preprocessor is not None:
        X_test_transformed = preprocessor.transform(X_test)
        try:
            feat_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            feat_names = [f"f_{i}" for i in range(X_test_transformed.shape[1])]
    else:
        X_test_transformed = X_test
        feat_names = list(X_test.columns)

    perm_imp = compute_permutation_importance(
        eta_model, X_test, y_reg_test.values,
        feature_names=num_cols + cat_cols,
    )
    plot_feature_importance(
        perm_imp,
        title="Permutation Feature Importance (ETA Model)",
        save_path=os.path.join(figures_dir, "feature_importance.png"),
    )

    # SHAP
    shap_res = compute_shap_values(
        eta_model, X_test, feature_names=feat_names,
    )
    if shap_res:
        plot_shap_summary(
            shap_res,
            save_path=os.path.join(figures_dir, "shap_summary.png"),
        )

    # ---- 4. Classification evaluation ----
    clf_metrics = None
    try:
        clf_model, clf_meta = load_classifier(models_dir)
        y_pred_clf = clf_model.predict(X_test)
        y_prob_clf = None
        if hasattr(clf_model, "predict_proba"):
            y_prob_clf = clf_model.predict_proba(X_test)[:, 1]

        clf_metrics = evaluate_classification(
            y_clf_test.values, y_pred_clf, y_prob_clf,
            model_name=clf_meta.get("model_name", "best"),
        )

        plot_confusion_matrix(
            y_clf_test.values, y_pred_clf,
            save_path=os.path.join(figures_dir, "confusion_matrix.png"),
        )
    except FileNotFoundError:
        logger.warning("No classifier found, skipping classification evaluation.")

    # ---- 5. Write report ----
    write_evaluation_report(reg_metrics, baseline_metrics, clf_metrics, reports_dir)

    print("\n=== Evaluation Complete ===")
    print(f"Regression: {reg_metrics}")
    print(f"Baseline:   {baseline_metrics}")
    if clf_metrics:
        print(f"Classifier: {clf_metrics}")
    print(f"\nFigures saved to: {figures_dir}")
    print(f"Report saved to: {reports_dir}/model_evaluation.md")

    return reg_metrics, baseline_metrics, clf_metrics


if __name__ == "__main__":
    main()
