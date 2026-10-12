"""
delay_classifier.py  –  Binary delay-risk classification.

Positive class: is_delayed == 1 (delay_target_minutes > threshold).
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src import CONFIG, logger
from src.feature_engineering import build_preprocessor


def train_classifiers(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Train Logistic Regression and Random Forest classifiers.

    Returns dict:
        models      – {name: fitted Pipeline}
        cv_results  – {name: {f1_mean, f1_std, roc_auc_mean, ...}}
        best_name   – name by CV F1
    """
    seed = seed or CONFIG["project"]["random_seed"]
    cv_folds = CONFIG["data"]["cv_folds"]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    models: dict[str, Pipeline] = {}
    cv_results: dict[str, dict] = {}

    # ---- 1. Logistic Regression ----
    lr_pipe = Pipeline([
        ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
        ("model", LogisticRegression(
            max_iter=1000, random_state=seed, class_weight="balanced",
        )),
    ])
    lr_pipe.fit(X_train, y_train)

    f1_scores = cross_val_score(lr_pipe, X_train, y_train, cv=skf, scoring="f1")
    roc_scores = cross_val_score(lr_pipe, X_train, y_train, cv=skf, scoring="roc_auc")

    models["logistic_regression"] = lr_pipe
    cv_results["logistic_regression"] = {
        "f1_mean": float(f1_scores.mean()),
        "f1_std": float(f1_scores.std()),
        "roc_auc_mean": float(roc_scores.mean()),
        "roc_auc_std": float(roc_scores.std()),
    }
    logger.info("LR F1: %.3f ± %.3f  |  ROC-AUC: %.3f",
                f1_scores.mean(), f1_scores.std(), roc_scores.mean())

    # ---- 2. Random Forest ----
    rf_pipe = Pipeline([
        ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
        ("model", RandomForestClassifier(
            random_state=seed, class_weight="balanced",
        )),
    ])
    param_grid = {
        f"model__{k}": v
        for k, v in CONFIG["model"]["classifier"]["rf_param_grid"].items()
    }
    gs = GridSearchCV(
        rf_pipe, param_grid,
        cv=skf, scoring="f1", n_jobs=-1, refit=True,
    )
    gs.fit(X_train, y_train)
    models["random_forest"] = gs.best_estimator_

    # Re-score with the best estimator for roc_auc
    roc_scores_rf = cross_val_score(
        gs.best_estimator_, X_train, y_train, cv=skf, scoring="roc_auc",
    )

    cv_results["random_forest"] = {
        "f1_mean": float(-gs.best_score_) if gs.best_score_ < 0 else float(gs.best_score_),
        "f1_std": float(gs.cv_results_["std_test_score"][gs.best_index_]),
        "roc_auc_mean": float(roc_scores_rf.mean()),
        "roc_auc_std": float(roc_scores_rf.std()),
        "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
    }
    logger.info("RF-clf F1: %.3f  |  ROC-AUC: %.3f",
                gs.best_score_, roc_scores_rf.mean())

    best_name = max(cv_results, key=lambda k: cv_results[k]["f1_mean"])
    logger.info("Best classifier: %s (F1=%.3f)", best_name, cv_results[best_name]["f1_mean"])

    return {
        "models": models,
        "cv_results": cv_results,
        "best_name": best_name,
    }


# ------------------------------------------------------------------ persistence
def save_classifier(
    pipeline: Pipeline,
    model_name: str,
    cv_results: dict,
    feature_cols: list[str],
    model_dir: str | None = None,
) -> str:
    model_dir = model_dir or CONFIG["paths"]["models_dir"]
    if not os.path.isabs(model_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, model_dir)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "delay_classifier.joblib")
    joblib.dump(pipeline, model_path)

    meta = {
        "model_name": model_name,
        "saved_at": datetime.utcnow().isoformat(),
        "cv_results": cv_results,
        "feature_columns": feature_cols,
        "target": "is_delayed",
        "delay_threshold_minutes": CONFIG["data"]["delay_threshold_minutes"],
    }
    meta_path = os.path.join(model_dir, "delay_classifier_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info("Classifier saved to %s", model_path)
    return model_path


def load_classifier(model_dir: str | None = None) -> tuple[Pipeline, dict]:
    model_dir = model_dir or CONFIG["paths"]["models_dir"]
    if not os.path.isabs(model_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, model_dir)

    model_path = os.path.join(model_dir, "delay_classifier.joblib")
    meta_path = os.path.join(model_dir, "delay_classifier_meta.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved classifier at {model_path}")

    pipeline = joblib.load(model_path)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return pipeline, meta
