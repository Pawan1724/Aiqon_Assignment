"""
eta_model.py  –  ETA / delay-magnitude regression pipeline.

Models
------
1. BaselineMean    – predicts training-set mean delay for every shipment.
2. Ridge           – linear baseline.
3. RandomForest    – non-linear candidate.

Target: delay_target_minutes  (actual_arrival − scheduled_arrival, in minutes).
Predicted ETA = scheduled_arrival + predicted_delay.
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

from src import CONFIG, logger
from src.feature_engineering import build_preprocessor


# ------------------------------------------------------------------ #
#  Baseline estimator                                                 #
# ------------------------------------------------------------------ #
class BaselineMeanRegressor(BaseEstimator, RegressorMixin):
    """Predict the training-set mean delay for every sample."""

    def __init__(self):
        self.mean_: float = 0.0

    def fit(self, X, y):
        self.mean_ = float(np.nanmean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


# ------------------------------------------------------------------ #
#  Training & selection                                               #
# ------------------------------------------------------------------ #

def train_eta_models(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    preprocessor,
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Train baseline, Ridge, and RandomForest regressors.

    Returns dict with keys:
        models        – {name: fitted Pipeline}
        cv_results    – {name: {mae_mean, mae_std}}
        best_name     – name of best model by CV MAE
    """
    seed = seed or CONFIG["project"]["random_seed"]
    cv_folds = CONFIG["data"]["cv_folds"]

    models: dict[str, Pipeline] = {}
    cv_results: dict[str, dict] = {}

    # ---- 1. Baseline ----
    baseline_pipe = Pipeline([
        ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
        ("model", BaselineMeanRegressor()),
    ])
    baseline_pipe.fit(X_train, y_train)
    scores = cross_val_score(
        baseline_pipe, X_train, y_train,
        cv=cv_folds, scoring="neg_mean_absolute_error",
    )
    models["baseline_mean"] = baseline_pipe
    cv_results["baseline_mean"] = {
        "mae_mean": float(-scores.mean()),
        "mae_std": float(scores.std()),
    }
    logger.info("Baseline MAE: %.2f ± %.2f", -scores.mean(), scores.std())

    # ---- 2. Ridge ----
    ridge_pipe = Pipeline([
        ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
        ("model", Ridge(alpha=1.0, random_state=seed)),
    ])
    # Quick alpha search
    alphas = CONFIG["model"]["eta"]["ridge_alphas"]
    best_ridge_score = -np.inf
    best_alpha = alphas[0]
    for alpha in alphas:
        ridge_pipe.set_params(model__alpha=alpha)
        sc = cross_val_score(
            ridge_pipe, X_train, y_train,
            cv=cv_folds, scoring="neg_mean_absolute_error",
        )
        if sc.mean() > best_ridge_score:
            best_ridge_score = sc.mean()
            best_alpha = alpha
    ridge_pipe.set_params(model__alpha=best_alpha)
    ridge_pipe.fit(X_train, y_train)
    scores = cross_val_score(
        ridge_pipe, X_train, y_train,
        cv=cv_folds, scoring="neg_mean_absolute_error",
    )
    models["ridge"] = ridge_pipe
    cv_results["ridge"] = {
        "mae_mean": float(-scores.mean()),
        "mae_std": float(scores.std()),
        "best_alpha": best_alpha,
    }
    logger.info("Ridge MAE: %.2f ± %.2f (alpha=%s)", -scores.mean(), scores.std(), best_alpha)

    # ---- 3. Random Forest ----
    rf_pipe = Pipeline([
        ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
        ("model", RandomForestRegressor(random_state=seed)),
    ])
    param_grid = {
        f"model__{k}": v
        for k, v in CONFIG["model"]["eta"]["rf_param_grid"].items()
    }
    gs = GridSearchCV(
        rf_pipe, param_grid,
        cv=cv_folds,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    models["random_forest"] = gs.best_estimator_
    cv_results["random_forest"] = {
        "mae_mean": float(-gs.best_score_),
        "mae_std": float(gs.cv_results_["std_test_score"][gs.best_index_]),
        "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
    }
    logger.info("RF MAE: %.2f ± %.2f", -gs.best_score_,
                gs.cv_results_["std_test_score"][gs.best_index_])

    # ---- Select best ----
    best_name = min(cv_results, key=lambda k: cv_results[k]["mae_mean"])
    logger.info("Best ETA model by CV MAE: %s (%.2f)", best_name, cv_results[best_name]["mae_mean"])

    return {
        "models": models,
        "cv_results": cv_results,
        "best_name": best_name,
    }


# ------------------------------------------------------------------ #
#  Persistence                                                        #
# ------------------------------------------------------------------ #

def save_model(
    pipeline: Pipeline,
    model_name: str,
    cv_results: dict,
    feature_cols: list[str],
    model_dir: str | None = None,
) -> str:
    """Save a fitted pipeline + metadata to disk."""
    model_dir = model_dir or CONFIG["paths"]["models_dir"]
    if not os.path.isabs(model_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, model_dir)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "eta_model.joblib")
    joblib.dump(pipeline, model_path)

    meta = {
        "model_name": model_name,
        "saved_at": datetime.utcnow().isoformat(),
        "cv_results": cv_results,
        "feature_columns": feature_cols,
        "target": "delay_target_minutes",
        "prediction_scenario": "pre-departure",
    }
    meta_path = os.path.join(model_dir, "eta_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info("ETA model saved to %s", model_path)
    return model_path


def load_model(model_dir: str | None = None) -> tuple[Pipeline, dict]:
    """Load a saved pipeline and its metadata."""
    model_dir = model_dir or CONFIG["paths"]["models_dir"]
    if not os.path.isabs(model_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, model_dir)

    model_path = os.path.join(model_dir, "eta_model.joblib")
    meta_path = os.path.join(model_dir, "eta_model_meta.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved ETA model found at {model_path}")

    pipeline = joblib.load(model_path)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return pipeline, meta
