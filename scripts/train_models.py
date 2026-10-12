"""
scripts/train_models.py  –  Full training pipeline.

Usage:
    python -m scripts.train_models
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import CONFIG, logger
from src.data_loader import load_raw_data
from src.preprocessing import preprocess, save_processed
from src.feature_engineering import (
    build_feature_matrix, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from src.eta_model import train_eta_models, save_model
from src.delay_classifier import train_classifiers, save_classifier


def main():
    seed = CONFIG["project"]["random_seed"]
    test_size = CONFIG["data"]["test_size"]

    # ---- 1. Load & preprocess ----
    logger.info("=== Phase 1: Load & Preprocess ===")
    raw = load_raw_data()
    df = preprocess(raw)
    save_processed(df)

    # ---- 2. Filter to records with valid outcomes ----
    df_train_pool = df[df["has_outcome"] == True].copy()
    logger.info("Records with valid outcome: %d", len(df_train_pool))

    # ---- 3. Build feature matrix ----
    X_all, num_cols, cat_cols = build_feature_matrix(df_train_pool)
    y_reg = df_train_pool["delay_target_minutes"].values
    y_clf = df_train_pool["is_delayed"].astype(int).values

    # ---- 4. Train/test split ----
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = (
        train_test_split(
            X_all, y_reg, y_clf,
            test_size=test_size,
            random_state=seed,
            stratify=y_clf,
        )
    )
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # ---- 5. Train ETA regression models ----
    logger.info("=== Phase 2: ETA Regression ===")
    eta_result = train_eta_models(
        X_train, y_reg_train,
        preprocessor=None,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        seed=seed,
    )

    best_name = eta_result["best_name"]
    best_model = eta_result["models"][best_name]

    # Save best model
    feature_cols = num_cols + cat_cols
    save_model(
        best_model, best_name,
        eta_result["cv_results"][best_name],
        feature_cols,
    )

    # ---- 6. Train delay classifiers ----
    logger.info("=== Phase 3: Delay Classification ===")
    clf_result = train_classifiers(
        X_train, y_clf_train,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        seed=seed,
    )

    best_clf_name = clf_result["best_name"]
    best_clf = clf_result["models"][best_clf_name]

    save_classifier(
        best_clf, best_clf_name,
        clf_result["cv_results"][best_clf_name],
        feature_cols,
    )

    # ---- 7. Save split indices for evaluation ----
    split_info = {
        "train_indices": X_train.index.tolist(),
        "test_indices": X_test.index.tolist(),
    }
    import json
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split_path = os.path.join(project_root, CONFIG["paths"]["models_dir"], "split_info.json")
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split_info, f)

    logger.info("=== Training Complete ===")
    print(f"\nETA CV results: {eta_result['cv_results']}")
    print(f"Best ETA model: {best_name}")
    print(f"\nClassifier CV results: {clf_result['cv_results']}")
    print(f"Best classifier: {best_clf_name}")

    return eta_result, clf_result


if __name__ == "__main__":
    main()
