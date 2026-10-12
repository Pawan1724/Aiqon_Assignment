"""Tests for ETA model training and prediction."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.eta_model import BaselineMeanRegressor, train_eta_models
from src.feature_engineering import build_preprocessor


@pytest.fixture
def synthetic_data():
    """Create a small synthetic feature matrix and target."""
    np.random.seed(42)
    n = 50
    X = pd.DataFrame({
        "route_distance_km": np.random.uniform(100, 5000, n),
        "carrier_on_time_rate": np.random.uniform(0.6, 1.0, n),
        "scheduled_duration_min": np.random.uniform(60, 600, n),
        "departure_delay_min": np.random.uniform(0, 30, n),
        "hour_of_departure": np.random.randint(0, 24, n),
        "day_of_week": np.random.randint(0, 7, n),
        "is_international": np.random.randint(0, 2, n),
        "route_type": np.random.choice(["highway", "sea", "air", "rail"], n),
        "vehicle_type": np.random.choice(["truck", "ship", "airplane", "train"], n),
        "weather_condition_departure": np.random.choice(["clear", "rain", "fog", "storm"], n),
        "priority_level": np.random.choice(["standard", "critical", "express"], n),
        "shipment_type": np.random.choice(["general", "hazardous", "perishable"], n),
    })
    y = np.random.uniform(0, 200, n)
    num_cols = ["route_distance_km", "carrier_on_time_rate", "scheduled_duration_min",
                "departure_delay_min", "hour_of_departure", "day_of_week", "is_international"]
    cat_cols = ["route_type", "vehicle_type", "weather_condition_departure",
                "priority_level", "shipment_type"]
    return X, y, num_cols, cat_cols


def test_baseline_mean_regressor():
    """Baseline predicts training mean."""
    model = BaselineMeanRegressor()
    y = np.array([10, 20, 30, 40, 50])
    model.fit(None, y)
    preds = model.predict(np.zeros((3, 1)))
    assert all(p == 30.0 for p in preds)


def test_train_eta_models_returns_all(synthetic_data):
    """Training returns models, cv_results, and best_name."""
    X, y, num_cols, cat_cols = synthetic_data
    result = train_eta_models(X, y, None, num_cols, cat_cols, seed=42)

    assert "models" in result
    assert "cv_results" in result
    assert "best_name" in result
    assert result["best_name"] in result["models"]

    # All three models should be trained
    assert "baseline_mean" in result["models"]
    assert "ridge" in result["models"]
    assert "random_forest" in result["models"]


def test_eta_model_predict(synthetic_data):
    """Trained model can predict."""
    X, y, num_cols, cat_cols = synthetic_data
    result = train_eta_models(X, y, None, num_cols, cat_cols, seed=42)
    best = result["models"][result["best_name"]]
    preds = best.predict(X.iloc[:5])
    assert len(preds) == 5
    assert all(np.isfinite(preds))


def test_no_leakage_in_features(synthetic_data):
    """Features should not contain target-related columns."""
    from src.feature_engineering import _LEAKAGE_COLS
    X, _, _, _ = synthetic_data
    for col in X.columns:
        assert col not in _LEAKAGE_COLS, f"Leakage column '{col}' found in features"
