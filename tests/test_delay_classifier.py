"""Tests for delay classifier."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from src.delay_classifier import train_classifiers

@pytest.fixture
def synthetic_clf_data():
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
    y = np.random.randint(0, 2, n)
    num_cols = ["route_distance_km", "carrier_on_time_rate", "scheduled_duration_min",
                "departure_delay_min", "hour_of_departure", "day_of_week", "is_international"]
    cat_cols = ["route_type", "vehicle_type", "weather_condition_departure",
                "priority_level", "shipment_type"]
    return X, y, num_cols, cat_cols

def test_train_classifiers_returns_all(synthetic_clf_data):
    X, y, num_cols, cat_cols = synthetic_clf_data
    result = train_classifiers(X, y, num_cols, cat_cols, seed=42)

    assert "models" in result
    assert "cv_results" in result
    assert "best_name" in result
    
    assert "logistic_regression" in result["models"]
    assert "random_forest" in result["models"]

def test_clf_model_predict(synthetic_clf_data):
    X, y, num_cols, cat_cols = synthetic_clf_data
    result = train_classifiers(X, y, num_cols, cat_cols, seed=42)
    best = result["models"][result["best_name"]]
    
    preds = best.predict(X.iloc[:5])
    assert len(preds) == 5
    assert all(p in [0, 1] for p in preds)
    
    if hasattr(best, "predict_proba"):
        probs = best.predict_proba(X.iloc[:5])
        assert probs.shape == (5, 2)
