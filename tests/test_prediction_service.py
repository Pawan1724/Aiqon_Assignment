"""Tests for the prediction service."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.prediction_service import PredictionService

@pytest.fixture
def service():
    # Will warn about missing models if they haven't been trained yet
    return PredictionService()

def test_prediction_service_predict_single_no_models(service):
    """Test that the service degrades gracefully when no models are loaded."""
    # Force models to None
    service.eta_model = None
    service.clf_model = None
    
    shipment = {
        "scheduled_arrival_datetime": "2024-06-01T12:00:00Z",
        "scheduled_departure_datetime": "2024-06-01T08:00:00Z",
        "origin_country": "USA",
        "destination_country": "USA",
        "route_distance_km": 500.0,
        "carrier_on_time_rate": 0.9,
    }
    
    res = service.predict_single(shipment)
    
    assert res["predicted_delay_min"] is None
    assert res["predicted_eta"] is None
    assert res["delay_risk"] is None
    assert res["propagation"] is not None  # Propagation is rule-based, should still run
    assert res["propagation"]["is_simulated"] is True

def test_prediction_service_what_if(service):
    """Test the what-if simulation wrapper."""
    shipment = {
        "scheduled_duration_min": 200.0,
        "departure_delay_min": 10.0,
    }
    
    res = service.run_what_if(shipment, "transit", 30.0)
    
    assert "baseline_total_delay_min" in res
    assert "injected_total_delay_min" in res
    assert res["injected_total_delay_min"] > res["baseline_total_delay_min"]
    assert res["is_simulated"] is True
