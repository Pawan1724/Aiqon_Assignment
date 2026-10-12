"""Tests for data_validation.py"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest
from src.data_validation import (
    check_schema, check_duplicates, check_missing_targets,
    check_timestamp_order, check_negative_delays,
)


@pytest.fixture
def sample_df():
    """Create a minimal sample DataFrame matching the expected schema."""
    return pd.DataFrame({
        "shipment_id": ["S001", "S002", "S003", "S001"],
        "scheduled_departure_datetime": [
            "2024-06-01T07:00:00Z", "2024-06-02T08:00:00Z",
            "2024-06-03T09:00:00Z", "2024-06-01T07:00:00Z",
        ],
        "actual_departure_datetime": [
            "2024-06-01T07:05:00Z", "2024-06-02T08:10:00Z",
            np.nan, "2024-06-01T07:05:00Z",
        ],
        "scheduled_arrival_datetime": [
            "2024-06-01T11:00:00Z", "2024-06-02T12:00:00Z",
            "2024-06-03T15:00:00Z", "2024-06-01T11:00:00Z",
        ],
        "actual_arrival_datetime": [
            "2024-06-01T11:30:00Z", "2024-06-02T11:50:00Z",
            np.nan, "2024-06-01T11:30:00Z",
        ],
        "origin_city": ["Sydney", "Melbourne", "Perth", "Sydney"],
        "origin_country": ["Australia", "Australia", "Australia", "Australia"],
        "destination_city": ["Brisbane", "Sydney", "Melbourne", "Brisbane"],
        "destination_country": ["Australia", "Australia", "Australia", "Australia"],
        "carrier_id": ["CR001", "CR002", "CR003", "CR001"],
        "route_distance_km": [900, 880, 3300, 900],
        "route_type": ["highway", "highway", "air", "highway"],
        "vehicle_type": ["truck", "truck", "airplane", "truck"],
        "weather_condition_departure": ["clear", "rain", "clear", "clear"],
        "delay_predicted_minutes": [10, 15, 5, 10],
        "delay_actual_minutes": [30, -10, np.nan, 30],
        "carrier_on_time_rate": [0.92, 0.85, 0.93, 0.92],
        "shipment_status": ["delivered", "delivered", "in_transit", "delivered"],
        "priority_level": ["standard", "critical", "standard", "standard"],
        "shipment_type": ["general", "hazardous", "perishable", "general"],
    })


def test_check_schema_complete(sample_df):
    """No missing columns when all expected columns are present."""
    missing = check_schema(sample_df)
    assert len(missing) == 0


def test_check_schema_missing_column(sample_df):
    """Detect missing columns."""
    df = sample_df.drop(columns=["route_distance_km"])
    missing = check_schema(df)
    assert "route_distance_km" in missing


def test_check_duplicates(sample_df):
    """Detect duplicated shipment IDs."""
    dups = check_duplicates(sample_df)
    assert len(dups) == 2  # S001 appears twice
    assert set(dups["shipment_id"].unique()) == {"S001"}


def test_check_missing_targets(sample_df):
    """Count missing target values."""
    result = check_missing_targets(sample_df)
    assert result["actual_arrival_datetime"] == 1
    assert result["delay_actual_minutes"] == 1


def test_check_negative_delays(sample_df):
    """Detect negative delays."""
    neg = check_negative_delays(sample_df)
    assert len(neg) == 1  # S002 has -10


def test_check_timestamp_order_valid():
    """Valid timestamps should produce no flags."""
    df = pd.DataFrame({
        "shipment_id": ["S001"],
        "scheduled_departure_datetime": ["2024-06-01T07:00:00Z"],
        "actual_departure_datetime": ["2024-06-01T07:05:00Z"],
        "scheduled_arrival_datetime": ["2024-06-01T11:00:00Z"],
        "actual_arrival_datetime": ["2024-06-01T11:30:00Z"],
        "delay_actual_minutes": [30.0],
    })
    flags = check_timestamp_order(df)
    assert len(flags) == 0


def test_check_timestamp_order_bad_arrival():
    """actual_arrival < actual_departure should be flagged."""
    df = pd.DataFrame({
        "shipment_id": ["S001"],
        "scheduled_departure_datetime": ["2024-06-01T07:00:00Z"],
        "actual_departure_datetime": ["2024-06-01T11:00:00Z"],
        "scheduled_arrival_datetime": ["2024-06-01T12:00:00Z"],
        "actual_arrival_datetime": ["2024-06-01T10:00:00Z"],
        "delay_actual_minutes": [-120.0],
    })
    flags = check_timestamp_order(df)
    flag_types = flags["flag"].tolist()
    assert any("actual_arrival < actual_departure" in f for f in flag_types)
