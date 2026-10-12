"""Tests for preprocessing.py"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest
from src.preprocessing import preprocess


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "shipment_id": ["S001", "S002", "S003"],
        "scheduled_departure_datetime": [
            "2024-06-01T07:00:00Z", "2024-06-02T08:00:00Z", "2024-06-03T09:00:00Z",
        ],
        "actual_departure_datetime": [
            "2024-06-01T07:10:00Z", "2024-06-02T08:05:00Z", np.nan,
        ],
        "scheduled_arrival_datetime": [
            "2024-06-01T11:00:00Z", "2024-06-02T12:00:00Z", "2024-06-03T15:00:00Z",
        ],
        "actual_arrival_datetime": [
            "2024-06-01T11:30:00Z", "2024-06-02T12:15:00Z", np.nan,
        ],
        "origin_city": ["Sydney", "Melbourne", "Perth"],
        "origin_state": [np.nan, "VIC", np.nan],
        "origin_country": ["Australia", "Australia", "Australia"],
        "destination_city": ["Brisbane", "Sydney", "Melbourne"],
        "destination_state": [np.nan, "NSW", np.nan],
        "destination_country": ["Australia", "Australia", "Australia"],
        "carrier_id": ["CR001", "CR002", "CR003"],
        "carrier_name": ["Test Carrier", "Another", np.nan],
        "route_distance_km": [900, 880, 3300],
        "route_type": ["highway", "highway", "air"],
        "vehicle_type": ["truck", "truck", "airplane"],
        "weather_condition_departure": ["clear", "rain", "clear"],
        "weather_condition_arrival": ["clear", "rain", np.nan],
        "delay_predicted_minutes": [10, 15, 5],
        "delay_actual_minutes": [30, 15, np.nan],
        "carrier_on_time_rate": [0.92, 0.85, 0.93],
        "shipment_status": ["delivered", "delivered", "in_transit"],
        "priority_level": ["standard", "critical", "standard"],
        "shipment_type": ["general", "hazardous", "perishable"],
        "notes": ["note1", "note2", "note3"],
    })


def test_preprocess_adds_record_id(raw_df):
    df = preprocess(raw_df)
    assert "record_id" in df.columns
    assert df["record_id"].nunique() == len(df)


def test_preprocess_has_outcome_flag(raw_df):
    df = preprocess(raw_df)
    assert df["has_outcome"].sum() == 2
    assert (~df["has_outcome"]).sum() == 1


def test_preprocess_delay_target(raw_df):
    df = preprocess(raw_df)
    # S001: actual 11:30 - scheduled 11:00 = 30 min
    s001 = df[df["shipment_id"] == "S001"].iloc[0]
    assert abs(s001["delay_target_minutes"] - 30.0) < 0.01

    # S002: actual 12:15 - scheduled 12:00 = 15 min
    s002 = df[df["shipment_id"] == "S002"].iloc[0]
    assert abs(s002["delay_target_minutes"] - 15.0) < 0.01


def test_preprocess_is_delayed(raw_df):
    df = preprocess(raw_df)
    # Both S001 (30min) and S002 (15min) should be delayed (threshold=0)
    completed = df[df["has_outcome"] == True]
    assert all(completed["is_delayed"] == 1)


def test_preprocess_no_outcome_has_nan_target(raw_df):
    df = preprocess(raw_df)
    s003 = df[df["shipment_id"] == "S003"].iloc[0]
    assert pd.isna(s003["delay_target_minutes"])
    assert pd.isna(s003["is_delayed"])


def test_preprocess_departure_delay(raw_df):
    df = preprocess(raw_df)
    s001 = df[df["shipment_id"] == "S001"].iloc[0]
    # actual 07:10 - scheduled 07:00 = 10 min
    assert abs(s001["departure_delay_min"] - 10.0) < 0.01


def test_preprocess_scheduled_duration(raw_df):
    df = preprocess(raw_df)
    s001 = df[df["shipment_id"] == "S001"].iloc[0]
    # 11:00 - 07:00 = 240 min
    assert abs(s001["scheduled_duration_min"] - 240.0) < 0.01


def test_preprocess_categorical_lowercase(raw_df):
    df = preprocess(raw_df)
    assert all(df["route_type"].dropna().str.islower())
    assert all(df["vehicle_type"].dropna().str.islower())
