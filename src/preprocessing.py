"""
preprocessing.py  –  Deterministic data cleaning and target engineering.

Key design decisions
--------------------
* The original Excel file is never modified.
* Timestamps are parsed to UTC-aware datetime and durations computed in minutes.
* `delay_target_minutes` is derived from (actual_arrival - scheduled_arrival).
  If the existing `delay_actual_minutes` column disagrees, the discrepancy is
  logged but the timestamp-derived value is used as the single source of truth.
* Records without a valid actual_arrival are kept but flagged
  (`has_outcome = False`) and excluded from supervised training / evaluation.
* `delay_predicted_minutes` is treated as a **potential leakage** feature and
  is excluded from model inputs by default.
* Duplicate shipment IDs receive a unique internal `record_id`.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from src import CONFIG, logger


# ------------------------------------------------------------------ helpers
def _to_utc_datetime(series: pd.Series) -> pd.Series:
    """Parse a Series of ISO-8601 strings to UTC-aware Timestamps."""
    return pd.to_datetime(series, utc=True, errors="coerce")


# ------------------------------------------------------------------ public
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame and engineer targets.

    Returns a new DataFrame with additional columns:
        record_id               – unique integer identifier
        has_outcome             – True if actual_arrival is present
        scheduled_duration_min  – scheduled journey duration in minutes
        actual_duration_min     – actual journey duration (NaN if missing)
        departure_delay_min     – actual_dep − scheduled_dep in minutes
        delay_target_minutes    – (actual_arrival − scheduled_arrival) in min
        is_delayed              – 1 if delay_target_minutes > threshold, else 0
        delay_discrepancy_min   – difference between file field and derived delay
    """
    df = df.copy()

    # ---- 1. Unique record ID ----
    df.insert(0, "record_id", range(1, len(df) + 1))

    # ---- 2. Standardise column names (already snake_case) ----
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ---- 3. Parse timestamps ----
    ts_cols = [
        "scheduled_departure_datetime",
        "actual_departure_datetime",
        "scheduled_arrival_datetime",
        "actual_arrival_datetime",
    ]
    for col in ts_cols:
        if col in df.columns:
            df[col] = _to_utc_datetime(df[col])

    # ---- 4. Outcome flag ----
    df["has_outcome"] = df["actual_arrival_datetime"].notna()

    # ---- 5. Duration features ----
    df["scheduled_duration_min"] = (
        (df["scheduled_arrival_datetime"] - df["scheduled_departure_datetime"])
        .dt.total_seconds() / 60.0
    )
    df["actual_duration_min"] = np.where(
        df["has_outcome"],
        (df["actual_arrival_datetime"] - df["actual_departure_datetime"])
        .dt.total_seconds() / 60.0,
        np.nan,
    )
    df["departure_delay_min"] = np.where(
        df["actual_departure_datetime"].notna(),
        (df["actual_departure_datetime"] - df["scheduled_departure_datetime"])
        .dt.total_seconds() / 60.0,
        np.nan,
    )

    # ---- 6. Target: delay from timestamps ----
    df["delay_target_minutes"] = np.where(
        df["has_outcome"],
        (df["actual_arrival_datetime"] - df["scheduled_arrival_datetime"])
        .dt.total_seconds() / 60.0,
        np.nan,
    )

    # ---- 7. Discrepancy check ----
    if "delay_actual_minutes" in df.columns:
        df["delay_discrepancy_min"] = np.where(
            df["has_outcome"] & df["delay_actual_minutes"].notna(),
            df["delay_target_minutes"] - df["delay_actual_minutes"],
            np.nan,
        )
        n_disc = (df["delay_discrepancy_min"].abs() > 1.0).sum()
        if n_disc:
            logger.warning(
                "%d records have >1 min discrepancy between "
                "delay_actual_minutes and timestamp-derived delay", n_disc,
            )

    # ---- 8. Binary classification target ----
    threshold = CONFIG["data"]["delay_threshold_minutes"]
    df["is_delayed"] = np.where(
        df["has_outcome"],
        (df["delay_target_minutes"] > threshold).astype(int),
        np.nan,
    )

    # ---- 9. Standardise categorical values ----
    cat_cols = [
        "origin_city", "origin_country", "destination_city",
        "destination_country", "route_type", "vehicle_type",
        "weather_condition_departure", "weather_condition_arrival",
        "shipment_status", "priority_level", "shipment_type",
        "carrier_id",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": np.nan, "none": np.nan, "": np.nan})

    # ---- 10. Numeric validation ----
    if "route_distance_km" in df.columns:
        df["route_distance_km"] = pd.to_numeric(
            df["route_distance_km"], errors="coerce"
        )
    if "carrier_on_time_rate" in df.columns:
        df["carrier_on_time_rate"] = pd.to_numeric(
            df["carrier_on_time_rate"], errors="coerce"
        )

    logger.info(
        "Preprocessing complete: %d rows, %d with outcome, %d without",
        len(df), df["has_outcome"].sum(), (~df["has_outcome"]).sum(),
    )
    return df


def save_processed(df: pd.DataFrame, path: str | None = None) -> str:
    """Save the processed DataFrame to CSV and return the path."""
    if path is None:
        path = CONFIG["paths"]["processed_data"]
    if not os.path.isabs(path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, path)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Processed data saved to %s", path)
    return path
