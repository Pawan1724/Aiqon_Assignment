"""
feature_engineering.py  –  Build model-ready feature matrices.

Prediction scenario: **Pre-departure prediction**.
Only features available before or at the moment of departure are used.

Excluded columns (leakage or non-feature):
  - actual_departure_datetime, actual_arrival_datetime
  - delay_actual_minutes, delay_predicted_minutes
  - delay_target_minutes, is_delayed
  - shipment_status (reveals outcome)
  - notes (free text, not modelled)
  - has_outcome, record_id, delay_discrepancy_min
  - actual_duration_min

Engineered features:
  - scheduled_duration_min  (already from preprocessing)
  - departure_delay_min     (already; set to 0 where missing for pre-departure)
  - hour_of_departure       (cyclical components)
  - day_of_week             (from scheduled departure)
  - is_international        (origin_country != destination_country)
  - route_type / vehicle_type / weather_condition_departure  (one-hot)
  - priority_level / shipment_type                           (one-hot)
  - carrier_on_time_rate    (numeric, already available)
  - route_distance_km       (numeric)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import logger

# Columns that are targets or contain future information
_LEAKAGE_COLS = {
    "actual_departure_datetime",
    "actual_arrival_datetime",
    "delay_actual_minutes",
    "delay_predicted_minutes",
    "delay_target_minutes",
    "is_delayed",
    "shipment_status",
    "notes",
    "has_outcome",
    "record_id",
    "delay_discrepancy_min",
    "actual_duration_min",
    "shipment_id",
    "origin_state",
    "destination_state",
    "carrier_name",
    "carrier_id",
    "weather_condition_arrival",   # not known pre-departure
}

NUMERIC_FEATURES = [
    "route_distance_km",
    "carrier_on_time_rate",
    "scheduled_duration_min",
    "departure_delay_min",
    "hour_of_departure",
    "day_of_week",
    "is_international",
]

CATEGORICAL_FEATURES = [
    "route_type",
    "vehicle_type",
    "weather_condition_departure",
    "priority_level",
    "shipment_type",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered columns and return a copy.
    Safe to call on the full dataset (train+test) before splitting,
    because no target or future information is used.
    """
    df = df.copy()

    # Hour & day from scheduled departure
    sd = pd.to_datetime(df["scheduled_departure_datetime"], utc=True, errors="coerce")
    df["hour_of_departure"] = sd.dt.hour
    df["day_of_week"] = sd.dt.dayofweek  # 0=Mon, 6=Sun

    # International flag
    df["is_international"] = (
        df["origin_country"].str.lower().str.strip()
        != df["destination_country"].str.lower().str.strip()
    ).astype(int)

    # Fill missing departure_delay_min with 0 (pre-departure scenario)
    if "departure_delay_min" in df.columns:
        df["departure_delay_min"] = df["departure_delay_min"].fillna(0.0)

    return df


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Select feature columns, separating numeric and categorical.

    Returns
    -------
    X : DataFrame   – feature columns only
    numeric_cols : list[str]
    categorical_cols : list[str]
    """
    df = engineer_features(df)

    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    X = df[num_cols + cat_cols].copy()
    logger.info(
        "Feature matrix: %d numeric + %d categorical = %d features",
        len(num_cols), len(cat_cols), len(num_cols) + len(cat_cols),
    )
    return X, num_cols, cat_cols


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer to be fitted on training data.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names after fitting the preprocessor."""
    names: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            names.extend(trans.get_feature_names_out())
        elif isinstance(trans, Pipeline):
            last = trans.steps[-1][1]
            if hasattr(last, "get_feature_names_out"):
                names.extend(last.get_feature_names_out())
            else:
                names.extend(cols)
        else:
            names.extend(cols)
    return names
