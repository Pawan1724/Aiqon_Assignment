"""
data_validation.py  –  Dataset quality checks and audit reporting.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from src import CONFIG, logger


# ------------------------------------------------------------------ #
#  Quality-check functions                                           #
# ------------------------------------------------------------------ #

def check_schema(df: pd.DataFrame) -> list[str]:
    """Return list of expected columns that are missing."""
    expected = [
        "shipment_id", "scheduled_departure_datetime",
        "actual_departure_datetime", "scheduled_arrival_datetime",
        "actual_arrival_datetime", "origin_city", "origin_country",
        "destination_city", "destination_country", "carrier_id",
        "route_distance_km", "route_type", "vehicle_type",
        "weather_condition_departure", "delay_predicted_minutes",
        "delay_actual_minutes", "carrier_on_time_rate",
        "shipment_status", "priority_level", "shipment_type",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        logger.warning("Missing expected columns: %s", missing)
    return missing


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with duplicated shipment IDs (all occurrences)."""
    id_col = CONFIG["data"]["shipment_id_col"]
    mask = df[id_col].duplicated(keep=False)
    dups = df.loc[mask].sort_values(id_col)
    if len(dups):
        logger.warning("%d rows share a duplicated shipment_id", len(dups))
    return dups


def check_missing_targets(df: pd.DataFrame) -> dict[str, int]:
    """Count missing values in critical target columns."""
    cols = [
        "actual_departure_datetime", "actual_arrival_datetime",
        "delay_actual_minutes", "weather_condition_arrival",
    ]
    result = {}
    for c in cols:
        if c in df.columns:
            n = int(df[c].isnull().sum())
            result[c] = n
    return result


def check_timestamp_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows where timestamps violate logical ordering:
      scheduled_dep <= actual_dep   (if both present)
      scheduled_arr >= scheduled_dep
      actual_arr >= actual_dep      (if both present)
    Returns a DataFrame of flagged rows with flag descriptions.
    """
    flags: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        sd = pd.to_datetime(row.get("scheduled_departure_datetime"), errors="coerce")
        sa = pd.to_datetime(row.get("scheduled_arrival_datetime"), errors="coerce")
        ad = pd.to_datetime(row.get("actual_departure_datetime"), errors="coerce")
        aa = pd.to_datetime(row.get("actual_arrival_datetime"), errors="coerce")

        if pd.notna(sa) and pd.notna(sd) and sa < sd:
            flags.append({"index": idx, "shipment_id": row.get("shipment_id"),
                          "flag": "scheduled_arrival < scheduled_departure"})
        if pd.notna(ad) and pd.notna(sd) and ad < sd:
            flags.append({"index": idx, "shipment_id": row.get("shipment_id"),
                          "flag": "actual_departure < scheduled_departure (early dep)"})
        if pd.notna(aa) and pd.notna(ad) and aa < ad:
            flags.append({"index": idx, "shipment_id": row.get("shipment_id"),
                          "flag": "actual_arrival < actual_departure"})
        if pd.notna(aa) and pd.notna(sa):
            delay_derived = (aa - sa).total_seconds() / 60.0
            delay_field = row.get("delay_actual_minutes")
            if pd.notna(delay_field) and abs(delay_derived - delay_field) > 1.0:
                flags.append({
                    "index": idx,
                    "shipment_id": row.get("shipment_id"),
                    "flag": (f"delay_actual_minutes ({delay_field}) != "
                             f"timestamp-derived ({delay_derived:.1f}min)"),
                })

    return pd.DataFrame(flags)


def check_negative_delays(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where delay_actual_minutes < 0."""
    if "delay_actual_minutes" not in df.columns:
        return pd.DataFrame()
    mask = df["delay_actual_minutes"].fillna(0) < 0
    return df.loc[mask]


# ------------------------------------------------------------------ #
#  Full audit report                                                  #
# ------------------------------------------------------------------ #

def run_full_audit(df: pd.DataFrame, save_dir: str | None = None) -> dict:
    """
    Execute all quality checks and optionally write a markdown report.

    Returns a dictionary of audit results.
    """
    results: dict[str, Any] = {}

    results["shape"] = df.shape
    results["missing_columns"] = check_schema(df)
    results["duplicate_ids"] = check_duplicates(df)
    results["missing_targets"] = check_missing_targets(df)
    results["timestamp_flags"] = check_timestamp_order(df)
    results["negative_delays"] = check_negative_delays(df)
    results["null_counts"] = df.isnull().sum().to_dict()

    # Unique shipment IDs
    id_col = CONFIG["data"]["shipment_id_col"]
    results["total_rows"] = len(df)
    results["unique_ids"] = df[id_col].nunique()
    results["duplicated_id_count"] = int(df[id_col].duplicated().sum())

    # Shipment status distribution
    if "shipment_status" in df.columns:
        results["status_distribution"] = df["shipment_status"].value_counts().to_dict()

    if save_dir:
        _write_report(results, save_dir)

    return results


def _write_report(results: dict, save_dir: str) -> None:
    """Write a markdown data-quality report."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "data_quality_report.md")

    lines: list[str] = [
        "# Data Quality Report",
        "",
        f"**Total rows**: {results['shape'][0]}  ",
        f"**Total columns**: {results['shape'][1]}  ",
        f"**Unique shipment IDs**: {results['unique_ids']}  ",
        f"**Duplicated IDs (extra rows)**: {results['duplicated_id_count']}  ",
        "",
    ]

    # Missing columns
    if results["missing_columns"]:
        lines.append("## Missing Expected Columns")
        for c in results["missing_columns"]:
            lines.append(f"- `{c}`")
        lines.append("")

    # Missing targets
    lines.append("## Missing Target / Outcome Values")
    lines.append("")
    lines.append("| Column | Missing Count |")
    lines.append("|--------|--------------|")
    for col, cnt in results["missing_targets"].items():
        lines.append(f"| `{col}` | {cnt} |")
    lines.append("")

    # Null counts
    lines.append("## Null Counts per Column")
    lines.append("")
    lines.append("| Column | Nulls |")
    lines.append("|--------|-------|")
    for col, cnt in results["null_counts"].items():
        if cnt > 0:
            lines.append(f"| `{col}` | {cnt} |")
    lines.append("")

    # Timestamp flags
    ts_flags = results["timestamp_flags"]
    lines.append(f"## Timestamp Ordering Flags ({len(ts_flags)} issues)")
    lines.append("")
    if len(ts_flags):
        lines.append("| Index | Shipment ID | Flag |")
        lines.append("|-------|-------------|------|")
        for _, row in ts_flags.iterrows():
            lines.append(f"| {row['index']} | {row['shipment_id']} | {row['flag']} |")
    else:
        lines.append("No timestamp ordering issues found.")
    lines.append("")

    # Duplicate IDs
    dup_df = results["duplicate_ids"]
    lines.append(f"## Duplicated Shipment IDs ({len(dup_df)} rows)")
    lines.append("")
    if len(dup_df):
        dup_ids = dup_df[CONFIG["data"]["shipment_id_col"]].unique().tolist()
        lines.append(f"IDs: {dup_ids}")
    lines.append("")

    # Status distribution
    if "status_distribution" in results:
        lines.append("## Shipment Status Distribution")
        lines.append("")
        lines.append("| Status | Count |")
        lines.append("|--------|-------|")
        for status, cnt in results["status_distribution"].items():
            lines.append(f"| {status} | {cnt} |")
        lines.append("")

    # Negative delays
    neg = results["negative_delays"]
    lines.append(f"## Negative Delays ({len(neg)} rows)")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Data quality report written to %s", path)
