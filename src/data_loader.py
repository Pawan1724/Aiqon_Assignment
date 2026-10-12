"""
data_loader.py  –  Load the raw Excel dataset.
"""
from __future__ import annotations

import os
import pandas as pd
from src import CONFIG, logger


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the raw Excel file and return as a DataFrame.

    Parameters
    ----------
    path : str, optional
        Absolute or project-relative path to the Excel file.
        Defaults to the path in settings.yaml.

    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        path = CONFIG["paths"]["raw_data"]

    if not os.path.isabs(path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw data file not found at '{path}'. "
            "Place 'logistics-shipment-delay-prediction.xlsx' in data/raw/."
        )

    logger.info("Loading raw data from %s", path)
    df = pd.read_excel(path, engine="openpyxl")
    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df
