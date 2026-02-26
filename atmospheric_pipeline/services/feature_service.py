"""
Feature Engineering Service
---------------------------
Rolling mean, std, skew, kurtosis, gradient, energy metrics.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_feature_engineering(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Extract temporal and statistical features from filtered sensor data.

    Uses columns: temperature_filt, humidity_filt, pressure_filt (or temperature, etc.)
    Configurable rolling window from config.yaml.

    Args:
        df: Filtered DataFrame
        config: Pipeline configuration

    Returns:
        DataFrame with feature columns
    """
    if df.empty:
        return df.copy()

    cfg = config.get("feature_engineering", {})
    window = cfg.get("rolling_window", 5)

    # Use filtered columns if available, else raw
    base_cols = []
    for col in ["temperature", "humidity", "pressure"]:
        if f"{col}_filt" in df.columns:
            base_cols.append(f"{col}_filt")
        elif col in df.columns:
            base_cols.append(col)

    if not base_cols:
        logger.warning("No sensor columns found for feature engineering")
        return df.copy()

    result = df.copy()
    for col in base_cols:
        raw_col = col.replace("_filt", "")
        prefix = raw_col
        # Rolling mean
        result[f"{prefix}_mean"] = result[col].rolling(window).mean()
        # Rolling std
        result[f"{prefix}_std"] = result[col].rolling(window).std()
        # Skewness
        result[f"{prefix}_skew"] = result[col].rolling(window).skew()
        # Kurtosis
        result[f"{prefix}_kurt"] = result[col].rolling(window).kurt()
        # Gradient (rate of change over window)
        result[f"{prefix}_gradient"] = result[col].diff(window)
        # Energy metric (squared derivative, rolling mean)
        delta = result[col].diff()
        result[f"{prefix}_energy"] = (delta ** 2).rolling(window).mean()

    # Composite ratios
    t_col = next((c for c in ["temperature_mean", "temperature_filt", "temperature"] if c in result.columns), None)
    h_col = next((c for c in ["humidity_mean", "humidity_filt", "humidity"] if c in result.columns), None)
    p_col = next((c for c in ["pressure_mean", "pressure_filt", "pressure"] if c in result.columns), None)
    if t_col and h_col:
        result["temp_rh_ratio"] = result[t_col] / (result[h_col] + 1e-6)
    if t_col and p_col:
        result["pressure_temp_product"] = result[p_col] * result[t_col]

    result = result.dropna().reset_index(drop=True)
    logger.info("Feature engineering: %d rows, window=%d", len(result), window)
    return result
