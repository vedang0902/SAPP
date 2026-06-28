"""
Feature Engineering Service
---------------------------
Rolling mean, std, skew, kurtosis, gradient, energy metrics.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from services.pipeline_schema import sensor_base_columns

logger = logging.getLogger(__name__)


def _pick_feature_col(df: pd.DataFrame, base: str) -> Optional[str]:
    for name in (f"{base}_mean", f"{base}_filt", base):
        if name in df.columns:
            return name
    return None


def run_feature_engineering(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Extract temporal and statistical features from filtered sensor data.

    Uses filtered columns when present (see sensors.columns in config).

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

    base_cols = []
    for col in sensor_base_columns(config):
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

    # Composite ratios (when temp / rh / pressure features exist)
    t_col = _pick_feature_col(result, "temp")
    h_col = _pick_feature_col(result, "rh")
    p_col = _pick_feature_col(result, "pressure")
    if t_col and h_col:
        result["temp_rh_ratio"] = result[t_col] / (result[h_col] + 1e-6)
    if t_col and p_col:
        result["pressure_temp_product"] = result[p_col] * result[t_col]

    result = result.dropna().reset_index(drop=True)
    logger.info("Feature engineering: %d rows, window=%d", len(result), window)
    return result
