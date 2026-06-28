"""
Seasonal Decomposition Service
------------------------------
Extracts trend, seasonal, and residual components from sensor time series.
Outputs are used by downstream prediction and anomaly detection.
"""

import logging

import numpy as np
import pandas as pd

from services.pipeline_schema import capped_seasonal_period, sensor_base_columns

logger = logging.getLogger(__name__)


def _decompose_series(
    series: pd.Series,
    period: int = 24,
    model: str = "additive",
) -> tuple:
    """
    Decompose series into trend, seasonal, residual.

    Args:
        series: Input time series
        period: Seasonality period (e.g., 24 for hourly)
        model: 'additive' or 'multiplicative'

    Returns:
        (trend, seasonal, residual) or (None, None, None) on failure
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        logger.warning("statsmodels not installed; skipping seasonal decomposition")
        return None, None, None

    if len(series.dropna()) < 2 * period:
        logger.warning("Insufficient data for decomposition (need >= 2*period)")
        return None, None, None

    try:
        filled = series.ffill().bfill()
        decomposed = seasonal_decompose(
            filled,
            period=period,
            model=model,
            extrapolate_trend="freq",
        )
        return decomposed.trend, decomposed.seasonal, decomposed.resid
    except Exception as e:
        logger.warning("Seasonal decomposition failed: %s", e)
        return None, None, None


def run_seasonal_decomposition(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Run seasonal decomposition on sensor columns.
    Adds trend, seasonal, residual columns for downstream use.

    Args:
        df: DataFrame with configured sensor columns (or _filt variants)
        config: Pipeline configuration

    Returns:
        DataFrame with added *_trend, *_seasonal, *_residual columns
    """
    if df.empty:
        return df.copy()

    sd_cfg = config.get("model", {}).get("seasonal_decomposition", {})
    period_cfg = int(sd_cfg.get("period", 24))
    model_type = sd_cfg.get("model", "additive")
    period = capped_seasonal_period(period_cfg, len(df))
    if period != period_cfg:
        logger.info(
            "Seasonal decomposition: capped period %s -> %s (rows=%s)",
            period_cfg,
            period,
            len(df),
        )

    result = df.copy()
    base_cols = []
    for c in sensor_base_columns(config):
        if f"{c}_filt" in df.columns:
            base_cols.append((f"{c}_filt", c))
        elif c in df.columns:
            base_cols.append((c, c))

    for col, prefix in base_cols:
        trend, seasonal, resid = _decompose_series(result[col], period, model_type)
        if trend is not None:
            result[f"{prefix}_trend"] = trend
        if seasonal is not None:
            result[f"{prefix}_seasonal"] = seasonal
        if resid is not None:
            result[f"{prefix}_residual"] = resid

    logger.info("Seasonal decomposition: period=%d, model=%s", period, model_type)
    return result
