"""
Model Service - Hybrid Anomaly Detection
---------------------------------------
Combines anomaly detection: seasonal decomposition residual, Isolation Forest,
Z-score thresholding, and forecast-error-based detection.
An anomaly is flagged if ANY of: Isolation Forest, Z-score, or forecast error > threshold.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from services.pipeline_schema import (
    capped_seasonal_period,
    prediction_primary_base,
    resolve_sensor_data_columns,
)

logger = logging.getLogger(__name__)


def seasonal_decomposition_residual(series: pd.Series, period: int = 24, model: str = "additive") -> pd.Series:
    """
    Decompose series into trend, seasonal, residual. Return residual for anomaly detection.

    Args:
        series: Input time series
        period: Seasonality period (e.g., 24 for hourly)
        model: 'additive' or 'multiplicative'

    Returns:
        Residual series (NaN where decomposition fails)
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        logger.warning("statsmodels not installed; skipping seasonal decomposition")
        return pd.Series(np.nan, index=series.index)

    # Need at least 2*period points
    if len(series.dropna()) < 2 * period:
        return pd.Series(np.nan, index=series.index)

    try:
        filled = series.ffill().bfill()
        decomposed = seasonal_decompose(filled, period=period, model=model, extrapolate_trend="freq")
        return decomposed.resid
    except Exception as e:
        logger.warning("Seasonal decomposition failed: %s", e)
        return pd.Series(np.nan, index=series.index)


def zscore_anomaly(series: pd.Series, threshold: float = 3.0) -> np.ndarray:
    """
    Flag anomalies where |z-score| > threshold.

    Returns:
        Boolean array: True = anomaly
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return np.zeros(len(series), dtype=bool)
    z = np.abs((series - mean) / std)
    return (z > threshold).values


def run_model_service(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Run hybrid anomaly detection: Isolation Forest + Z-score + seasonal residual + forecast error.
    An anomaly is flagged if ANY of: Isolation Forest, Z-score, residual, or forecast error > threshold.

    Args:
        df: DataFrame with features (sensor columns or _filt variants).
            If forecast_error present (from prediction_service), used for forecast-based anomaly.
        config: Pipeline configuration

    Returns:
        DataFrame with anomaly columns: anomaly_iforest, anomaly_zscore, anomaly_residual,
        anomaly_forecast_error, anomaly_combined
    """
    if df.empty:
        return df.copy()

    cfg = config.get("model", {})
    if_cfg = cfg.get("isolation_forest", {})
    z_cfg = cfg.get("z_score", {})
    sd_cfg = cfg.get("seasonal_decomposition", {})

    contamination = if_cfg.get("contamination", 0.04)
    z_threshold = z_cfg.get("threshold", 3.0)
    period_cfg = int(sd_cfg.get("period", 24))
    period = capped_seasonal_period(period_cfg, len(df))
    model_type = sd_cfg.get("model", "additive")

    # Forecast error threshold (from prediction config if available)
    pred_cfg = config.get("prediction", {})
    forecast_error_threshold = pred_cfg.get("forecast_error_threshold", 2.5)

    result = df.copy()

    use_cols = resolve_sensor_data_columns(result, config)
    if not use_cols:
        logger.warning("No sensor columns for model service")
        return result

    X = result[use_cols].copy()
    X = X.fillna(X.mean())

    # 1. Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=if_cfg.get("random_state", 42))
    pred = iso.fit_predict(X)
    result["anomaly_iforest"] = (pred == -1).astype(int)

    # 2. Z-score per column
    z_flags = np.zeros(len(result), dtype=int)
    for col in use_cols:
        z_flags |= zscore_anomaly(result[col], z_threshold).astype(int)
    result["anomaly_zscore"] = z_flags

    # 3. Seasonal decomposition residual (use pre-computed if available from seasonal_decomposition_service)
    primary_base = prediction_primary_base(config)
    primary = next(
        (c for c in use_cols if c.replace("_filt", "") == primary_base),
        use_cols[0],
    )
    resid = None
    if f"{primary_base}_residual" in result.columns and result[f"{primary_base}_residual"].notna().any():
        resid = result[f"{primary_base}_residual"]
    else:
        resid = seasonal_decomposition_residual(result[primary], period=period, model=model_type)

    if resid is not None and resid.notna().any():
        res_flags = zscore_anomaly(resid.dropna(), z_threshold)
        result["anomaly_residual"] = 0
        valid_idx = resid.dropna().index
        result.loc[valid_idx, "anomaly_residual"] = res_flags.astype(int)
    else:
        result["anomaly_residual"] = 0

    # 4. Forecast-error-based anomaly (if prediction_service ran and added forecast_error)
    if "forecast_error" in result.columns:
        result["anomaly_forecast_error"] = (
            result["forecast_error"] > forecast_error_threshold
        ).astype(int)
    else:
        result["anomaly_forecast_error"] = 0

    # 5. Combined: anomaly if ANY method flags
    result["anomaly_combined"] = (
        result["anomaly_iforest"]
        | result["anomaly_zscore"]
        | result["anomaly_residual"]
        | result["anomaly_forecast_error"]
    ).astype(int)
    result["anomaly"] = result["anomaly_combined"].map(lambda x: -1 if x == 1 else 1)

    logger.info(
        "Model service: IF=%d, Z=%d, Resid=%d, ForecastErr=%d, Combined=%d anomalies",
        result["anomaly_iforest"].sum(),
        result["anomaly_zscore"].sum(),
        result["anomaly_residual"].sum(),
        result["anomaly_forecast_error"].sum(),
        result["anomaly_combined"].sum(),
    )
    return result
