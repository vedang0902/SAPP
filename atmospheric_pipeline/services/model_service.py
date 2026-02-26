"""
Model Service - Seasonal Decomposition + Isolation Forest + Z-score
------------------------------------------------------------------
Combines anomaly detection: seasonal decomposition residual, Isolation Forest,
and Z-score thresholding. Final anomaly flag is combined from all methods.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

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
    Run anomaly detection: seasonal decomposition + Isolation Forest + Z-score.
    Combine flags: anomaly if ANY method flags it.

    Args:
        df: DataFrame with features (must have temperature, humidity, pressure or _filt variants)
        config: Pipeline configuration

    Returns:
        DataFrame with anomaly columns: anomaly_iforest, anomaly_zscore, anomaly_residual, anomaly_combined
    """
    if df.empty:
        return df.copy()

    cfg = config.get("model", {})
    if_cfg = cfg.get("isolation_forest", {})
    z_cfg = cfg.get("z_score", {})
    sd_cfg = cfg.get("seasonal_decomposition", {})

    contamination = if_cfg.get("contamination", 0.04)
    z_threshold = z_cfg.get("threshold", 3.0)
    period = sd_cfg.get("period", 24)
    model_type = sd_cfg.get("model", "additive")

    result = df.copy()

    # Columns for ML
    use_cols = []
    for c in ["temperature", "humidity", "pressure"]:
        if f"{c}_filt" in result.columns:
            use_cols.append(f"{c}_filt")
        elif c in result.columns:
            use_cols.append(c)
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

    # 3. Seasonal decomposition residual (use primary column, e.g. temperature)
    primary = use_cols[0]
    resid = seasonal_decomposition_residual(result[primary], period=period, model=model_type)
    if resid.notna().any():
        res_flags = zscore_anomaly(resid.dropna(), z_threshold)
        # Map back to full index
        result["anomaly_residual"] = 0
        valid_idx = resid.dropna().index
        result.loc[valid_idx, "anomaly_residual"] = res_flags.astype(int)
    else:
        result["anomaly_residual"] = 0

    # 4. Combined: anomaly if any method flags
    result["anomaly_combined"] = (
        (result["anomaly_iforest"] | result["anomaly_zscore"] | result["anomaly_residual"])
    ).astype(int)
    result["anomaly"] = result["anomaly_combined"].map(lambda x: -1 if x == 1 else 1)  # -1 = outlier for compatibility

    logger.info(
        "Model service: IF=%d, Z=%d, Resid=%d, Combined=%d anomalies",
        result["anomaly_iforest"].sum(),
        result["anomaly_zscore"].sum(),
        result["anomaly_residual"].sum(),
        result["anomaly_combined"].sum(),
    )
    return result
