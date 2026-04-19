"""
Shared sensor schema for the atmospheric pipeline.
---------------------------------------------------
Canonical column names match the Google Sheet export (normalized to lowercase).
"""

from __future__ import annotations

import pandas as pd

# Default multivariate sensor columns (order preserved for LSTM / IsolationForest)
DEFAULT_SENSOR_COLUMNS = ["ws", "wd", "pressure", "rh", "temp", "dew", "rain"]


def sensor_base_columns(config: dict) -> list[str]:
    """Return ordered list of raw sensor column names expected in the master CSV."""
    cols = (config.get("sensors") or {}).get("columns")
    if not cols:
        return list(DEFAULT_SENSOR_COLUMNS)
    return [str(c).strip().lower() for c in cols]


def resolve_sensor_data_columns(df: pd.DataFrame, config: dict) -> list[str]:
    """Prefer Kalman-filtered columns when present."""
    out: list[str] = []
    for c in sensor_base_columns(config):
        filt = f"{c}_filt"
        if filt in df.columns:
            out.append(filt)
        elif c in df.columns:
            out.append(c)
    return out


def capped_seasonal_period(requested: int, n_rows: int) -> int:
    """
    Shrink configured season length when there are not enough rows.

    ``statsmodels.seasonal_decompose`` needs roughly ``n_rows >= 2 * period``.
    """
    requested = max(2, int(requested))
    if n_rows < 8:
        return 2
    max_p = max(2, n_rows // 3)
    return min(requested, max_p)


def capped_sarima_seasonal_m(requested: int, n_rows: int, min_seasons: int = 10) -> int:
    """
    Cap SARIMAX seasonal period ``m`` so there are enough rows per season.

    Statsmodels warns when ``m`` is large relative to ``n`` (seasonal ARMA
    poorly identified). Use ``min_seasons`` full cycles as a floor:
    ``m <= n_rows // min_seasons`` (approximately).
    """
    requested = max(2, int(requested))
    ms = max(2, int(min_seasons))
    if n_rows < ms + 20:
        return 2
    max_m = max(2, n_rows // ms)
    return min(requested, max_m)


def prediction_primary_base(config: dict) -> str:
    """Base name (no _filt) of the series used for SARIMA and forecast-error primary."""
    pred = config.get("prediction") or {}
    primary = str(pred.get("primary_sensor", "temp")).strip().lower()
    bases = sensor_base_columns(config)
    if primary in bases:
        return primary
    return bases[0] if bases else "temp"
