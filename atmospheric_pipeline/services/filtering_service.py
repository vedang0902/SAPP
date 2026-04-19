"""
Filtering Service - Median Filter + Adaptive Kalman Filter
---------------------------------------------------------
Median filter removes impulse noise; adaptive Kalman filter smooths data
with auto-adjustment of Q/R based on variance.
"""

import logging
import statistics
from typing import Optional

import numpy as np
import pandas as pd

from services.pipeline_schema import sensor_base_columns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Median Filter
# ---------------------------------------------------------------------------
def median_filter(data_window: list, new_value: float, window_size: int) -> float:
    """
    Apply median filter to smooth impulse noise.

    Args:
        data_window: Rolling window (list, mutated in-place)
        new_value: Latest sensor reading
        window_size: Window size (e.g., 3, 5)

    Returns:
        Median-filtered value
    """
    data_window.append(new_value)
    if len(data_window) > window_size:
        data_window.pop(0)
    return statistics.median(data_window)


# ---------------------------------------------------------------------------
# Adaptive Kalman Filter
# ---------------------------------------------------------------------------
class AdaptiveKalmanFilter:
    """
    Kalman filter with adaptive Q (process noise) and R (measurement noise)
    based on observed variance.
    """

    def __init__(
        self,
        process_variance: float = 0.01,
        measurement_variance: float = 0.5,
        estimated_error: float = 1.0,
        initial_value: Optional[float] = None,
        adaptation_rate: float = 0.1,
    ):
        self.Q = process_variance
        self.R = measurement_variance
        self.P = estimated_error
        self.x = initial_value
        self.adaptation_rate = adaptation_rate
        self._measurement_history: list = []
        self._history_max = 20

    def update(self, measurement: float) -> float:
        """
        Update filter with new measurement; optionally adapt Q/R.

        Args:
            measurement: Current sensor reading

        Returns:
            Filtered estimate
        """
        if self.x is None:
            self.x = measurement
            self._measurement_history.append(measurement)
            return self.x

        # Adaptive step: estimate measurement variance from recent history
        self._measurement_history.append(measurement)
        if len(self._measurement_history) > self._history_max:
            self._measurement_history.pop(0)
        if len(self._measurement_history) >= 5:
            var = np.var(self._measurement_history)
            # Blend R toward observed variance (avoid division by zero)
            self.R = (1 - self.adaptation_rate) * self.R + self.adaptation_rate * max(var, 1e-6)

        # Prediction
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P

        return self.x


# ---------------------------------------------------------------------------
# Apply Median + Kalman to Series
# ---------------------------------------------------------------------------
def apply_median_kalman(
    series: pd.Series,
    window_size: int = 5,
    kf_params: Optional[dict] = None,
) -> pd.Series:
    """
    Apply median filter followed by Kalman filter over a pandas Series.

    Args:
        series: Input sensor values
        window_size: Median filter window
        kf_params: Kalman params (process_variance, measurement_variance, estimated_error)
        adaptive: If True, use adaptive Kalman (auto-adjust Q/R)

    Returns:
        Filtered Series
    """
    if kf_params is None:
        kf_params = {
            "process_variance": 0.01,
            "measurement_variance": 0.5,
            "estimated_error": 1.0,
        }

    non_na = series.dropna()
    if non_na.empty:
        return series.copy()

    kf = AdaptiveKalmanFilter(
        process_variance=kf_params.get("process_variance", 0.01),
        measurement_variance=kf_params.get("measurement_variance", 0.5),
        estimated_error=kf_params.get("estimated_error", 1.0),
        initial_value=float(non_na.iloc[0]),
    )

    window: list = []
    filtered = []
    filled = series.ffill().bfill()
    for v in filled.values:
        med = median_filter(window, float(v), window_size)
        filtered.append(kf.update(med))

    return pd.Series(filtered, index=series.index)


# ---------------------------------------------------------------------------
# Run Filtering for All Sensor Columns
# ---------------------------------------------------------------------------
def run_filtering(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply median + adaptive Kalman filtering to each configured sensor column.

    Args:
        df: Validated DataFrame
        config: Pipeline configuration

    Returns:
        DataFrame with filtered columns (<sensor>_filt)
    """
    if df.empty:
        return df.copy()

    filt_cfg = config.get("filtering", {})
    window_size = filt_cfg.get("median_window", 5)
    kf_params = filt_cfg.get("kalman", {})

    result = df.copy()
    for col in sensor_base_columns(config):
        if col not in result.columns:
            continue
        result[f"{col}_filt"] = apply_median_kalman(
            result[col], window_size=window_size, kf_params=kf_params
        )

    logger.info("Filtering applied to %d rows", len(result))
    return result
