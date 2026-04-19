"""
Prediction Service - Hybrid SARIMA + LSTM Forecasting
-----------------------------------------------------
Implements hybrid ensemble forecasting for atmospheric sensor streams:
- SARIMA: statistical seasonal model via statsmodels SARIMAX
- LSTM: neural network for multivariate time-series
- Ensemble: weighted average of both predictions
- Forecast error modeling: rolling mean/std for anomaly detection
- Retraining triggered by drift detection flag
"""

import logging
import os
import time
from pathlib import Path

import json
import numpy as np
import pandas as pd

from services.pipeline_schema import (
    capped_sarima_seasonal_m,
    prediction_primary_base,
    resolve_sensor_data_columns,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------
DRIFT_FLAG_PATH = Path("logs/drift_triggered.flag")
MODELS_DIR = Path("models")
MANIFEST_NAME = "training_manifest.json"


def _get_project_root() -> Path:
    """Resolve project root for model/flag paths."""
    return Path(__file__).resolve().parent.parent


def _check_drift_triggered(project_root: Path) -> bool:
    """Check if drift was detected (triggers model retraining)."""
    flag_path = project_root / DRIFT_FLAG_PATH
    return flag_path.exists()


def _prediction_sensor_bases(sensor_cols: list[str]) -> list[str]:
    return [c.replace("_filt", "") for c in sensor_cols]


def _remove_cached_forecast_models(models_dir: Path) -> None:
    for name in ("lstm_model.keras", "lstm_scaler.json", "sarima_fitted.pkl", MANIFEST_NAME):
        p = models_dir / name
        try:
            p.unlink()
        except OSError:
            pass


def _forecast_models_stale(
    models_dir: Path,
    sensor_cols: list[str],
    primary_base: str,
) -> bool:
    """True if saved LSTM/SARIMA artifacts do not match current sensor schema."""
    expected_bases = _prediction_sensor_bases(sensor_cols)
    manifest_path = models_dir / MANIFEST_NAME
    lstm_path = models_dir / "lstm_model.keras"

    if lstm_path.exists():
        mdl = _load_lstm_model(lstm_path)
        if mdl is None:
            return True
        sh = getattr(mdl, "input_shape", None)
        if sh is None or len(sh) < 3:
            return True
        if sh[-1] != len(sensor_cols):
            logger.info(
                "LSTM input features mismatch (saved=%s current=%s); retraining",
                sh[-1],
                len(sensor_cols),
            )
            return True

    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            if m.get("sensor_columns") != expected_bases or m.get("primary_sensor") != primary_base:
                return True
        except Exception:
            return True

    return False


def _write_training_manifest(
    models_dir: Path,
    sensor_cols: list[str],
    primary_base: str,
) -> None:
    payload = {
        "sensor_columns": _prediction_sensor_bases(sensor_cols),
        "primary_sensor": primary_base,
    }
    path = models_dir / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _clear_drift_flag(project_root: Path) -> None:
    """Clear drift trigger flag after retraining."""
    flag_path = project_root / DRIFT_FLAG_PATH
    if flag_path.exists():
        try:
            flag_path.unlink()
            logger.info("Cleared drift trigger flag")
        except OSError as e:
            logger.warning("Could not clear drift flag: %s", e)


# ---------------------------------------------------------------------------
# SARIMA Forecasting
# ---------------------------------------------------------------------------
def _fit_sarima(
    series: pd.Series,
    order: tuple,
    seasonal_order: tuple,
) -> object:
    """Fit SARIMAX model on series."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as e:
        logger.error("statsmodels required for SARIMA: %s", e)
        return None

    series_clean = series.dropna()
    if len(series_clean) < 30:
        logger.warning("Insufficient data for SARIMA (need >= 30 points)")
        return None

    try:
        model = SARIMAX(
            series_clean,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=100)
        return fitted
    except Exception as e:
        logger.warning("SARIMA fit failed: %s", e)
        return None


def _forecast_sarima(fitted: object, steps: int) -> tuple:
    """Generate SARIMA forecast and confidence intervals."""
    if fitted is None:
        return None, None, None
    try:
        fcast = fitted.get_forecast(steps=steps)
        pred = fcast.predicted_mean
        ci = fcast.conf_int()
        return pred, ci.iloc[:, 0].values, ci.iloc[:, 1].values
    except Exception as e:
        logger.warning("SARIMA forecast failed: %s", e)
        return None, None, None


# ---------------------------------------------------------------------------
# LSTM Forecasting
# ---------------------------------------------------------------------------
def _build_lstm_model(
    lookback: int,
    n_features: int,
    lstm_units: int = 32,
    dropout: float = 0.2,
    l2_reg: float = 1e-4,
) -> object:
    """
    Build lightweight LSTM model with dropout and L2 regularization to avoid overfitting.
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.regularizers import l2
    except ImportError as e:
        logger.error("TensorFlow required for LSTM: %s", e)
        return None

    model = Sequential([
        LSTM(
            lstm_units,
            input_shape=(lookback, n_features),
            return_sequences=False,
            kernel_regularizer=l2(l2_reg),
        ),
        Dropout(dropout),
        Dense(16, activation="relu", kernel_regularizer=l2(l2_reg)),
        Dropout(dropout),
        Dense(n_features),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _create_sequences(
    data: np.ndarray,
    lookback: int,
    forecast_horizon: int,
) -> tuple:
    """Create sliding window sequences for LSTM."""
    X, y = [], []
    for i in range(lookback, len(data) - forecast_horizon + 1):
        X.append(data[i - lookback:i])
        y.append(data[i : i + forecast_horizon])
    return np.array(X), np.array(y)


def _scale_minmax(data: np.ndarray) -> tuple:
    """Apply MinMax scaling. Returns scaled data and (min, max) for inverse."""
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-6
    scaled = (data - min_vals) / range_vals
    return scaled, min_vals, max_vals


def _inv_scale(scaled: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    """Inverse MinMax scaling."""
    return scaled * (max_vals - min_vals) + min_vals


def _fit_lstm(
    df: pd.DataFrame,
    sensor_cols: list,
    lookback: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    model_path: Path,
) -> tuple:
    """Train LSTM on multivariate sensor data. Returns (model, scaler_params)."""
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        return None, None

    data = df[sensor_cols].values.astype(np.float32)
    if len(data) < lookback + horizon:
        logger.warning("Insufficient data for LSTM (need >= lookback + horizon)")
        return None, None

    scaled, min_vals, max_vals = _scale_minmax(data)
    X, y = _create_sequences(scaled, lookback, horizon)
    if len(X) < 10:
        return None, None

    # Use last step of horizon for single-step forecast alignment
    y_single = y[:, -1, :]  # Predict last step of horizon
    n_features = len(sensor_cols)

    model = _build_lstm_model(lookback, n_features)
    if model is None:
        return None, None

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=0,
    )

    start = time.perf_counter()
    model.fit(
        X, y_single,
        epochs=epochs,
        batch_size=min(batch_size, len(X)),
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0,
    )
    train_time = time.perf_counter() - start
    logger.info("LSTM trained in %.2f s", train_time)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    scaler_params = {"min": min_vals.tolist(), "max": max_vals.tolist()}
    return model, scaler_params


def _load_lstm_model(model_path: Path) -> object:
    """Load a persisted LSTM model once for reuse."""
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        return None

    if not model_path.exists():
        return None

    try:
        return load_model(str(model_path))
    except Exception as e:
        logger.warning("Could not load LSTM model: %s", e)
        return None


def _forecast_lstm_window(
    window_df: pd.DataFrame,
    sensor_cols: list,
    lstm_model: object,
    scaler_params: dict,
    lookback: int,
) -> np.ndarray:
    """Generate LSTM forecast for a single rolling window using a loaded model."""
    if lstm_model is None or scaler_params is None:
        return None

    data = window_df[sensor_cols].values.astype(np.float32)
    if len(data) < lookback:
        return None

    min_vals = np.array(scaler_params["min"], dtype=np.float32)
    max_vals = np.array(scaler_params["max"], dtype=np.float32)
    last_window = data[-lookback:]
    scaled = (last_window - min_vals) / (max_vals - min_vals + 1e-6)
    X = np.expand_dims(scaled, axis=0)

    pred_scaled = lstm_model.predict(X, verbose=0)
    pred = _inv_scale(pred_scaled, min_vals, max_vals)
    return pred[0]


def _fast_window_baseline(window_df: pd.DataFrame, sensor_cols: list, tail_size: int = 3) -> np.ndarray:
    """
    Build a cheap per-sensor baseline forecast from the latest observed values.

    Using a tiny rolling mean keeps fast_mode responsive for frequent DAG runs
    while still producing separate predictions for each sensor.
    """
    preds = []
    for col in sensor_cols:
        series = pd.to_numeric(window_df[col], errors="coerce").dropna()
        if series.empty:
            preds.append(np.nan)
            continue
        preds.append(float(series.tail(tail_size).mean()))
    return np.array(preds, dtype=np.float32)


# ---------------------------------------------------------------------------
# Hybrid Ensemble and Forecast Error
# ---------------------------------------------------------------------------
def _ensemble_predict(
    sarima_pred: float,
    lstm_pred: np.ndarray,
    sensor_idx: int,
    weights: dict,
) -> float:
    """Combine SARIMA and LSTM predictions via weighted average."""
    w_sarima = weights.get("sarima", 0.5)
    w_lstm = weights.get("lstm", 0.5)
    if sarima_pred is None or np.isnan(sarima_pred):
        return float(lstm_pred[sensor_idx]) if lstm_pred is not None else np.nan
    if lstm_pred is None or np.any(np.isnan(lstm_pred)):
        return float(sarima_pred)
    return w_sarima * sarima_pred + w_lstm * float(lstm_pred[sensor_idx])


def _compute_forecast_errors(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    error_window: int = 20,
) -> pd.DataFrame:
    """Compute forecast error, rolling mean, rolling std."""
    result = df.copy()
    result["forecast_error"] = np.abs(result[actual_col] - result[pred_col])
    result["forecast_error_mean"] = result["forecast_error"].rolling(error_window, min_periods=1).mean()
    result["forecast_error_std"] = result["forecast_error"].rolling(error_window, min_periods=1).std().fillna(0)
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_prediction_service(
    df: pd.DataFrame,
    config: dict,
    project_root: Path = None,
    fast_mode: bool = False,
) -> pd.DataFrame:
    """
    Run hybrid SARIMA + LSTM prediction, add forecasts and forecast errors.

    Hybrid Forecasting Approach:
    ---------------------------
    SARIMA captures seasonal/trend structure; LSTM captures nonlinear patterns.
    Ensemble combines both via configurable weights to reduce variance.

    Training Strategy:
    ------------------
    - Train on initial historical window (rolling)
    - Retrain when drift flag exists (logs/drift_triggered.flag)
    - Log model retraining events

    Args:
        df: DataFrame with sensor columns (see sensors.columns; *_filt preferred)
        config: Pipeline configuration
        project_root: Project root for model/flag paths
        fast_mode: When True, prioritize responsiveness over exhaustive rolling
            prediction by limiting forecast rows and skipping LSTM inference.

    Returns:
        DataFrame with forecast_* columns and forecast_error columns
    """
    if df.empty or len(df) < 30:
        logger.warning("Insufficient data for prediction (need >= 30 rows)")
        return df.copy()

    if project_root is None:
        project_root = _get_project_root()

    pred_cfg = config.get("prediction", {})
    horizon = pred_cfg.get("horizon", 5)
    sarima_order = tuple(pred_cfg.get("sarima_order", [1, 1, 1]))
    seasonal_order = tuple(pred_cfg.get("seasonal_order", [1, 1, 1, 24]))
    if len(seasonal_order) == 4:
        m_cfg = int(seasonal_order[3])
        min_seasons = int(pred_cfg.get("sarima_min_seasons", 10))
        m_eff = capped_sarima_seasonal_m(m_cfg, len(df), min_seasons=min_seasons)
        if m_eff != m_cfg:
            logger.info(
                "SARIMA: capped seasonal m %s -> %s (rows=%s, min_seasons=%s)",
                m_cfg,
                m_eff,
                len(df),
                min_seasons,
            )
        seasonal_order = (seasonal_order[0], seasonal_order[1], seasonal_order[2], m_eff)
    lstm_cfg = pred_cfg.get("lstm", {})
    epochs = lstm_cfg.get("epochs", 10)
    batch_size = lstm_cfg.get("batch_size", 32)
    lookback = lstm_cfg.get("lookback", 20)
    weights = pred_cfg.get("ensemble_weights", {"sarima": 0.5, "lstm": 0.5})
    history_limit = int(pred_cfg.get("history_limit", 240))
    max_fast_forecast_rows = int(pred_cfg.get("max_fast_forecast_rows", 96))

    sensor_cols = resolve_sensor_data_columns(df, config)
    if not sensor_cols:
        logger.warning("No sensor columns for prediction")
        return df.copy()

    primary_base = prediction_primary_base(config)
    primary_col = next(
        (c for c in sensor_cols if c.replace("_filt", "") == primary_base),
        sensor_cols[0],
    )

    models_dir = project_root / MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    schema_stale = _forecast_models_stale(models_dir, sensor_cols, primary_base)
    if schema_stale:
        logger.info("Forecast model cache incompatible with current sensors; clearing cached models")
        _remove_cached_forecast_models(models_dir)

    drift_retrain = _check_drift_triggered(project_root)
    use_lstm = not fast_mode
    if drift_retrain:
        logger.info("Drift trigger detected - retraining prediction models")
        _clear_drift_flag(project_root)
    if fast_mode:
        drift_retrain = False

    train_forecasters = drift_retrain or schema_stale
    train_sarima = train_forecasters or not (models_dir / "sarima_fitted.pkl").exists()
    train_lstm = use_lstm and (train_forecasters or not (models_dir / "lstm_model.keras").exists())

    total_training_time = 0.0

    # SARIMA: univariate on configured primary sensor
    sarima_fitted = None

    if train_sarima:
        t0 = time.perf_counter()
        sarima_fitted = _fit_sarima(df[primary_col], sarima_order, seasonal_order)
        if sarima_fitted is not None:
            try:
                import joblib
                joblib.dump(sarima_fitted, models_dir / "sarima_fitted.pkl")
            except Exception as e:
                logger.warning("Could not save SARIMA: %s", e)
        total_training_time += time.perf_counter() - t0
        logger.info("SARIMA training time: %.2f s", time.perf_counter() - t0)
    else:
        try:
            import joblib
            sarima_fitted = joblib.load(models_dir / "sarima_fitted.pkl")
        except Exception as e:
            logger.warning("Could not load SARIMA: %s", e)
            sarima_fitted = _fit_sarima(df[primary_col], sarima_order, seasonal_order)

    # LSTM: multivariate
    lstm_path = models_dir / "lstm_model.keras"
    scaler_params = None
    if (models_dir / "lstm_scaler.json").exists():
        try:
            with open(models_dir / "lstm_scaler.json", encoding="utf-8") as f:
                scaler_params = json.load(f)
        except Exception:
            pass

    lstm_model = None
    if train_lstm:
        t0 = time.perf_counter()
        lstm_model, scaler_params = _fit_lstm(
            df, sensor_cols, lookback, horizon, epochs, batch_size, lstm_path
        )
        total_training_time += time.perf_counter() - t0
        if scaler_params is not None:
            try:
                with open(models_dir / "lstm_scaler.json", "w") as f:
                    json.dump(scaler_params, f)
            except Exception as e:
                logger.warning("Could not save LSTM scaler: %s", e)
        logger.info("LSTM training time: %.2f s", time.perf_counter() - t0)
    elif use_lstm:
        lstm_model = _load_lstm_model(lstm_path)

    if use_lstm and train_lstm and lstm_path.exists():
        lstm_model = _load_lstm_model(lstm_path)

    # Rolling forecast: refit SARIMA every REFIT_INTERVAL steps for accuracy/speed balance
    REFIT_INTERVAL = int(pred_cfg.get("sarima_refit_interval", 25))
    if fast_mode:
        REFIT_INTERVAL = 0
    result = df.copy()
    n = len(result)
    pred_columns = [f"forecast_{c.replace('_filt', '')}" for c in sensor_cols]

    for col in pred_columns:
        result[col] = np.nan

    train_size = max(lookback + horizon, 2 * (seasonal_order[-1] if seasonal_order else 24))
    train_size = min(train_size, n - 1)
    forecast_start = train_size
    if fast_mode:
        forecast_start = max(train_size, n - max_fast_forecast_rows)
    else:
        # Full mode defaults to evaluating only the last N rows here; each step runs
        # SARIMA forecast (+ LSTM inference). A full O(n) pass over thousands of rows
        # looks "hung" for minutes. Use 0 for unlimited historical rolling forecasts.
        tail_cfg = pred_cfg.get("rolling_forecast_tail_rows", 600)
        try:
            tail_n = int(tail_cfg)
        except (TypeError, ValueError):
            tail_n = 600
        if tail_n > 0:
            forecast_start = max(forecast_start, n - tail_n)

    n_steps = max(0, n - forecast_start)
    logger.info(
        "Rolling hybrid forecast: %d row-indices (%d..%d); refit_interval=%s; LSTM=%s",
        n_steps,
        forecast_start,
        n - 1,
        REFIT_INTERVAL if not fast_mode else 0,
        lstm_model is not None,
    )

    _loop_log_every = 800
    for i in range(forecast_start, n):
        if (i - forecast_start) > 0 and (i - forecast_start) % _loop_log_every == 0:
            logger.info(
                "Prediction progress: %d / %d rows",
                i - forecast_start,
                n_steps,
            )
        window_start = max(0, i - history_limit)
        window = result.iloc[window_start:i]
        baseline_pred = _fast_window_baseline(window, sensor_cols) if fast_mode else None
        # SARIMA: refit periodically for rolling window accuracy
        sarima_pred_val = None
        if sarima_fitted is not None:
            try:
                pred, _, _ = _forecast_sarima(sarima_fitted, 1)
                if pred is not None and len(pred) > 0:
                    sarima_pred_val = float(pred.iloc[0])
            except Exception:
                pass
            if REFIT_INTERVAL > 0 and (i - forecast_start) % REFIT_INTERVAL == 0 and i > forecast_start:
                try:
                    sarima_fitted = _fit_sarima(
                        window[primary_col].iloc[-150:], sarima_order, seasonal_order
                    )
                except Exception:
                    pass

        # LSTM: sliding window prediction (no refit in loop)
        lstm_pred = None
        if lstm_model is not None and scaler_params:
            lstm_pred = _forecast_lstm_window(window, sensor_cols, lstm_model, scaler_params, lookback)

        for j, col in enumerate(pred_columns):
            if fast_mode and baseline_pred is not None and not np.isnan(baseline_pred[j]):
                pred_val = float(baseline_pred[j])
                if j == 0 and sarima_pred_val is not None and not np.isnan(sarima_pred_val):
                    pred_val = _ensemble_predict(sarima_pred_val, baseline_pred, j, weights)
            else:
                pred_val = _ensemble_predict(sarima_pred_val, lstm_pred, j, weights)
            if not np.isnan(pred_val):
                result.loc[result.index[i], col] = pred_val

    if not fast_mode:
        for col in pred_columns:
            result[col] = result[col].ffill().bfill()

    # Forecast error: use primary sensor for anomaly
    primary_idx = next(
        (j for j, c in enumerate(sensor_cols) if c.replace("_filt", "") == primary_base),
        0,
    )
    pred_col = pred_columns[primary_idx]
    actual_col = sensor_cols[primary_idx]
    error_window = pred_cfg.get("error_window", 20)
    if result[pred_col].notna().any():
        result = _compute_forecast_errors(result, pred_col, actual_col, error_window=error_window)
    else:
        result["forecast_error"] = np.nan
        result["forecast_error_mean"] = np.nan
        result["forecast_error_std"] = np.nan
        logger.warning("No valid forecasts; forecast_error columns set to NaN")

    # Metrics
    valid = result[pred_col].notna() & result[actual_col].notna()
    if valid.sum() > 0:
        diff = result.loc[valid, actual_col] - result.loc[valid, pred_col]
        rmse = np.sqrt((diff ** 2).mean())
        mae = np.abs(diff).mean()
        fcast_var = result.loc[valid, pred_col].var()
        logger.info("Prediction metrics: RMSE=%.4f, MAE=%.4f, Forecast variance=%.4f, Training=%.2fs", rmse, mae, fcast_var, total_training_time)
        _log_metrics(rmse, mae, fcast_var, total_training_time)

    if (models_dir / "sarima_fitted.pkl").exists():
        _write_training_manifest(models_dir, sensor_cols, primary_base)

    return result


def _log_metrics(rmse: float, mae: float, fcast_var: float, training_time: float = 0.0) -> None:
    """Expose metrics for Prometheus scraping."""
    try:
        from prometheus_client import Gauge
        gauges = getattr(_log_metrics, "_gauges", None)
        if gauges is None:
            gauges = {
                "rmse": Gauge("atmospheric_forecast_rmse", "Forecast RMSE"),
                "mae": Gauge("atmospheric_forecast_mae", "Forecast MAE"),
                "variance": Gauge("atmospheric_forecast_variance", "Forecast variance"),
                "training_time": Gauge("atmospheric_model_training_seconds", "Model training time (s)"),
            }
            _log_metrics._gauges = gauges
        gauges["rmse"].set(rmse)
        gauges["mae"].set(mae)
        gauges["variance"].set(fcast_var)
        gauges["training_time"].set(training_time)
    except ImportError:
        pass
