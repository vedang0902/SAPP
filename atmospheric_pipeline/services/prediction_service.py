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
import threading
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
    *,
    maxiter: int = 100,
    disp: bool = False,
    heartbeat_seconds: float = 0.0,
    log_label: str = "SARIMAX",
) -> object:
    """Fit SARIMAX model on series. Optional heartbeat logs while statsmodels optimizes."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as e:
        logger.error("statsmodels required for SARIMA: %s", e)
        return None

    series_clean = series.dropna()
    n = len(series_clean)
    if n < 30:
        logger.warning("Insufficient data for SARIMA (need >= 30 points)")
        return None

    logger.info(
        "%s fit starting: n=%d order=%s seasonal_order=%s maxiter=%s disp=%s",
        log_label,
        n,
        order,
        seasonal_order,
        maxiter,
        disp,
    )
    t_start = time.perf_counter()
    stop_hb = threading.Event()

    def _heartbeat() -> None:
        interval = max(15.0, float(heartbeat_seconds))
        while not stop_hb.wait(timeout=interval):
            logger.info(
                "%s: optimizer still running (%.0f s elapsed, n=%d)",
                log_label,
                time.perf_counter() - t_start,
                n,
            )

    hb_thread = None
    if heartbeat_seconds and float(heartbeat_seconds) > 0:
        hb_thread = threading.Thread(
            target=_heartbeat,
            name="sarima-heartbeat",
            daemon=True,
        )
        hb_thread.start()

    fitted = None
    try:
        model = SARIMAX(
            series_clean,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=disp, maxiter=int(maxiter))
    except Exception as e:
        logger.warning("%s fit failed after %.1f s: %s", log_label, time.perf_counter() - t_start, e)
    finally:
        stop_hb.set()

    elapsed = time.perf_counter() - t_start
    if fitted is not None:
        summary_bits = []
        try:
            ret = getattr(fitted, "mle_retvals", None)
            if isinstance(ret, dict):
                for key in ("iterations", "fcalls", "warnflag"):
                    if key in ret:
                        summary_bits.append(f"{key}={ret[key]}")
        except Exception:
            pass
        tail = (" " + ", ".join(summary_bits)) if summary_bits else ""
        logger.info("%s fit finished in %.1f s%s", log_label, elapsed, tail)
    return fitted


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
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
        from tensorflow.keras.regularizers import l2
    except ImportError as e:
        logger.error("TensorFlow required for LSTM: %s", e)
        return None

    model = Sequential([
        Input(shape=(lookback, n_features)),
        LSTM(
            lstm_units,
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
    patience: int = 3,
) -> tuple:
    """Train LSTM on multivariate sensor data. Returns (model, scaler_params)."""
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as e:
        logger.warning("LSTM disabled: TensorFlow import failed (%s)", e)
        return None, None

    data = df[sensor_cols].values.astype(np.float32)
    if len(data) < lookback + horizon:
        logger.warning("Insufficient data for LSTM (need >= lookback + horizon)")
        return None, None

    scaled, min_vals, max_vals = _scale_minmax(data)
    X, y = _create_sequences(scaled, lookback, horizon)
    if len(X) < 10:
        logger.warning(
            "LSTM disabled: insufficient sequences after windowing (len(X)=%d, need>=10)",
            len(X),
        )
        return None, None

    # Use last step of horizon for single-step forecast alignment
    y_single = y[:, -1, :]  # Predict last step of horizon
    n_features = len(sensor_cols)

    model = _build_lstm_model(lookback, n_features)
    if model is None:
        logger.warning("LSTM disabled: model build failed")
        return None, None

    logger.info(
        "LSTM training starting: train_rows=%d sequences=%d n_features=%d epochs=%d batch=%d lookback=%d horizon=%d",
        len(data),
        len(X),
        n_features,
        epochs,
        min(batch_size, len(X)),
        lookback,
        horizon,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=max(1, int(patience)),
        restore_best_weights=True,
        verbose=0,
    )

    start = time.perf_counter()
    try:
        model.fit(
            X, y_single,
            epochs=epochs,
            batch_size=min(batch_size, len(X)),
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0,
        )
    except Exception as e:
        logger.warning("LSTM training failed: %s", e)
        return None, None
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
def _forecast_mode_label(use_sarima: bool, use_lstm: bool) -> str:
    if use_sarima and use_lstm:
        return "hybrid"
    if use_sarima:
        return "SARIMA-only"
    if use_lstm:
        return "LSTM-only"
    return "none"


def _ensemble_predict(
    sarima_pred: float,
    lstm_pred: np.ndarray,
    sensor_idx: int,
    weights: dict,
    weight_keys: tuple[str, str] = ("sarima", "lstm"),
) -> float:
    """Combine SARIMA and LSTM predictions via weighted average."""
    w_sarima = weights.get(weight_keys[0], weights.get("sarima", 0.5))
    w_lstm = weights.get(weight_keys[1], weights.get("lstm", 0.5))
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
    use_sarima = bool(pred_cfg.get("use_sarima", True))
    use_lstm_cfg = bool(pred_cfg.get("use_lstm", pred_cfg.get("train_lstm", True)))
    if not use_sarima and not use_lstm_cfg:
        logger.error(
            "Prediction disabled: set at least one of prediction.use_sarima or prediction.use_lstm to true"
        )
        return df.copy()
    horizon = pred_cfg.get("horizon", 5)
    sarima_order = tuple(pred_cfg.get("sarima_order", [1, 1, 1]))
    seasonal_order = tuple(pred_cfg.get("seasonal_order", [1, 1, 1, 24]))
    if use_sarima and len(seasonal_order) == 4:
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
    lstm_patience = int(lstm_cfg.get("patience", 3))
    weights = pred_cfg.get("ensemble_weights", {"sarima": 0.5, "lstm": 0.5})
    split_weights_by_primary = bool(pred_cfg.get("split_weights_by_primary", True))
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
    mode_label = _forecast_mode_label(use_sarima, use_lstm_cfg)
    logger.info(
        "%s prediction: n_rows=%d sensor_cols=%s primary=%s fast_mode=%s",
        mode_label,
        len(df),
        sensor_cols,
        primary_col,
        fast_mode,
    )

    models_dir = project_root / MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    schema_stale = _forecast_models_stale(models_dir, sensor_cols, primary_base)
    if schema_stale:
        logger.info("Forecast model cache incompatible with current sensors; clearing cached models")
        _remove_cached_forecast_models(models_dir)

    drift_retrain = _check_drift_triggered(project_root)
    use_lstm = use_lstm_cfg and (not fast_mode)
    if drift_retrain:
        logger.info("Drift trigger detected - retraining prediction models")
        _clear_drift_flag(project_root)
    if fast_mode:
        drift_retrain = False

    train_forecasters = drift_retrain or schema_stale
    train_sarima = use_sarima and (
        train_forecasters or not (models_dir / "sarima_fitted.pkl").exists()
    )
    train_lstm = use_lstm and (train_forecasters or not (models_dir / "lstm_model.keras").exists())

    total_training_time = 0.0

    sarima_maxiter = int(pred_cfg.get("sarima_maxiter", 100))
    sarima_disp = bool(pred_cfg.get("sarima_disp", False))
    sarima_heartbeat = float(pred_cfg.get("sarima_heartbeat_seconds", 60))
    sarima_refit_maxiter = int(pred_cfg.get("sarima_refit_maxiter", 40))
    sarima_refit_window_rows = int(pred_cfg.get("sarima_refit_window_rows", 360))
    sarima_train_tail_rows = int(pred_cfg.get("sarima_train_tail_rows", 0))

    # SARIMA: univariate on configured primary sensor (skipped when use_sarima=false)
    sarima_fitted = None

    if not use_sarima:
        logger.info("SARIMA disabled (prediction.use_sarima=false)")
    elif train_sarima:
        series_for_fit = df[primary_col]
        if sarima_train_tail_rows > 0:
            series_for_fit = series_for_fit.iloc[-sarima_train_tail_rows:]
        logger.info(
            "Training SARIMA on column %r (train_sarima=True, heartbeat=%ss, tail_rows=%s)",
            primary_col,
            sarima_heartbeat,
            sarima_train_tail_rows if sarima_train_tail_rows > 0 else "all",
        )
        t0 = time.perf_counter()
        sarima_fitted = _fit_sarima(
            series_for_fit,
            sarima_order,
            seasonal_order,
            maxiter=sarima_maxiter,
            disp=sarima_disp,
            heartbeat_seconds=sarima_heartbeat,
            log_label="SARIMAX(initial)",
        )
        if sarima_fitted is not None:
            try:
                import joblib
                joblib.dump(sarima_fitted, models_dir / "sarima_fitted.pkl")
            except Exception as e:
                logger.warning("Could not save SARIMA: %s", e)
        sarima_dt = time.perf_counter() - t0
        total_training_time += sarima_dt
        logger.info("SARIMA step wall time (fit + save): %.2f s", sarima_dt)
    elif use_sarima:
        logger.info("Loading cached SARIMA from %s", models_dir / "sarima_fitted.pkl")
        try:
            import joblib
            sarima_fitted = joblib.load(models_dir / "sarima_fitted.pkl")
        except Exception as e:
            logger.warning("Could not load SARIMA: %s; refitting", e)
            sarima_fitted = _fit_sarima(
                df[primary_col],
                sarima_order,
                seasonal_order,
                maxiter=sarima_maxiter,
                disp=sarima_disp,
                heartbeat_seconds=sarima_heartbeat,
                log_label="SARIMAX(fallback-load)",
            )

    if use_sarima and not use_lstm and sarima_fitted is None:
        logger.error("SARIMA-only mode enabled but SARIMA is unavailable; forecasts will be empty")

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
    if not use_lstm:
        logger.info("LSTM disabled (prediction.use_lstm=false)")
    elif train_lstm:
        logger.info("Training LSTM (train_lstm=True)")
        t0 = time.perf_counter()
        lstm_model, scaler_params = _fit_lstm(
            df, sensor_cols, lookback, horizon, epochs, batch_size, lstm_path, patience=lstm_patience
        )
        lstm_dt = time.perf_counter() - t0
        total_training_time += lstm_dt
        if scaler_params is not None:
            try:
                with open(models_dir / "lstm_scaler.json", "w") as f:
                    json.dump(scaler_params, f)
            except Exception as e:
                logger.warning("Could not save LSTM scaler: %s", e)
        if lstm_model is None:
            if use_sarima:
                logger.warning(
                    "LSTM enabled but unavailable after training attempt; continuing with SARIMA-only forecasts"
                )
            else:
                logger.error("LSTM-only mode enabled but LSTM training failed; forecasts will be empty")
        logger.info("LSTM step wall time (train + save): %.2f s", lstm_dt)
    elif use_lstm:
        logger.info("Loading cached LSTM from %s", lstm_path)
        lstm_model = _load_lstm_model(lstm_path)

    if use_lstm and train_lstm and lstm_path.exists():
        lstm_model = _load_lstm_model(lstm_path)

    # Rolling forecast: refit SARIMA every REFIT_INTERVAL steps for accuracy/speed balance
    REFIT_INTERVAL = int(pred_cfg.get("sarima_refit_interval", 25)) if use_sarima else 0
    if fast_mode:
        REFIT_INTERVAL = 0
    result = df.copy()
    n = len(result)
    pred_columns = [f"forecast_{c.replace('_filt', '')}" for c in sensor_cols]
    primary_idx = next(
        (j for j, c in enumerate(sensor_cols) if c.replace("_filt", "") == primary_base),
        0,
    )

    for col in pred_columns:
        result[col] = np.nan

    if use_sarima and seasonal_order:
        train_size = max(lookback + horizon, 2 * seasonal_order[-1])
    elif use_lstm:
        train_size = lookback + horizon
    else:
        train_size = max(30, lookback + horizon)
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
        "Rolling %s forecast: %d row-indices (%d..%d); refit_interval=%s; LSTM=%s",
        mode_label,
        n_steps,
        forecast_start,
        n - 1,
        REFIT_INTERVAL if not fast_mode else 0,
        lstm_model is not None,
    )

    try:
        _loop_log_every = max(1, int(pred_cfg.get("rolling_log_interval", 100)))
    except (TypeError, ValueError):
        _loop_log_every = 100
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
        if use_sarima and sarima_fitted is not None:
            try:
                pred, _, _ = _forecast_sarima(sarima_fitted, 1)
                if pred is not None and len(pred) > 0:
                    sarima_pred_val = float(pred.iloc[0])
            except Exception:
                pass
            if REFIT_INTERVAL > 0 and (i - forecast_start) % REFIT_INTERVAL == 0 and i > forecast_start:
                try:
                    refit_window = max(30, sarima_refit_window_rows)
                    sarima_fitted = _fit_sarima(
                        window[primary_col].iloc[-refit_window:],
                        sarima_order,
                        seasonal_order,
                        maxiter=sarima_refit_maxiter,
                        disp=False,
                        heartbeat_seconds=0.0,
                        log_label="SARIMAX(refit)",
                    )
                except Exception:
                    pass

        # LSTM: sliding window prediction (no refit in loop)
        lstm_pred = None
        if lstm_model is not None and scaler_params:
            lstm_pred = _forecast_lstm_window(window, sensor_cols, lstm_model, scaler_params, lookback)

        for j, col in enumerate(pred_columns):
            weight_keys = ("sarima", "lstm")
            if split_weights_by_primary and j != primary_idx:
                weight_keys = ("sarima_other", "lstm_other")
            if fast_mode and baseline_pred is not None and not np.isnan(baseline_pred[j]):
                pred_val = float(baseline_pred[j])
                if (
                    use_sarima
                    and use_lstm_cfg
                    and j == 0
                    and sarima_pred_val is not None
                    and not np.isnan(sarima_pred_val)
                ):
                    pred_val = _ensemble_predict(
                        sarima_pred_val,
                        baseline_pred,
                        j,
                        weights,
                        weight_keys=weight_keys,
                    )
            else:
                pred_val = _ensemble_predict(
                    sarima_pred_val,
                    lstm_pred,
                    j,
                    weights,
                    weight_keys=weight_keys,
                )
            if not np.isnan(pred_val):
                result.loc[result.index[i], col] = pred_val

    if not fast_mode:
        for col in pred_columns:
            result[col] = result[col].ffill().bfill()

    # Forecast error: use primary sensor for anomaly
    pred_col = pred_columns[primary_idx]
    actual_col = sensor_cols[primary_idx]
    error_window = pred_cfg.get("error_window", 20)
    warmup_rows = int(pred_cfg.get("forecast_error_warmup_rows", lookback))
    if result[pred_col].notna().any():
        result = _compute_forecast_errors(result, pred_col, actual_col, error_window=error_window)
        if warmup_rows > 0:
            warm_start = max(forecast_start, 0)
            warm_end = min(warm_start + warmup_rows, len(result))
            warm_idx = result.index[warm_start:warm_end]
            if len(warm_idx) > 0:
                result.loc[warm_idx, ["forecast_error", "forecast_error_mean", "forecast_error_std"]] = np.nan
                logger.info(
                    "Forecast-error warmup: masked %d initial rows for anomaly scoring",
                    len(warm_idx),
                )
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

    if (models_dir / "lstm_model.keras").exists() or (models_dir / "sarima_fitted.pkl").exists():
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
