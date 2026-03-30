"""
Main Pipeline - Orchestrator
----------------------------
Orchestrates all services sequentially:
  Ingestion -> Validation -> Filtering -> Feature Engineering -> Model -> Drift -> Output -> Alerts
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.ingestion_service import run_ingestion
from services.ingestion_service import load_master_data
from services.validation_service import run_validation
from services.filtering_service import run_filtering
from services.feature_service import run_feature_engineering
from services.seasonal_decomposition_service import run_seasonal_decomposition
from services.prediction_service import run_prediction_service
from services.model_service import run_model_service
from services.drift_service import run_drift_service
from services.alert_service import run_alert_service

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
OUTPUT_DIR = PROJECT_ROOT / "output"
METRICS_SNAPSHOT_PATH = OUTPUT_DIR / "metrics_snapshot.json"
INGESTION_STATE_PATH = OUTPUT_DIR / "ingestion_state.json"

# Fallback configuration used when PyYAML is not installed.
# This lets the app run in environments where installing `tensorflow`/other
# heavy deps (from requirements.txt) would be problematic.
DEFAULT_CONFIG = {
    "sensor_bounds": {
        "temperature": {"min": -40.0, "max": 60.0},
        "humidity": {"min": 0.0, "max": 100.0},
        "pressure": {"min": 800.0, "max": 1100.0},
    },
    "feature_engineering": {"rolling_window": 5},
    "filtering": {
        "median_window": 5,
        "kalman": {
            "process_variance": 0.01,
            "measurement_variance": 0.5,
            "estimated_error": 1.0,
        },
    },
    "model": {
        "isolation_forest": {"contamination": 0.04, "random_state": 42},
        "z_score": {"threshold": 3.0},
        "seasonal_decomposition": {"period": 24, "model": "additive"},
    },
    "prediction": {
        "horizon": 5,
        "sarima_order": [1, 1, 1],
        "seasonal_order": [1, 1, 1, 24],
        "lstm": {"epochs": 10, "batch_size": 32, "lookback": 20},
        "ensemble_weights": {"sarima": 0.5, "lstm": 0.5},
        "forecast_error_threshold": 2.5,
        "error_window": 20,
    },
    "drift": {
        "p_value_threshold": 0.05,
        "baseline_window": 100,
        "comparison_window": 50,
    },
    "paths": {
        "input_stream": "data/stream",
        "master_csv": "data/master_sensor_data.csv",
        "output_csv": "output/anomaly_results.csv",
        "invalid_log": "logs/invalid_rows.log",
    },
    "alerts": {"slack_webhook_url": ""},
}


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    try:
        import yaml  # optional dependency

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        logger.warning(
            "PyYAML not installed; using DEFAULT_CONFIG instead of %s.", config_path
        )
        return DEFAULT_CONFIG


def save_output(df: pd.DataFrame, config: dict) -> str:
    """Save anomaly results to CSV."""
    paths = config.get("paths", {})
    out_path = paths.get("output_csv", "output/anomaly_results.csv")
    full_path = PROJECT_ROOT / out_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=False)
    logger.info("Output saved to %s", full_path)
    return str(full_path)


def _read_json(path: Path) -> dict:
    """Read a small JSON file if it exists."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return {}


def _write_json(path: Path, payload: dict) -> None:
    """Persist a small JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def write_metrics_snapshot(df: pd.DataFrame, project_root: Path = PROJECT_ROOT) -> dict:
    """Write a shared metrics snapshot for Prometheus to consume."""
    output_dir = project_root / "output"
    metrics_path = output_dir / "metrics_snapshot.json"
    ingestion_path = output_dir / "ingestion_state.json"

    existing = _read_json(metrics_path)
    ingestion_state = _read_json(ingestion_path)

    if "anomaly_combined" in df.columns:
        anomaly_count = int((df["anomaly_combined"] == 1).sum())
    elif "anomaly" in df.columns:
        anomaly_count = int((df["anomaly"] == -1).sum())
    else:
        anomaly_count = 0

    payload = {
        "pipeline_runs_total_increment": 1,
        "records_total": int(len(df)),
        "anomalies": anomaly_count,
        "new_records_last_run": int(ingestion_state.get("new_records_last_run", 0)),
        "pipeline_runs_total": int(existing.get("pipeline_runs_total", 0)) + 1,
    }
    _write_json(metrics_path, payload)
    logger.info("Metrics snapshot updated at %s", metrics_path)
    return payload


def run_pipeline(
    config: dict = None,
    refresh_ingestion: bool = True,
    fast_mode: bool = False,
) -> pd.DataFrame:
    """
    Run the full atmospheric monitoring pipeline.

    Args:
        config: Optional config dict; if None, loads from config.yaml
        refresh_ingestion: If False, prefer existing master data and skip live
            ingestion unless no local data exists.
        fast_mode: If True, use faster approximations for interactive runs.

    Returns:
        Final DataFrame with anomaly results
    """
    if config is None:
        config = load_config()

    base_path = str(PROJECT_ROOT)

    # 1. Ingestion / Load Existing Data
    logger.info("=== Step 1: Ingestion ===")
    df = pd.DataFrame()
    master_path = config.get("paths", {}).get("master_csv", "data/master_sensor_data.csv")

    if not refresh_ingestion:
        df = load_master_data(master_path, base_path)
        if not df.empty:
            logger.info("Loaded %d rows from master CSV (refresh disabled)", len(df))

    if df.empty:
        df = run_ingestion(config, base_path)

    # If no stream data, try loading from master CSV as fallback
    if df.empty:
        master_full = PROJECT_ROOT / master_path
        if master_full.exists():
            df = pd.read_csv(master_full)
            df.columns = df.columns.str.strip().str.lower()
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            logger.info("Loaded %d rows from master CSV (no stream data)", len(df))
        else:
            # Generate sample data for demo
            import numpy as np
            from datetime import datetime, timedelta
            n = 200
            base_time = datetime(2025, 1, 1)
            df = pd.DataFrame({
                "timestamp": [base_time + timedelta(hours=i) for i in range(n)],
                "temperature": np.random.normal(25, 2, n),
                "humidity": np.random.normal(60, 5, n),
                "pressure": np.random.normal(1013, 3, n),
            })
            logger.info("No input data; using %d rows of sample data", n)

    if df.empty:
        logger.warning("Pipeline aborted: no data")
        return pd.DataFrame()

    # 2. Validation
    logger.info("=== Step 2: Validation ===")
    df = run_validation(df, config)
    if df.empty:
        logger.warning("Pipeline aborted: no valid rows after validation")
        return pd.DataFrame()

    # 3. Filtering
    logger.info("=== Step 3: Filtering ===")
    df = run_filtering(df, config)

    # 4. Feature Engineering
    logger.info("=== Step 4: Feature Engineering ===")
    df = run_feature_engineering(df, config)
    if df.empty:
        logger.warning("Pipeline aborted: no rows after feature engineering")
        return pd.DataFrame()

    # 5. Seasonal Decomposition
    logger.info("=== Step 5: Seasonal Decomposition ===")
    df = run_seasonal_decomposition(df, config)

    # 6. Hybrid Prediction
    logger.info("=== Step 6: Hybrid Prediction ===")
    df = run_prediction_service(df, config, PROJECT_ROOT, fast_mode=fast_mode)

    # 7. Model (Hybrid Anomaly Detection)
    logger.info("=== Step 7: Model Service ===")
    df = run_model_service(df, config)

    # 8. Drift Detection
    logger.info("=== Step 8: Drift Detection ===")
    df = run_drift_service(df, config, PROJECT_ROOT)

    # 9. Save Output
    logger.info("=== Step 9: Save Output ===")
    save_output(df, config)
    write_metrics_snapshot(df, PROJECT_ROOT)

    # 10. Alerts
    logger.info("=== Step 10: Alert Service ===")
    run_alert_service(df, config)

    logger.info("=== Pipeline Complete ===")
    return df


if __name__ == "__main__":
    run_pipeline()
