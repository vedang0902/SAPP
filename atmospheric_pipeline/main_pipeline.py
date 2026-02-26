"""
Main Pipeline - Orchestrator
----------------------------
Orchestrates all services sequentially:
  Ingestion -> Validation -> Filtering -> Feature Engineering -> Model -> Drift -> Output -> Alerts
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.ingestion_service import run_ingestion
from services.validation_service import run_validation
from services.filtering_service import run_filtering
from services.feature_service import run_feature_engineering
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


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_output(df: pd.DataFrame, config: dict) -> str:
    """Save anomaly results to CSV."""
    paths = config.get("paths", {})
    out_path = paths.get("output_csv", "output/anomaly_results.csv")
    full_path = PROJECT_ROOT / out_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=False)
    logger.info("Output saved to %s", full_path)
    return str(full_path)


def run_pipeline(config: dict = None) -> pd.DataFrame:
    """
    Run the full atmospheric monitoring pipeline.

    Args:
        config: Optional config dict; if None, loads from config.yaml

    Returns:
        Final DataFrame with anomaly results
    """
    if config is None:
        config = load_config()

    base_path = str(PROJECT_ROOT)

    # 1. Ingestion
    logger.info("=== Step 1: Ingestion ===")
    df = run_ingestion(config, base_path)

    # If no stream data, try loading from master CSV as fallback
    if df.empty:
        master_path = config.get("paths", {}).get("master_csv", "data/master_sensor_data.csv")
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

    # 5. Model (Anomaly Detection)
    logger.info("=== Step 5: Model Service ===")
    df = run_model_service(df, config)

    # 6. Drift Detection
    logger.info("=== Step 6: Drift Detection ===")
    df = run_drift_service(df, config)

    # 7. Save Output
    logger.info("=== Step 7: Save Output ===")
    save_output(df, config)

    # 8. Alerts
    logger.info("=== Step 8: Alert Service ===")
    run_alert_service(df, config)

    logger.info("=== Pipeline Complete ===")
    return df


if __name__ == "__main__":
    run_pipeline()
