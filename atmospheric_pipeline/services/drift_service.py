"""
Drift Detection Service - Kolmogorov-Smirnov Test
-------------------------------------------------
Compares rolling window distribution vs baseline using KS test.
Logs drift if p-value < threshold (configurable).
Writes drift trigger flag for prediction model retraining.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from services.pipeline_schema import resolve_sensor_data_columns

logger = logging.getLogger(__name__)

DRIFT_FLAG_PATH = Path("logs/drift_triggered.flag")


def load_drift_config(config: dict) -> dict:
    """Extract drift detection parameters from config."""
    drift = config.get("drift", {})
    return {
        "p_value_threshold": drift.get("p_value_threshold", 0.05),
        "baseline_window": drift.get("baseline_window", 100),
        "comparison_window": drift.get("comparison_window", 50),
    }


def ks_drift_test(
    baseline: np.ndarray,
    comparison: np.ndarray,
    p_threshold: float = 0.05,
) -> tuple[bool, float]:
    """
    Perform Kolmogorov-Smirnov test between baseline and comparison samples.

    Args:
        baseline: Baseline distribution sample
        comparison: Current window to compare
        p_threshold: Threshold below which we declare drift

    Returns:
        (drift_detected, p_value)
    """
    if len(baseline) < 10 or len(comparison) < 10:
        return False, 1.0
    try:
        stat, p_value = scipy_stats.ks_2samp(baseline, comparison)
        return p_value < p_threshold, float(p_value)
    except Exception as e:
        logger.warning("KS test failed: %s", e)
        return False, 1.0


def _write_drift_flag(project_root: Path = None) -> None:
    """Write drift trigger flag for prediction model retraining."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    flag_path = project_root / DRIFT_FLAG_PATH
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        flag_path.touch()
        logger.info("Drift trigger flag written; prediction models will retrain on next run")
    except OSError as e:
        logger.warning("Could not write drift flag: %s", e)


def run_drift_detection(
    df: pd.DataFrame,
    config: dict,
    project_root: Path = None,
) -> pd.DataFrame:
    """
    Run drift detection on sensor columns. Log drift events.
    When drift is detected, writes flag to trigger prediction model retraining.

    Args:
        df: DataFrame with sensor data (including *_filt columns when present)
        config: Pipeline configuration
        project_root: Project root for drift flag path

    Returns:
        Same DataFrame (drift detection is side-effect: logging + flag)
    """
    if df.empty or len(df) < 50:
        logger.info("Insufficient data for drift detection (need >= 50 rows)")
        return df.copy()

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    cfg = load_drift_config(config)
    p_thresh = cfg["p_value_threshold"]
    baseline_n = min(cfg["baseline_window"], len(df) // 2)
    comp_n = min(cfg["comparison_window"], len(df) - baseline_n)

    if baseline_n < 10 or comp_n < 10:
        return df.copy()

    result = df.copy()
    sensor_cols = resolve_sensor_data_columns(df, config)

    drift_detected = False
    for col in sensor_cols:
        baseline = result[col].iloc[:baseline_n].dropna().values
        comparison = result[col].iloc[-comp_n:].dropna().values
        drifted, p_val = ks_drift_test(baseline, comparison, p_thresh)
        if drifted:
            logger.warning("DRIFT DETECTED: %s (p=%.4f < %.4f)", col, p_val, p_thresh)
            drift_detected = True

    if drift_detected:
        _write_drift_flag(project_root)

    return result


def run_drift_service(
    df: pd.DataFrame,
    config: dict,
    project_root: Path = None,
) -> pd.DataFrame:
    """Main entry point for drift service."""
    return run_drift_detection(df, config, project_root)
