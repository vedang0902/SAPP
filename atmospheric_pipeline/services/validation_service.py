"""
Validation Service - Data Validation Layer
------------------------------------------
Checks for missing values and sensor bounds (configurable via config.yaml).
Logs invalid rows.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from services.pipeline_schema import sensor_base_columns

logger = logging.getLogger(__name__)


def load_bounds(config: dict) -> dict:
    """Load sensor bounds from config."""
    return config.get("sensor_bounds", {})


def get_invalid_log_path(config: dict, base_path: str = ".") -> Path:
    """Get path for invalid rows log file."""
    paths = config.get("paths", {})
    log_path = paths.get("invalid_log", "logs/invalid_rows.log")
    full_path = Path(base_path) / log_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    return full_path


def validate_row(row: pd.Series, bounds: dict) -> tuple[bool, list[str]]:
    """
    Validate a single row against sensor bounds.

    Returns:
        (is_valid, list of violation messages)
    """
    violations = []

    for col, lim in bounds.items():
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val):
            violations.append(f"{col}: missing value")
        else:
            try:
                v = float(val)
                if v < lim["min"]:
                    violations.append(f"{col}={v} below min {lim['min']}")
                elif v > lim["max"]:
                    violations.append(f"{col}={v} above max {lim['max']}")
            except (TypeError, ValueError):
                violations.append(f"{col}: invalid type/value")

    return len(violations) == 0, violations


def validate_dataframe(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Validate DataFrame: check missing values and sensor bounds.
    Invalid rows are logged; only valid rows are returned.

    Args:
        df: Input DataFrame with configured sensor columns (see sensors.columns)
        config: Pipeline configuration

    Returns:
        DataFrame containing only valid rows
    """
    if df.empty:
        return df.copy()

    bounds = load_bounds(config)
    valid_mask = np.ones(len(df), dtype=bool)
    invalid_rows = []

    for col in sensor_base_columns(config):
        if col in df.columns:
            valid_mask &= df[col].notna()

    for i, (idx, row) in enumerate(df.iterrows()):
        is_valid, violations = validate_row(row, bounds)
        if not is_valid:
            valid_mask[i] = False
            invalid_rows.append({"index": idx, "row": row.to_dict(), "violations": violations})

    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        log_path = get_invalid_log_path(config)
        with open(log_path, "a", encoding="utf-8") as f:
            for rec in invalid_rows:
                f.write(f"Invalid row: {rec}\n")
        logger.warning("Logged %d invalid rows to %s", invalid_count, log_path)

    result = df[valid_mask].copy().reset_index(drop=True)
    logger.info("Validation: %d valid, %d invalid rows", len(result), invalid_count)
    return result


def run_validation(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Main entry point for validation service.

    Args:
        df: Input DataFrame from ingestion
        config: Pipeline configuration

    Returns:
        Validated DataFrame
    """
    return validate_dataframe(df, config)
