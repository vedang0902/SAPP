"""
Ingestion Service - CSV Stream Ingestion
----------------------------------------
Reads real-time CSV files from a folder (simulates hardware sensor stream)
and appends new rows to a master CSV.
"""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Expected CSV columns for sensor stream
EXPECTED_COLUMNS = ["timestamp", "temperature", "humidity", "pressure"]


def load_config(config: dict) -> tuple:
    """Extract ingestion paths from config."""
    paths = config.get("paths", {})
    stream_dir = paths.get("input_stream", "data/stream")
    master_csv = paths.get("master_csv", "data/master_sensor_data.csv")
    return stream_dir, master_csv


def read_stream_csvs(stream_dir: str, base_path: str = ".") -> pd.DataFrame:
    """
    Read all CSV files from the stream folder and concatenate them.

    Args:
        stream_dir: Relative path to folder containing CSV files
        base_path: Base path for resolving stream_dir (e.g., pipeline root)

    Returns:
        DataFrame with combined data from all CSVs, or empty DataFrame if none found
    """
    full_path = Path(base_path) / stream_dir
    if not full_path.exists():
        logger.warning("Stream directory does not exist: %s", full_path)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    csv_files = list(full_path.glob("*.csv"))
    if not csv_files:
        logger.info("No CSV files found in stream directory: %s", full_path)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    frames = []
    for f in sorted(csv_files):
        try:
            df = pd.read_csv(f)
            # Normalize column names to lowercase
            df.columns = df.columns.str.strip().str.lower()
            # Ensure required columns exist
            for col in EXPECTED_COLUMNS:
                if col not in df.columns:
                    logger.warning("Missing column '%s' in %s", col, f.name)
                    return pd.DataFrame(columns=EXPECTED_COLUMNS)
            df = df[EXPECTED_COLUMNS].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            frames.append(df)
        except Exception as e:
            logger.error("Error reading %s: %s", f, e)

    if not frames:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    logger.info("Ingested %d rows from %d CSV files", len(combined), len(csv_files))
    return combined


def append_to_master(df: pd.DataFrame, master_path: str, base_path: str = ".") -> str:
    """
    Append new rows to the master CSV. Creates master file if it doesn't exist.

    Args:
        df: DataFrame with new sensor data
        master_path: Relative path to master CSV
        base_path: Base path for resolving master_path

    Returns:
        Absolute path to master CSV
    """
    full_path = Path(base_path) / master_path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        logger.info("No new data to append to master CSV")
        return str(full_path)

    if full_path.exists():
        existing = pd.read_csv(full_path)
        existing.columns = existing.columns.str.strip().str.lower()
        combined = pd.concat([existing, df], ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    else:
        combined = df.copy()

    combined.to_csv(full_path, index=False)
    logger.info("Master CSV updated: %s (%d total rows)", full_path, len(combined))
    return str(full_path)


def run_ingestion(config: dict, base_path: str = ".") -> pd.DataFrame:
    """
    Main entry point: read from stream folder and append to master CSV.

    Args:
        config: Pipeline configuration dict (from config.yaml)
        base_path: Base path for resolving relative paths

    Returns:
        DataFrame of ingested data (for downstream pipeline use)
    """
    stream_dir, master_csv = load_config(config)
    df = read_stream_csvs(stream_dir, base_path)
    if not df.empty:
        append_to_master(df, master_csv, base_path)
    return df
