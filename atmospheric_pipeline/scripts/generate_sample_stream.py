"""
Generate Sample CSV Stream Data
-------------------------------
Creates sample sensor CSV files in data/stream for testing the pipeline.
Simulates hardware sensor stream.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_sample_csv(
    output_path: Path,
    n_rows: int = 100,
    freq: str = "1h",
    add_anomalies: bool = True,
) -> None:
    """
    Generate a sample sensor CSV with temperature, humidity, pressure.

    Args:
        output_path: Path to output CSV
        n_rows: Number of rows
        freq: Pandas frequency string (e.g., '1h', '1min')
        add_anomalies: Whether to inject a few anomalies
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_time = datetime(2025, 1, 1)
    timestamps = pd.date_range(base_time, periods=n_rows, freq=freq)

    np.random.seed(42)
    temperature = np.random.normal(25, 2, n_rows)
    humidity = np.random.normal(60, 5, n_rows)
    pressure = np.random.normal(1013, 3, n_rows)

    if add_anomalies:
        # Inject a few anomalies
        idx = [10, 50, 80]
        temperature[idx] = [45, -10, 70]  # Out of normal range
        humidity[25] = 150  # Invalid
        pressure[60] = 500  # Invalid

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
    })
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {n_rows} rows")


if __name__ == "__main__":
    stream_dir = PROJECT_ROOT / "data" / "stream"
    stream_dir.mkdir(parents=True, exist_ok=True)
    generate_sample_csv(stream_dir / "sensor_batch_001.csv", n_rows=150)
    print("Done. Place CSVs in data/stream/ to simulate hardware stream.")
