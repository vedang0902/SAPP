"""
Generate Sample CSV Stream Data
-------------------------------
Creates sample sensor CSV files in data/stream for testing the pipeline.
Uses the same column names as the Google Sheet export (DateTime, WS, WD, ...).
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_sample_csv(
    output_path: Path,
    n_rows: int = 100,
    freq: str = "1min",
    add_anomalies: bool = True,
) -> None:
    """
    Generate a sample sensor CSV compatible with ingestion normalization.

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
    ws = np.clip(np.random.normal(2, 1, n_rows), 0, None)
    wd = np.random.uniform(0, 360, n_rows)
    pressure = np.random.normal(980, 5, n_rows)
    rh = np.clip(np.random.normal(45, 8, n_rows), 0, 100)
    temp = np.random.normal(28, 2, n_rows)
    dew = np.random.normal(15, 2, n_rows)
    rain = np.zeros(n_rows)

    if add_anomalies:
        temp[[10, 50, 80]] = [50, -15, 52]
        rh[25] = 110
        pressure[60] = 500

    df = pd.DataFrame({
        "DateTime": timestamps,
        "WS": ws,
        "WD": wd,
        "Pressure": pressure,
        "RH": rh,
        "Temp": temp,
        "Dew": dew,
        "Rain": rain,
    })
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {n_rows} rows")


if __name__ == "__main__":
    stream_dir = PROJECT_ROOT / "data" / "stream"
    stream_dir.mkdir(parents=True, exist_ok=True)
    generate_sample_csv(stream_dir / "sensor_batch_001.csv", n_rows=150)
    print("Done. Place CSVs in data/stream/ to simulate hardware stream.")
