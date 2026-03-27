"""
Ingestion Service - Open-Meteo Ingestion
----------------------------------------
Fetches weather data from Open-Meteo APIs and appends it to a master CSV.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Expected columns for pipeline compatibility
EXPECTED_COLUMNS = ["timestamp", "city", "temperature", "humidity", "pressure"]

FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"

LOCATIONS = {
    "Delhi": {"lat": 28.61, "lon": 77.23},
    "Mumbai": {"lat": 19.07, "lon": 72.87},
    "Kolkata": {"lat": 22.57, "lon": 88.36},
    "Bangalore": {"lat": 12.97, "lon": 77.59},
    "Chennai": {"lat": 13.08, "lon": 80.27},
    "Pune": {"lat": 18.52, "lon": 73.85},
    "Jammu & Kashmir": {"lat": 34.08, "lon": 74.79},
}


def load_config(config: dict) -> tuple:
    """Extract ingestion settings from config."""
    paths = config.get("paths", {})
    master_csv = paths.get("master_csv", "data/master_sensor_data.csv")
    ingestion_cfg = config.get("ingestion", {})
    lookback_days = int(ingestion_cfg.get("history_days", 10))
    interval_seconds = int(ingestion_cfg.get("interval_seconds", 300))
    return master_csv, lookback_days, interval_seconds


def fetch_current(lat: float, lon: float) -> dict:
    """Fetch current weather for one location."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,pressure_msl",
    }
    resp = requests.get(FORECAST_API_URL, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("current", {})


def fetch_history(lat: float, lon: float, lookback_days: int = 10) -> dict:
    """Fetch hourly weather history for one location."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=lookback_days)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl",
    }
    resp = requests.get(ARCHIVE_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("hourly", {})


def collect_open_meteo_data(lookback_days: int = 10) -> pd.DataFrame:
    """Collect historical + current weather across configured cities."""
    all_rows = []

    for city, coord in LOCATIONS.items():
        lat = coord["lat"]
        lon = coord["lon"]
        try:
            hist = fetch_history(lat, lon, lookback_days)
            times = hist.get("time", [])
            temps = hist.get("temperature_2m", [])
            humidities = hist.get("relative_humidity_2m", [])
            pressures = hist.get("pressure_msl", [])

            n = min(len(times), len(temps), len(humidities), len(pressures))
            for i in range(n):
                all_rows.append(
                    {
                        "timestamp": times[i],
                        "city": city,
                        "temperature": temps[i],
                        "humidity": humidities[i],
                        "pressure": pressures[i],
                    }
                )

            curr = fetch_current(lat, lon)
            if curr:
                all_rows.append(
                    {
                        "timestamp": curr.get("time"),
                        "city": city,
                        "temperature": curr.get("temperature_2m"),
                        "humidity": curr.get("relative_humidity_2m"),
                        "pressure": curr.get("pressure_msl"),
                    }
                )
        except requests.RequestException as e:
            logger.error("Open-Meteo request failed for %s: %s", city, e)
        except Exception as e:
            logger.error("Open-Meteo processing failed for %s: %s", city, e)

    if not all_rows:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = pd.DataFrame(all_rows, columns=EXPECTED_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "temperature", "humidity", "pressure"]).copy()
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
    df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
    df = df.dropna(subset=["temperature", "humidity", "pressure"])
    df = df.sort_values(["timestamp", "city"]).drop_duplicates(subset=["timestamp", "city"])
    logger.info("Collected %d rows from Open-Meteo", len(df))
    return df


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

    dedupe_keys = ["timestamp"]
    if "city" in df.columns:
        dedupe_keys.append("city")

    if full_path.exists():
        existing = pd.read_csv(full_path)
        existing.columns = existing.columns.str.strip().str.lower()
        if "city" in existing.columns and "city" not in dedupe_keys:
            dedupe_keys.append("city")
        combined = pd.concat([existing, df], ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        combined = combined.drop_duplicates(subset=dedupe_keys).sort_values(dedupe_keys)
    else:
        combined = df.copy()

    combined.to_csv(full_path, index=False)
    logger.info("Master CSV updated: %s (%d total rows)", full_path, len(combined))
    return str(full_path)


def run_ingestion(config: dict, base_path: str = ".") -> pd.DataFrame:
    """
    Main entry point: collect from Open-Meteo and append to master CSV.

    Args:
        config: Pipeline configuration dict (from config.yaml)
        base_path: Base path for resolving relative paths

    Returns:
        DataFrame of ingested data (for downstream pipeline use)
    """
    master_csv, lookback_days, _ = load_config(config)
    df = collect_open_meteo_data(lookback_days=lookback_days)
    if not df.empty:
        append_to_master(df, master_csv, base_path)
    return df


def run_ingestion_stream(config: dict, base_path: str = ".") -> None:
    """Continuously run ingestion every configured interval."""
    _, _, interval_seconds = load_config(config)
    logger.info("Starting ingestion stream loop with interval=%ss", interval_seconds)
    while True:
        try:
            run_ingestion(config, base_path=base_path)
        except Exception as e:
            logger.exception("Ingestion iteration failed: %s", e)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.yaml"
    cfg = {}
    try:
        import yaml  # optional; ingestion can run with defaults

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        logger.warning(
            "PyYAML not installed; starting ingestion stream with defaults "
            "(history_days=10, interval_seconds=300)."
        )
    except Exception as e:
        logger.warning("Failed to load %s (%s); using defaults.", config_path, e)

    run_ingestion_stream(cfg, base_path=str(project_root))
