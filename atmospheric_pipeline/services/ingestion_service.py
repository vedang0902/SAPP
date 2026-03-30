"""
Ingestion Service - API + Local Stream Ingestion
------------------------------------------------
Fetches weather data from Open-Meteo APIs, falls back to local CSV stream data
when needed, and maintains a deduplicated master CSV for downstream services.
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
    input_stream = paths.get("input_stream", "data/stream")
    ingestion_cfg = config.get("ingestion", {})
    lookback_days = int(ingestion_cfg.get("history_days", 10))
    interval_seconds = int(ingestion_cfg.get("interval_seconds", 300))
    return master_csv, input_stream, lookback_days, interval_seconds


def _normalize_sensor_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns/types to the format expected by the pipeline."""
    if df.empty:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    result = df.copy()
    result.columns = result.columns.str.strip().str.lower()

    if "relative_humidity" in result.columns and "humidity" not in result.columns:
        result = result.rename(columns={"relative_humidity": "humidity"})
    if "pressure_msl" in result.columns and "pressure" not in result.columns:
        result = result.rename(columns={"pressure_msl": "pressure"})

    for col in EXPECTED_COLUMNS:
        if col not in result.columns:
            result[col] = None

    result = result[EXPECTED_COLUMNS].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    for col in ["temperature", "humidity", "pressure"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result = result.dropna(subset=["timestamp", "temperature", "humidity", "pressure"])

    if result["city"].isna().all():
        result["city"] = "stream"
    else:
        result["city"] = result["city"].fillna("stream")

    return result.sort_values(["timestamp", "city"]).drop_duplicates(
        subset=["timestamp", "city"]
    )


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

    df = _normalize_sensor_frame(pd.DataFrame(all_rows, columns=EXPECTED_COLUMNS))
    logger.info("Collected %d rows from Open-Meteo", len(df))
    return df


def load_stream_data(stream_path: str, base_path: str = ".") -> pd.DataFrame:
    """Load and combine local CSV stream files as an offline ingestion fallback."""
    full_path = Path(base_path) / stream_path
    if not full_path.exists():
        logger.info("Input stream path not found: %s", full_path)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    csv_files = sorted(full_path.glob("*.csv"))
    if not csv_files:
        logger.info("No CSV files found in stream path: %s", full_path)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    frames = []
    for csv_file in csv_files:
        try:
            frames.append(pd.read_csv(csv_file))
        except Exception as e:
            logger.warning("Failed to read stream file %s: %s", csv_file, e)

    if not frames:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = _normalize_sensor_frame(pd.concat(frames, ignore_index=True))
    logger.info("Loaded %d rows from local stream files", len(df))
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


def load_master_data(master_path: str, base_path: str = ".") -> pd.DataFrame:
    """Load master CSV if it exists."""
    full_path = Path(base_path) / master_path
    if not full_path.exists():
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    try:
        return _normalize_sensor_frame(pd.read_csv(full_path))
    except Exception as e:
        logger.warning("Failed to load master CSV %s: %s", full_path, e)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)


def run_ingestion(config: dict, base_path: str = ".") -> pd.DataFrame:
    """
    Main entry point: collect from Open-Meteo and append to master CSV.

    Args:
        config: Pipeline configuration dict (from config.yaml)
        base_path: Base path for resolving relative paths

    Returns:
        DataFrame of ingested data (for downstream pipeline use)
    """
    master_csv, input_stream, lookback_days, _ = load_config(config)

    api_df = collect_open_meteo_data(lookback_days=lookback_days)
    stream_df = pd.DataFrame(columns=EXPECTED_COLUMNS)

    if api_df.empty:
        logger.warning("Open-Meteo returned no rows; falling back to local stream data")
        stream_df = load_stream_data(input_stream, base_path)

    incoming = api_df if not api_df.empty else stream_df
    if not incoming.empty:
        append_to_master(incoming, master_csv, base_path)

    master_df = load_master_data(master_csv, base_path)
    if not master_df.empty:
        return master_df

    return incoming


def run_ingestion_stream(config: dict, base_path: str = ".") -> None:
    """Continuously run ingestion every configured interval."""
    _, _, _, interval_seconds = load_config(config)
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
