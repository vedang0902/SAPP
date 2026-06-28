"""
Ingestion Service - Google Sheets (CSV export) + local stream fallback
---------------------------------------------------------------------
Pulls published sheet data via HTTP as CSV on a fixed interval (see config),
deduplicates into a master CSV for downstream services.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

from services.pipeline_schema import sensor_base_columns

logger = logging.getLogger(__name__)

# Map common header variants (after lower/strip) to canonical names
_HEADER_ALIASES = {
    "datetime": "timestamp",
    "date time": "timestamp",
    "date_time": "timestamp",
    "timestamp": "timestamp",
    "time": "timestamp",
    "temperature": "temp",
    "temperature_2m": "temp",
    "temp": "temp",
    "rh": "rh",
    "humidity": "rh",
    "relative_humidity": "rh",
    "relative humidity": "rh",
    "pressure": "pressure",
    "pressure_msl": "pressure",
    "ws": "ws",
    "wind_speed": "ws",
    "wind speed": "ws",
    "wd": "wd",
    "wind_direction": "wd",
    "wind direction": "wd",
    "dew": "dew",
    "dew_point": "dew",
    "dew point": "dew",
    "rain": "rain",
    "rainfall": "rain",
    "precipitation": "rain",
}


def _canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace("  ", " ") for c in out.columns]
    out.columns = [_HEADER_ALIASES.get(c, c.replace(" ", "_")) for c in out.columns]
    return out


def _normalize_sensor_frame(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Normalize sheet / stream / master rows to timestamp + configured sensor columns."""
    cfg = config or {}
    sensors = sensor_base_columns(cfg)

    if df.empty:
        return pd.DataFrame(columns=["timestamp", *sensors])

    result = _canonical_columns(df)

    # Legacy Open-Meteo / old master compatibility
    if "temp" not in result.columns and "temperature" in result.columns:
        result["temp"] = result["temperature"]
    if "rh" not in result.columns and "humidity" in result.columns:
        result["rh"] = result["humidity"]

    if "timestamp" not in result.columns:
        logger.warning("No timestamp/datetime column found after normalization")
        return pd.DataFrame(columns=["timestamp", *sensors])

    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")

    for col in sensors:
        if col not in result.columns:
            result[col] = pd.NA

    for col in sensors:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    # Sheet may omit rain on dry rows; wind can be blank in some exports
    if "rain" in sensors:
        result["rain"] = result["rain"].fillna(0)
    if "ws" in sensors:
        result["ws"] = result["ws"].fillna(0)
    if "wd" in sensors:
        result["wd"] = result["wd"].fillna(0)
    if "dew" in sensors:
        result["dew"] = result["dew"].fillna(result["temp"] - 5.0)

    result = result.dropna(subset=["timestamp", "temp", "rh", "pressure"])

    keep = ["timestamp"] + [c for c in sensors if c in result.columns]
    result = result[keep].copy()
    result = result.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return result


def load_config(config: dict) -> tuple:
    """Extract ingestion settings from config."""
    paths = config.get("paths", {})
    master_csv = paths.get("master_csv", "data/master_sensor_data.csv")
    input_stream = paths.get("input_stream", "data/stream")
    ingestion_cfg = config.get("ingestion", {})
    interval_seconds = int(ingestion_cfg.get("interval_seconds", 300))
    return master_csv, input_stream, interval_seconds


def _resolve_csv_export_url(config: dict) -> str:
    """
    Prefer GOOGLE_SHEETS_CSV_URL unless it looks like a tutorial placeholder (``...``),
    which would override config and produce HTTP 404.
    """
    env = os.environ.get("GOOGLE_SHEETS_CSV_URL", "").strip()
    if env and "/.../" in env.replace(" ", ""):
        logger.warning(
            "GOOGLE_SHEETS_CSV_URL contains a placeholder (...); "
            "ignoring it and using ingestion.csv_export_url from config.yaml. "
            "Unset the env var or paste the full /export?format=csv URL."
        )
        env = ""
    if env:
        logger.info("Using CSV URL from environment GOOGLE_SHEETS_CSV_URL")
        return env
    ing = config.get("ingestion") or {}
    url = str(ing.get("csv_export_url", "") or ing.get("google_sheets_csv_url", "")).strip()
    if url:
        logger.info("Using CSV URL from config ingestion.csv_export_url")
    return url


def fetch_google_sheets_csv(url: str, timeout: int = 45) -> pd.DataFrame:
    """Download CSV from a published Google Sheets export (or any HTTP CSV URL)."""
    if not url:
        return pd.DataFrame()

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        text = resp.content.decode("utf-8-sig", errors="replace")
    except requests.RequestException as e:
        logger.error("CSV URL request failed: %s", e)
        return pd.DataFrame()
    except Exception as e:
        logger.error("CSV download failed: %s", e)
        return pd.DataFrame()

    buf = io.StringIO(text)
    try:
        df = pd.read_csv(buf)
    except Exception:
        buf.seek(0)
        try:
            df = pd.read_csv(buf, sep="\t")
        except Exception as e:
            logger.error("Could not parse CSV/TSV: %s", e)
            return pd.DataFrame()

    logger.info("Fetched CSV: %d rows, %d columns", len(df), len(df.columns))
    return df


def collect_sheets_csv_data(config: dict) -> pd.DataFrame:
    """Fetch and normalize data from the configured CSV export URL."""
    url = _resolve_csv_export_url(config)
    if not url:
        logger.warning("No GOOGLE_SHEETS_CSV_URL or ingestion.csv_export_url configured")
        return pd.DataFrame()

    raw = fetch_google_sheets_csv(url)
    if raw.empty:
        return pd.DataFrame()

    df = _normalize_sensor_frame(raw, config)
    logger.info("Normalized %d rows from Google Sheets CSV", len(df))
    return df


def load_stream_data(stream_path: str, base_path: str = ".", config: dict | None = None) -> pd.DataFrame:
    """Load and combine local CSV stream files as an offline ingestion fallback."""
    full_path = Path(base_path) / stream_path
    if not full_path.exists():
        logger.info("Input stream path not found: %s", full_path)
        return pd.DataFrame(columns=["timestamp", *sensor_base_columns(config or {})])

    csv_files = sorted(full_path.glob("*.csv"))
    if not csv_files:
        logger.info("No CSV files found in stream path: %s", full_path)
        return pd.DataFrame(columns=["timestamp", *sensor_base_columns(config or {})])

    frames = []
    for csv_file in csv_files:
        try:
            frames.append(pd.read_csv(csv_file))
        except Exception as e:
            logger.warning("Failed to read stream file %s: %s", csv_file, e)

    if not frames:
        return pd.DataFrame(columns=["timestamp", *sensor_base_columns(config or {})])

    df = _normalize_sensor_frame(pd.concat(frames, ignore_index=True), config)
    logger.info("Loaded %d rows from local stream files", len(df))
    return df


def write_master_full(df: pd.DataFrame, master_path: str, base_path: str = ".", config: dict | None = None) -> str:
    """
    Overwrite the master CSV with the given frame (normalized).

    Used when Google Sheets is the source of truth so local master matches the sheet snapshot.
    """
    full_path = Path(base_path) / master_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    sensors = sensor_base_columns(config or {})
    combined = _normalize_sensor_frame(df, config)
    if combined.empty:
        logger.info("No rows to write to master CSV")
        return str(full_path)

    cols = ["timestamp", *sensors]
    for c in cols:
        if c not in combined.columns:
            combined[c] = pd.NA
    combined = combined[cols].sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    combined.to_csv(full_path, index=False)
    logger.info("Master CSV replaced: %s (%d rows)", full_path, len(combined))
    return str(full_path)


def append_to_master(df: pd.DataFrame, master_path: str, base_path: str = ".", config: dict | None = None) -> str:
    """
    Append new rows to the master CSV. Creates master file if it doesn't exist.
    """
    full_path = Path(base_path) / master_path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        logger.info("No new data to append to master CSV")
        return str(full_path)

    dedupe_keys = ["timestamp"]
    sensors = sensor_base_columns(config or {})

    if full_path.exists():
        existing = pd.read_csv(full_path)
        existing = _normalize_sensor_frame(existing, config)
        combined = pd.concat([existing, df], ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        combined = combined.drop_duplicates(subset=dedupe_keys, keep="last").sort_values(dedupe_keys)
    else:
        combined = df.copy()

    cols = ["timestamp", *sensors]
    for c in cols:
        if c not in combined.columns:
            combined[c] = pd.NA
    combined = combined[cols]
    combined.to_csv(full_path, index=False)
    logger.info("Master CSV updated: %s (%d total rows)", full_path, len(combined))
    return str(full_path)


def load_master_data(master_path: str, base_path: str = ".", config: dict | None = None) -> pd.DataFrame:
    """Load master CSV if it exists."""
    full_path = Path(base_path) / master_path
    if not full_path.exists():
        return pd.DataFrame(columns=["timestamp", *sensor_base_columns(config or {})])

    try:
        return _normalize_sensor_frame(pd.read_csv(full_path), config)
    except Exception as e:
        logger.warning("Failed to load master CSV %s: %s", full_path, e)
        return pd.DataFrame(columns=["timestamp", *sensor_base_columns(config or {})])


def _write_ingestion_state(base_path: str, new_records_last_run: int, total_master_records: int) -> None:
    path = Path(base_path) / "output" / "ingestion_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "new_records_last_run": int(new_records_last_run),
        "total_master_records": int(total_master_records),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    logger.info("Ingestion state updated at %s", path)


def run_ingestion(config: dict, base_path: str = ".") -> pd.DataFrame:
    """
    Fetch published Google Sheets CSV via HTTP and update master CSV.

    When ``ingestion.replace_master_from_sheet`` is true (default), a successful sheet fetch
    overwrites ``master_sensor_data.csv`` with that snapshot so the file matches the sheet.

    Falls back to local stream CSVs when the URL is unset or the fetch returns no rows;
    stream fallback uses merge + append (see ``append_to_master``).
    """
    master_csv, input_stream, _ = load_config(config)
    ingestion_cfg = config.get("ingestion") or {}
    replace_master = bool(ingestion_cfg.get("replace_master_from_sheet", True))

    existing_master_df = load_master_data(master_csv, base_path, config)
    existing_count = len(existing_master_df)

    api_df = collect_sheets_csv_data(config)
    stream_df = pd.DataFrame()
    from_sheet = bool(not api_df.empty)

    if api_df.empty:
        logger.warning("Google Sheets CSV returned no rows; falling back to local stream data")
        stream_df = load_stream_data(input_stream, base_path, config)

    incoming = api_df if not api_df.empty else stream_df
    if not incoming.empty:
        if from_sheet and replace_master:
            write_master_full(incoming, master_csv, base_path, config)
        else:
            append_to_master(incoming, master_csv, base_path, config)

    master_df = load_master_data(master_csv, base_path, config)
    if not master_df.empty:
        if from_sheet and replace_master:
            new_run = len(incoming)
        else:
            new_run = max(len(master_df) - existing_count, 0)
        _write_ingestion_state(
            base_path,
            new_records_last_run=new_run,
            total_master_records=len(master_df),
        )
        return master_df

    if not incoming.empty:
        _write_ingestion_state(
            base_path,
            new_records_last_run=len(incoming),
            total_master_records=len(incoming),
        )

    return incoming


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
    cfg: dict = {}
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        logger.warning("PyYAML not installed; ingestion stream using empty config defaults.")
    except Exception as e:
        logger.warning("Failed to load %s (%s); using defaults.", config_path, e)

    run_ingestion_stream(cfg, base_path=str(project_root))
