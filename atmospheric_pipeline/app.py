"""
Flask API with Prometheus Metrics
---------------------------------
Endpoints: /run, /health, /metrics (for Grafana scraping via Prometheus)
"""

import json
import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, request
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from main_pipeline import run_pipeline, load_config

# ---------------------------------------------------------------------------
# Prometheus metrics (optional - only if prometheus_client installed)
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["TRUSTED_HOSTS"] = ["localhost", "127.0.0.1", "atmospheric_app", "*"]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)
OUTPUT_DIR = PROJECT_ROOT / "output"
METRICS_SNAPSHOT_PATH = OUTPUT_DIR / "metrics_snapshot.json"
INGESTION_STATE_PATH = OUTPUT_DIR / "ingestion_state.json"

# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    pipeline_runs_total = Counter(
        "atmospheric_pipeline_runs_total",
        "Total number of pipeline runs",
    )
    pipeline_anomalies_gauge = Gauge(
        "atmospheric_pipeline_anomalies",
        "Number of anomalies in last run",
    )
    pipeline_records_gauge = Gauge(
        "atmospheric_pipeline_records_total",
        "Total records processed in last run",
    )
    pipeline_new_records_gauge = Gauge(
        "atmospheric_pipeline_new_records_last_run",
        "New records ingested during the latest run",
    )
    # Hybrid forecasting metrics (RMSE, MAE, variance, training time) are
    # populated by prediction_service and exposed via /metrics


def _load_json(path: Path) -> dict:
    """Load a small JSON state file if it exists."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return {}


def _sync_shared_metrics() -> None:
    """Sync shared on-disk pipeline metrics into this process for Prometheus."""
    if not PROMETHEUS_AVAILABLE:
        return

    snapshot = _load_json(METRICS_SNAPSHOT_PATH)
    ingestion_state = _load_json(INGESTION_STATE_PATH)

    if snapshot:
        total_runs = int(snapshot.get("pipeline_runs_total", 0))
        last_total_runs = getattr(_sync_shared_metrics, "_last_total_runs", 0)
        if total_runs > last_total_runs:
            pipeline_runs_total.inc(total_runs - last_total_runs)
            _sync_shared_metrics._last_total_runs = total_runs

        pipeline_records_gauge.set(int(snapshot.get("records_total", 0)))
        pipeline_anomalies_gauge.set(int(snapshot.get("anomalies", 0)))

        if "new_records_last_run" in snapshot:
            pipeline_new_records_gauge.set(int(snapshot.get("new_records_last_run", 0)))

    if ingestion_state:
        if "total_master_records" in ingestion_state:
            pipeline_records_gauge.set(int(ingestion_state.get("total_master_records", 0)))
        if "new_records_last_run" in ingestion_state:
            pipeline_new_records_gauge.set(int(ingestion_state.get("new_records_last_run", 0)))


@app.route("/")
def index():
    """Landing page with available endpoints."""
    return jsonify({
        "service": "Atmospheric Monitoring Pipeline",
        "endpoints": {
            "/": "This page",
            "/run": "Execute pipeline (GET)",
            "/health": "Health check (GET)",
            "/metrics": "Prometheus metrics (GET)",
        },
    })


@app.route("/run", methods=["GET"])
def run():
    """Execute the pipeline and return anomaly summary."""
    try:
        refresh_flag = request.args.get("refresh", "").strip().lower()
        refresh_ingestion = refresh_flag in {"1", "true", "yes"}
        full_flag = request.args.get("full", "").strip().lower()
        fast_mode = full_flag not in {"1", "true", "yes"}
        df = run_pipeline(refresh_ingestion=refresh_ingestion, fast_mode=fast_mode)
        if df.empty:
            return jsonify({
                "status": "ok",
                "total_records": 0,
                "anomalies": 0,
                "refresh_ingestion": refresh_ingestion,
                "fast_mode": fast_mode,
                "message": "No data processed",
            })
        if "anomaly_combined" in df.columns:
            outliers = df[df["anomaly_combined"] == 1]
        else:
            outliers = df[df["anomaly"] == -1] if "anomaly" in df.columns else pd.DataFrame()

        if PROMETHEUS_AVAILABLE:
            pipeline_runs_total.inc()
            pipeline_anomalies_gauge.set(len(outliers))
            pipeline_records_gauge.set(len(df))

        return jsonify({
            "status": "ok",
            "total_records": len(df),
            "anomalies": len(outliers),
            "refresh_ingestion": refresh_ingestion,
            "fast_mode": fast_mode,
            "sample": outliers.head(5).to_dict(orient="records"),
        })
    except Exception as e:
        logger.exception("Pipeline run failed: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/metrics")
def metrics():
    """Prometheus metrics endpoint for Grafana scraping."""
    if not PROMETHEUS_AVAILABLE:
        return "prometheus_client not installed", 503
    _sync_shared_metrics()
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
