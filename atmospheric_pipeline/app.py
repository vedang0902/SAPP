"""
Flask API with Prometheus Metrics
---------------------------------
Endpoints: /run, /health, /metrics (for Grafana scraping via Prometheus)
"""

import logging
import sys
from pathlib import Path

from flask import Flask, jsonify

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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

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


@app.route("/run", methods=["GET"])
def run():
    """Execute the pipeline and return anomaly summary."""
    try:
        df = run_pipeline()
        if df.empty:
            return jsonify({
                "status": "ok",
                "total_records": 0,
                "anomalies": 0,
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
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
