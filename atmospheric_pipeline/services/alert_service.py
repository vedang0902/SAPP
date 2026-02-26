"""
Alert Service - Anomaly Alerts
------------------------------
Prints anomaly alerts and prepares Slack webhook integration (placeholder).
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def get_slack_webhook_url(config: dict) -> str:
    """Get Slack webhook URL from config or environment."""
    url = config.get("alerts", {}).get("slack_webhook_url", "")
    if not url:
        url = os.environ.get("SLACK_WEBHOOK_URL", "")
    return url


def format_alert_message(df: pd.DataFrame, anomaly_count: int) -> dict:
    """
    Format anomaly summary for Slack/console.

    Returns:
        Dict with summary and sample rows
    """
    total = len(df)
    return {
        "total_records": total,
        "anomaly_count": int(anomaly_count),
        "anomaly_pct": round(100 * anomaly_count / total, 2) if total else 0,
        "sample": df.head(5).to_dict(orient="records") if anomaly_count > 0 else [],
    }


def send_slack_alert(webhook_url: str, message: dict) -> bool:
    """
    Send alert to Slack webhook (placeholder - requires requests).

    Args:
        webhook_url: Slack incoming webhook URL
        message: Payload dict

    Returns:
        True if sent successfully
    """
    if not webhook_url:
        return False
    try:
        import requests
        payload = {
            "text": f"Atmospheric Pipeline Alert: {message['anomaly_count']} anomalies detected ({message['anomaly_pct']}%)"
        }
        r = requests.post(webhook_url, json=payload, timeout=5)
        return r.status_code == 200
    except ImportError:
        logger.info("requests not installed; Slack alert skipped")
        return False
    except Exception as e:
        logger.warning("Slack alert failed: %s", e)
        return False


def run_alert_service(df: pd.DataFrame, config: dict) -> None:
    """
    Print anomaly alerts and optionally send to Slack.

    Args:
        df: DataFrame with anomaly_combined or anomaly column
        config: Pipeline configuration
    """
    if df.empty:
        return

    if "anomaly_combined" in df.columns:
        anomalies = df[df["anomaly_combined"] == 1]
    elif "anomaly" in df.columns:
        anomalies = df[df["anomaly"] == -1]
    else:
        logger.warning("No anomaly column found for alert service")
        return

    anomaly_count = len(anomalies)
    msg = format_alert_message(df, anomaly_count)

    # Console output
    logger.info(
        "ALERT: %d anomalies out of %d records (%.2f%%)",
        anomaly_count, len(df), msg["anomaly_pct"],
    )
    if anomaly_count > 0:
        logger.info("Sample anomalies:\n%s", pd.DataFrame(msg["sample"]).to_string())

    # Slack (if configured)
    webhook = get_slack_webhook_url(config)
    if webhook:
        send_slack_alert(webhook, msg)
