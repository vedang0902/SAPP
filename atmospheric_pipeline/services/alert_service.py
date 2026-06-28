"""
Alert Service - Anomaly Alerts
------------------------------
Logs anomaly summaries and optionally sends email alerts.

Email settings are read from config and can be overridden with environment
variables so the service is safe to run on a fresh machine or in containers.
"""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

logger = logging.getLogger(__name__)


def format_alert_message(df: pd.DataFrame, anomaly_count: int) -> dict:
    """Build a compact alert payload from anomaly rows."""
    total = len(df)
    return {
        "total_records": total,
        "anomaly_count": int(anomaly_count),
        "anomaly_pct": round(100 * anomaly_count / total, 2) if total else 0,
        "sample": df.head(5).to_dict(orient="records") if anomaly_count > 0 else [],
    }


def _resolve_alert_config(config: dict) -> dict:
    """Merge alert config with environment-variable overrides."""
    alerts_cfg = dict(config.get("alerts", {}))

    env_receivers = os.environ.get("ALERT_RECEIVER_EMAILS", "").strip()
    if env_receivers:
        alerts_cfg["receiver_emails"] = [
            email.strip() for email in env_receivers.split(",") if email.strip()
        ]

    alerts_cfg["sender_email"] = os.environ.get(
        "ALERT_SENDER_EMAIL", alerts_cfg.get("sender_email", "")
    )
    alerts_cfg["sender_password"] = os.environ.get(
        "EMAIL_APP_PASSWORD",
        os.environ.get("ALERT_SENDER_PASSWORD", alerts_cfg.get("sender_password", "")),
    )
    alerts_cfg["smtp_host"] = os.environ.get(
        "ALERT_SMTP_HOST", alerts_cfg.get("smtp_host", "smtp.gmail.com")
    )
    alerts_cfg["smtp_port"] = int(
        os.environ.get("ALERT_SMTP_PORT", alerts_cfg.get("smtp_port", 587))
    )
    return alerts_cfg


def send_email_alert(config: dict, message: dict) -> bool:
    """Send anomaly summary via SMTP when configuration is present."""
    try:
        alerts_cfg = _resolve_alert_config(config)

        sender = alerts_cfg.get("sender_email")
        password = alerts_cfg.get("sender_password")
        receivers = alerts_cfg.get("receiver_emails", [])
        smtp_host = alerts_cfg.get("smtp_host", "smtp.gmail.com")
        smtp_port = int(alerts_cfg.get("smtp_port", 587))

        if not sender or not password or not receivers:
            logger.info("Email config not set; skipping email alert")
            return False

        subject = "Atmospheric Pipeline Alert"
        sample_df = pd.DataFrame(message["sample"])
        sample_text = (
            sample_df.to_string(index=False) if not sample_df.empty else "No anomalies"
        )
        body = (
            "Atmospheric Pipeline Alert\n\n"
            f"Total Records: {message['total_records']}\n"
            f"Anomalies: {message['anomaly_count']}\n"
            f"Percentage: {message['anomaly_pct']}%\n\n"
            f"Sample:\n{sample_text}"
        )

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = ", ".join(receivers)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receivers, msg.as_string())

        logger.info("Email alert sent successfully")
        return True
    except Exception as e:
        logger.warning("Email alert failed: %s", e)
        return False


def run_alert_service(df: pd.DataFrame, config: dict) -> None:
    """Log anomaly summary and trigger email when threshold is met."""
    if df.empty:
        logger.info("Alert service skipped: no rows")
        return

    if "anomaly_combined" in df.columns:
        anomalies = df[df["anomaly_combined"] == 1]
    elif "anomaly" in df.columns:
        anomalies = df[df["anomaly"] == -1]
    else:
        logger.warning("No anomaly column found for alert service")
        return

    anomaly_count = len(anomalies)
    msg = format_alert_message(anomalies, anomaly_count)

    logger.info(
        "ALERT: %d anomalies out of %d records (%.2f%%)",
        anomaly_count,
        len(df),
        (100 * anomaly_count / len(df)) if len(df) else 0.0,
    )

    if anomaly_count > 0:
        logger.info("Sample anomalies:\n%s", pd.DataFrame(msg["sample"]).to_string())

    threshold = config.get("alerts", {}).get("min_anomalies", 1)
    if anomaly_count < threshold:
        logger.info("Below threshold (%d), skipping email alert", threshold)
        return

    send_email_alert(config, msg)
