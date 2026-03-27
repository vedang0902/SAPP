"""
Alert Service - Anomaly Alerts
------------------------------
Consumes a DataFrame and sends anomaly alerts via email.
"""

import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd

logger = logging.getLogger(__name__)
config = {
        "alerts": {
            "sender_email": "sarthaksjoshi2004@gmail.com",
            "sender_password": "joazvitsasmocgax",
            "receiver_emails": ["18sarthakjoshi@gmail.com", "vedangkane@gmail.com"],
            
            "min_anomalies": 2
        }
    }

# 🔹 Format alert message
def format_alert_message(df: pd.DataFrame, anomaly_count: int) -> dict:
    total = len(df)
    return {
        "total_records": total,
        "anomaly_count": int(anomaly_count),
        "anomaly_pct": round(100 * anomaly_count / total, 2) if total else 0,
        "sample": df.head(5).to_dict(orient="records") if anomaly_count > 0 else [],
    }

# 🔹 Send email alert
def send_email_alert(config: dict, message: dict) -> bool:
    try:
        alerts_cfg = config.get("alerts", {})

        sender = alerts_cfg.get("sender_email")
        password = alerts_cfg.get("sender_password") or os.environ.get("EMAIL_APP_PASSWORD")
        receivers = alerts_cfg.get("receiver_emails", [])

        if not sender or not password or not receivers:
            logger.warning("Email config incomplete")
            return False

        subject = "Atmospheric Pipeline Alert"

        sample_df = pd.DataFrame(message["sample"])

        body = (
            f"Atmospheric Pipeline Alert\n\n"
            f"Total Records: {message['total_records']}\n"
            f"Anomalies: {message['anomaly_count']}\n"
            f"Percentage: {message['anomaly_pct']}%\n\n"
            f"Sample:\n{sample_df.to_string(index=False) if not sample_df.empty else 'No anomalies'}"
        )

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = ", ".join(receivers)
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receivers, msg.as_string())
        server.quit()

        logger.info("Email alert sent successfully")
        return True

    except Exception as e:
        logger.warning("Email alert failed: %s", e)
        return False


# 🔹 Main alert service (UNCHANGED LOGIC)
def run_alert_service(df: pd.DataFrame, config: dict) -> None:
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
    msg = format_alert_message(anomalies, anomaly_count)

    logger.info(
        "ALERT: %d anomalies out of %d records (%.2f%%)",
        anomaly_count, len(df), msg["anomaly_pct"],
    )

    if anomaly_count > 0:
        logger.info("Sample anomalies:\n%s", pd.DataFrame(msg["sample"]).to_string())

  
    threshold = config.get("alerts", {}).get("min_anomalies", 1)
    if anomaly_count < threshold:
        logger.info("Below threshold (%d), skipping email alert", threshold)
        return

    send_email_alert(config, msg)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
