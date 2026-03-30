"""
Airflow DAG - Atmospheric Monitoring Pipeline
---------------------------------------------
TaskFlow API - each service as separate task.
Runs every 10 minutes.
"""

from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add atmospheric_pipeline to path (when running in Airflow container or locally)
AIRFLOW_PIPELINE = Path("/opt/airflow/atmospheric_pipeline")
LOCAL_PIPELINE = Path(__file__).resolve().parent.parent
pipeline_path = AIRFLOW_PIPELINE if AIRFLOW_PIPELINE.exists() else LOCAL_PIPELINE
sys.path.insert(0, str(pipeline_path))

from airflow.decorators import dag, task
import pandas as pd


@dag(
    dag_id="atmospheric_monitoring_pipeline",
    default_args={
        "owner": "vedang",
        "depends_on_past": False,
        "start_date": datetime(2025, 1, 1),
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    schedule="*/10 * * * *",  # Run every 10 minutes
    catchup=False,
    tags=["atmospheric", "monitoring", "anomaly"],
)
def atmospheric_dag():
    """Atmospheric monitoring pipeline with TaskFlow API."""

    @task
    def ingestion_task():
        from services.ingestion_service import run_ingestion
        from main_pipeline import load_config, PROJECT_ROOT
        config = load_config()
        df = run_ingestion(config, str(PROJECT_ROOT))
        if df.empty:
            # Load from master as fallback
            master = PROJECT_ROOT / config.get("paths", {}).get("master_csv", "data/master_sensor_data.csv")
            if master.exists():
                df = pd.read_csv(master)
                df.columns = df.columns.str.strip().str.lower()
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.to_json(date_format="iso") if not df.empty else "{}"

    @task
    def validation_task(ingestion_json: str):
        from services.validation_service import run_validation
        from main_pipeline import load_config
        config = load_config()
        if ingestion_json == "{}":
            df = pd.DataFrame()
        else:
            df = pd.read_json(ingestion_json)
        df = run_validation(df, config)
        return df.to_json(date_format="iso") if not df.empty else "{}"

    @task
    def filtering_task(validation_json: str):
        from services.filtering_service import run_filtering
        from main_pipeline import load_config
        config = load_config()
        if validation_json == "{}":
            return "{}"
        df = pd.read_json(validation_json)
        df = run_filtering(df, config)
        return df.to_json(date_format="iso")

    @task
    def feature_task(filtering_json: str):
        from services.feature_service import run_feature_engineering
        from main_pipeline import load_config
        config = load_config()
        if filtering_json == "{}":
            return "{}"
        df = pd.read_json(filtering_json)
        df = run_feature_engineering(df, config)
        return df.to_json(date_format="iso") if not df.empty else "{}"

    @task
    def seasonal_decompose_task(feature_json: str):
        from services.seasonal_decomposition_service import run_seasonal_decomposition
        from main_pipeline import load_config
        config = load_config()
        if feature_json == "{}":
            return "{}"
        df = pd.read_json(feature_json)
        df = run_seasonal_decomposition(df, config)
        return df.to_json(date_format="iso") if not df.empty else "{}"

    @task
    def prediction_task(seasonal_json: str):
        from services.prediction_service import run_prediction_service
        from main_pipeline import load_config, PROJECT_ROOT
        config = load_config()
        if seasonal_json == "{}":
            return "{}"
        df = pd.read_json(seasonal_json)
        # Use the responsive prediction path for frequent scheduled DAG runs.
        df = run_prediction_service(df, config, PROJECT_ROOT, fast_mode=True)
        return df.to_json(date_format="iso") if not df.empty else "{}"

    @task
    def model_task(prediction_json: str):
        from services.model_service import run_model_service
        from main_pipeline import load_config
        config = load_config()
        if prediction_json == "{}":
            return "{}"
        df = pd.read_json(prediction_json)
        df = run_model_service(df, config)
        return df.to_json(date_format="iso")

    @task
    def drift_task(model_json: str):
        from services.drift_service import run_drift_service
        from main_pipeline import load_config, PROJECT_ROOT
        config = load_config()
        if model_json == "{}":
            return "{}"
        df = pd.read_json(model_json)
        df = run_drift_service(df, config, PROJECT_ROOT)
        return df.to_json(date_format="iso")

    @task
    def output_task(drift_json: str):
        from main_pipeline import load_config, save_output, write_metrics_snapshot, PROJECT_ROOT
        config = load_config()
        if drift_json == "{}":
            return 0
        df = pd.read_json(drift_json)
        save_output(df, config)
        write_metrics_snapshot(df, PROJECT_ROOT)
        return len(df)

    @task
    def alert_task(drift_json: str):
        from services.alert_service import run_alert_service
        from main_pipeline import load_config
        config = load_config()
        if drift_json == "{}":
            return
        df = pd.read_json(drift_json)
        run_alert_service(df, config)

    # Task flow: ingest → validate → filter → feature → seasonal_decompose → prediction → anomaly_detection → drift → alert
    ing = ingestion_task()
    val = validation_task(ing)
    flt = filtering_task(val)
    feat = feature_task(flt)
    seasonal = seasonal_decompose_task(feat)
    pred = prediction_task(seasonal)
    mod = model_task(pred)
    drf = drift_task(mod)
    output_task(drf)
    alert_task(drf)


# Instantiate DAG
dag_instance = atmospheric_dag()
