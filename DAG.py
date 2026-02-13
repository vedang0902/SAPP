############################################
# dags/atmospheric_dag.py (Airflow DAG)
############################################
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

sys.path.append('/opt/airflow/atmospheric_pipeline')
from pipeline import run_pipeline


def pipeline_task():
    run_pipeline()


def notify():
    print("Pipeline completed successfully")


def_args = {
    'owner':'vedang',
    'depends_on_past':False,
    'start_date':datetime(2025,1,1),
    'retries':1,
    'retry_delay':timedelta(minutes=5)
}

with DAG(
    dag_id='atmospheric_monitoring_pipeline',
    default_args=def_args,
    schedule_interval='@daily',
    catchup=False
) as dag:

    run_job = PythonOperator(
        task_id='run_pipeline',
        python_callable=pipeline_task
    )

    notify_job = PythonOperator(
        task_id='notify',
        python_callable=notify
    )

    run_job >> notify_job

############################################
# Automation (cron example)
############################################
# 0 2 * * * cd /app/atmospheric_pipeline && python pipeline.py
