# Atmospheric Monitoring Pipeline

## Overview

This project implements an end-to-end atmospheric monitoring and anomaly
detection pipeline using signal filtering, feature engineering, machine
learning, and workflow automation.

It integrates: - Median + Kalman filtering - Rolling statistical feature
extraction - Isolation Forest anomaly detection - Flask REST API -
Airflow orchestration

## Architecture

    Weather Data (Meteostat)
            ↓
    Median + Kalman Filter
            ↓
    Feature Engineering
            ↓
    Isolation Forest (ML)
            ↓
    CSV / API / Airflow

## Project Structure

    atmospheric_pipeline/
    ├── feature_extractor.py
    ├── filters.py
    ├── pipeline.py
    ├── app.py
    └── dags/
        └── atmospheric_dag.py

## Requirements

-   Python 3.9+
-   pandas
-   numpy
-   scikit-learn
-   meteostat
-   flask
-   apache-airflow
-   matplotlib

Install dependencies:

``` bash
pip install -r requirements.txt
```

## Running the Pipeline

### Local Execution

``` bash
python pipeline.py
```

### Flask API

``` bash
python app.py
```

Endpoints:

-   GET /run
-   GET /health

### Airflow Deployment

1.  Copy project to: /opt/airflow/atmospheric_pipeline

2.  Copy DAG: /opt/airflow/dags/

3.  Restart Airflow services

### Cron Automation

``` bash
0 2 * * * cd /app/atmospheric_pipeline && python pipeline.py
```

## Output

-   anomaly_results.csv
-   API JSON response
-   Airflow logs

## Machine Learning

Algorithm: - Isolation Forest

Features: - Rolling mean, std, skew, kurtosis - Gradients - Energy
metrics - Composite ratios

## Use Cases

-   Weather anomaly monitoring
-   Environmental research
-   Smart city systems
-   Sensor diagnostics
-   Climate analytics

## Future Enhancements

-   Docker + Kubernetes
-   MLflow integration
-   Kafka streaming
-   Grafana dashboards
-   Cloud deployment (AWS/GCP)

## Author

Vedang Kane
