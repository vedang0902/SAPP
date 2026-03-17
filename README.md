# Atmospheric Monitoring Pipeline

Production-ready modular atmospheric monitoring and anomaly detection system .

## Architecture

```
Sensor CSV Stream (Temperature, Humidity, Pressure)
    ↓
Data Validation Layer
    ↓
Filtering Service (Median Filter + Adaptive Kalman Filter)
    ↓
Feature Engineering Service
    ↓
Model Service (Seasonal Decomposition + Isolation Forest + Z-score)
    ↓
Drift Detection (Kolmogorov-Smirnov Test)
    ↓
Anomaly Output CSV
    ↓
Alert Service (Slack Webhook ready)
    ↓
Metrics exposed for Grafana
```

## Project Structure

```
atmospheric_pipeline/
├── services/
│   ├── ingestion_service.py
│   ├── validation_service.py
│   ├── filtering_service.py
│   ├── feature_service.py
│   ├── model_service.py
│   ├── drift_service.py
│   ├── alert_service.py
├── config.yaml
├── main_pipeline.py
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml
├── prometheus.yml
└── dags/
    └── atmospheric_dag.py
```

## Quick Start

### Local Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample stream data (optional)
python scripts/generate_sample_stream.py

# Run pipeline
python main_pipeline.py
```

### Flask API

```bash
python app.py
```

- **GET /run** – Execute pipeline and return anomaly summary
- **GET /health** – Health check
- **GET /metrics** – Prometheus metrics for Grafana

### Docker

```bash
# Build and run app only
docker build -t atmospheric-pipeline .
docker run -p 8000:8000 atmospheric-pipeline python app.py

# Full stack (app, Airflow, Redis, Prometheus, Grafana)
docker-compose up -d

# Initialize Airflow DB (first time only)
docker-compose run airflow-webserver airflow db init
```

### Airflow DAG

Copy `dags/atmospheric_dag.py` to your Airflow dags folder, or use the docker-compose volume mount. Schedule: daily (configurable to hourly in the DAG).

## Configuration

Edit `config.yaml` for:

- Sensor bounds (temperature, humidity, pressure)
- Rolling window size
- Isolation Forest contamination
- Z-score threshold
- KS p-value threshold for drift
- Paths (input stream, master CSV, output, logs)
- Slack webhook URL (or set `SLACK_WEBHOOK_URL` env var)

## Output

- `output/anomaly_results.csv` – Anomaly results with all features and flags
- `logs/invalid_rows.log` – Logged invalid sensor rows
- API JSON response (from `/run`)
- Airflow task logs

## Ports

| Service            | Port |
|--------------------|------|
| Atmospheric App    | TBD |
| Airflow Webserver  | TBD |
| Grafana            | TBD |
| Prometheus         | TBD |
| Redis              | TBD|
