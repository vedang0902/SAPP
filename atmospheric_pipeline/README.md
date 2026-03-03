# Atmospheric Monitoring Pipeline

Production-ready modular atmospheric monitoring and anomaly detection system with hybrid predictive modeling.

## Architecture

```
Sensor Stream (CSV: timestamp, temperature, humidity, pressure)
    ↓
Validation
    ↓
Adaptive Filtering (Median + Kalman)
    ↓
Feature Engineering
    ↓
Seasonal Decomposition
    ↓
Hybrid Prediction Service (SARIMA + LSTM ensemble)
    ↓
Hybrid Anomaly Detection (IF + Z-score + Forecast Error)
    ↓
Drift Detection (Kolmogorov-Smirnov)
    ↓
Alerts + Metrics (Prometheus)
```

## Project Structure

```
atmospheric_pipeline/
├── services/
│   ├── ingestion_service.py
│   ├── validation_service.py
│   ├── filtering_service.py
│   ├── feature_service.py
│   ├── seasonal_decomposition_service.py
│   ├── prediction_service.py
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

# Airflow init runs automatically on first startup (db init + admin user).
# Login: airflow / airflow
```

### Airflow DAG

Copy `dags/atmospheric_dag.py` to your Airflow dags folder, or use the docker-compose volume mount. Schedule: daily (configurable to hourly in the DAG).

**Pipeline order:**
`ingest` → `validate` → `filter` → `feature` → `seasonal_decompose` → `prediction` → `anomaly_detection` (model_task) → `drift_detection` → `output` → `alert`

## Configuration

Edit `config.yaml` for:

- Sensor bounds (temperature, humidity, pressure)
- Rolling window size
- Isolation Forest contamination
- Z-score threshold
- **Prediction**: SARIMA order, LSTM (epochs, lookback), ensemble weights, forecast error threshold
- KS p-value threshold for drift
- Paths (input stream, master CSV, output, logs)
- Slack webhook URL (or set `SLACK_WEBHOOK_URL` env var)

---

## Hybrid Forecasting Module

### Approach

The pipeline includes a **Hybrid Predictive Modeling** module that combines:

- **SARIMA** (statsmodels SARIMAX): Statistical seasonal model capturing trend and periodicity; auto-configurable order via `config.yaml` (`sarima_order`, `seasonal_order`)
- **LSTM** (TensorFlow/Keras): Lightweight multivariate neural network for nonlinear patterns; sliding-window input with MinMax scaling
- **Ensemble**: Weighted average of both predictions (configurable via `config.yaml`)

```
Final_Prediction = w1 * SARIMA_prediction + w2 * LSTM_prediction
```

Both models train on a rolling historical window and forecast the next step(s). SARIMA produces confidence intervals; LSTM uses dropout and L2 regularization to avoid overfitting.

### Ensemble Reasoning

- **SARIMA** excels at seasonal/cyclic structure and interpretability.
- **LSTM** captures complex temporal dependencies and multivariate interactions.
- **Combining** them reduces variance and improves robustness when either model alone might falter.
- Weights default to `sarima: 0.5`, `lstm: 0.5` and can be tuned per deployment in `config.yaml`.

### Forecast Error Modeling

For each timestep:

- `forecast_error = |actual - predicted|`
- `forecast_error_mean`: rolling mean of forecast errors (window configurable via `error_window`)
- `forecast_error_std`: rolling standard deviation of forecast errors

These support adaptive anomaly thresholds and uncertainty quantification.

### Forecast-Based Anomaly Detection

An anomaly is flagged if **any** of:

- Isolation Forest flag
- Z-score > threshold
- Seasonal residual anomaly (Z-score on decomposition residual)
- Forecast error > `forecast_error_threshold`

Large forecast errors indicate the model failed to predict the observation, suggesting an anomalous event. Threshold is configurable via `prediction.forecast_error_threshold`.

### Retraining Logic

- **Initial run**: Models are trained on the available historical window (rolling).
- **Drift trigger**: When drift is detected (Kolmogorov–Smirnov p-value < threshold), `drift_service` writes `logs/drift_triggered.flag`.
- **Next run**: `prediction_service` detects the flag, retrains SARIMA and LSTM on fresh data, clears the flag, and logs the event.
- **Periodic refit**: SARIMA is refit periodically within the rolling forecast loop for accuracy; LSTM is retrained only on drift.

### Performance Logging (Prometheus)

The prediction service logs and exposes the following metrics for Prometheus scraping:

| Metric | Description |
|--------|-------------|
| `atmospheric_forecast_rmse` | Root Mean Squared Error |
| `atmospheric_forecast_mae` | Mean Absolute Error |
| `atmospheric_forecast_variance` | Forecast variance |
| `atmospheric_model_training_seconds` | Model training time (s) |

Access via `GET /metrics` when running the Flask app. Configure Prometheus to scrape the app endpoint for Grafana dashboards.

## Output

- `output/anomaly_results.csv` – Anomaly results with all features and flags
- `logs/invalid_rows.log` – Logged invalid sensor rows
- API JSON response (from `/run`)
- Airflow task logs

## Ports

| Service            | Port |
|--------------------|------|
| Atmospheric App    | 8000 |
| Airflow Webserver  | 8082 |
| Grafana            | 3000 |
| Prometheus         | 9090 |
| Redis              | 6379 |
