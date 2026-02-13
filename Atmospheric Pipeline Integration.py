# Project Structure
# atmospheric_pipeline/
# ├── feature_extractor.py
# ├── filters.py
# ├── pipeline.py
# ├── app.py
# └── dags/
#     └── atmospheric_dag.py

############################################
# feature_extractor.py
############################################
import pandas as pd

def extract_features(df, window=5):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    for col in ['temperature', 'humidity', 'pressure']:
        df[f'{col}_mean'] = df[col].rolling(window).mean()
        df[f'{col}_std'] = df[col].rolling(window).std()
        df[f'{col}_min'] = df[col].rolling(window).min()
        df[f'{col}_max'] = df[col].rolling(window).max()
        df[f'{col}_skew'] = df[col].rolling(window).skew()
        df[f'{col}_kurt'] = df[col].rolling(window).kurt()
        df[f'{col}_delta'] = df[col].diff()
        df[f'{col}_gradient'] = df[col].diff(window)
        df[f'{col}_energy'] = (df[f'{col}_delta']**2).rolling(window).mean()

    df['temp_rh_ratio'] = df['temperature'] / (df['humidity'] + 1e-6)
    df['pressure_temp_product'] = df['pressure'] * df['temperature']

    return df.dropna().reset_index(drop=True)

############################################
# filters.py
############################################
import statistics

class KalmanFilter:
    def __init__(self, Q, R, P, x):
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def update(self, measurement):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P
        return self.x


def median_filter(window, value, size):
    window.append(value)
    if len(window) > size:
        window.pop(0)
    return statistics.median(window)


def apply_median_kalman(series, window=5, params=None):
    if params is None:
        params = {"Q":0.01,"R":0.5,"P":1.0}

    s = series.ffill().bfill()
    kf = KalmanFilter(params['Q'],params['R'],params['P'],s.iloc[0])

    buf, out = [], []

    for v in s:
        med = median_filter(buf,float(v),window)
        out.append(kf.update(med))

    return out

############################################
# pipeline.py (Main Integration Logic)
############################################
import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly
from sklearn.ensemble import IsolationForest

from feature_extractor import extract_features
from filters import apply_median_kalman


def fetch_weather(start, end, lat, lon):
    loc = Point(lat, lon)
    data = Hourly(loc, start, end)
    return data.fetch()


def run_pipeline():
    start = datetime(2025,1,1)
    end = datetime(2025,9,9)

    df = fetch_weather(start,end,18.5204,73.8567)
    df = df[['temp','pres','rhum']]
    df.columns = ['temperature','pressure','humidity']

    df['temp_f'] = apply_median_kalman(df['temperature'])
    df['pres_f'] = apply_median_kalman(df['pressure'])
    df['hum_f'] = apply_median_kalman(df['humidity'])

    feat_df = pd.DataFrame({
        'timestamp': df.index,
        'temperature': df['temp_f'],
        'pressure': df['pres_f'],
        'humidity': df['hum_f']
    })

    feat_df = extract_features(feat_df)

    model = IsolationForest(contamination=0.04)
    X = feat_df[['temperature','pressure','humidity']]

    feat_df['anomaly'] = model.fit_predict(X)

    feat_df.to_csv('output/anomaly_results.csv',index=False)
    return feat_df


if __name__ == '__main__':
    run_pipeline()

############################################
# app.py (Flask API)
############################################
from flask import Flask, jsonify
from pipeline import run_pipeline

app = Flask(__name__)

@app.route("/run", methods=['GET'])
def run():
    df = run_pipeline()
    outliers = df[df['anomaly']==-1]

    return jsonify({
        "total_records": len(df),
        "anomalies": len(outliers),
        "sample": outliers.head(5).to_dict()
    })


@app.route("/health")
def health():
    return {"status":"ok"}


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)

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
