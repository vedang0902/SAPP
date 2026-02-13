

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
# Automation (cron example)
############################################
# 0 2 * * * cd /app/atmospheric_pipeline && python pipeline.py
