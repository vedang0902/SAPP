#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-03T17:13:42.151Z
"""

# Core libraries
import os
import importlib.util
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Meteostat library
from meteostat import Point, Hourly

# Local filters
from kalmedian import median_filter, KalmanFilter


def load_feature_extractor():
    """Dynamically load extract_features from 'Featrue Ext_FYP.py' file."""
    module_path = os.path.join(os.path.dirname(__file__), "Featrue Ext_FYP.py")
    spec = importlib.util.spec_from_file_location("feature_extraction", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.extract_features


def apply_median_kalman(series, window_size=5, kf_params=None):
    """Apply median filter followed by Kalman filter over a pandas Series."""
    if kf_params is None:
        kf_params = {
            "process_variance": 0.01,
            "measurement_variance": 0.5,
            "estimated_error": 1.0,
        }

    non_na_first = series.dropna()
    if non_na_first.empty:
        return series.copy()

    kf = KalmanFilter(
        process_variance=kf_params["process_variance"],
        measurement_variance=kf_params["measurement_variance"],
        estimated_error=kf_params["estimated_error"],
        initial_value=float(non_na_first.iloc[0]),
    )

    window = []
    filtered_values = []
    filled_series = series.ffill().bfill()
    for value in filled_series.values:
        med = median_filter(window, float(value), window_size)
        filtered_values.append(kf.update(med))

    return pd.Series(filtered_values, index=series.index)


def main():
    # Time period (shorter range = faster fetch)
    start = datetime(2025, 1, 1)
    end = datetime(2025, 9, 9)

    # Pune coordinates
    location = Point(18.5204, 73.8567)

    # Fetch hourly weather data
    data = Hourly(location, start, end)
    df = data.fetch()

    # Check available columns
    print("Available columns:", df.columns.tolist())

    # Select required features (if available)
    needed = ["temp", "pres", "rhum"]
    available = [c for c in needed if c in df.columns]
    if not available:
        raise RuntimeError("Required columns not found in Meteostat response: temp/pres/rhum")

    df = df[available].copy()

    # Rename for clarity
    rename_map = {"temp": "Temperature", "pres": "Pressure", "rhum": "Humidity"}
    df.rename(columns=rename_map, inplace=True)

    print("\nCleaned DataFrame:")
    print(df.head())

    # 1) Median + Kalman filtering for each variable
    kf_params_temp = {"process_variance": 0.01, "measurement_variance": 0.5, "estimated_error": 1.0}
    kf_params_pres = {"process_variance": 0.05, "measurement_variance": 1.0, "estimated_error": 1.0}
    kf_params_hum = {"process_variance": 0.02, "measurement_variance": 1.0, "estimated_error": 1.0}

    df["Temperature_filt"] = apply_median_kalman(df["Temperature"], window_size=5, kf_params=kf_params_temp)
    df["Pressure_filt"] = apply_median_kalman(df["Pressure"], window_size=5, kf_params=kf_params_pres)
    df["Humidity_filt"] = apply_median_kalman(df["Humidity"], window_size=5, kf_params=kf_params_hum)

    # Prepare for feature extraction
    df_for_features = pd.DataFrame(
        {
            "timestamp": df.index,
            "temperature": df["Temperature_filt"].values,
            "humidity": df["Humidity_filt"].values,
            "pressure": df["Pressure_filt"].values,
        }
    )

    extract_features = load_feature_extractor()
    features_df = extract_features(df_for_features, window=5)

    # 2) Isolation Forest on filtered primary variables
    model = IsolationForest(contamination=0.04, random_state=42)
    X = features_df[["temperature", "pressure", "humidity"]]
    features_df["anomaly"] = model.fit_predict(X)
    features_df["anomaly_flag"] = features_df["anomaly"].apply(lambda x: "Outlier" if x == -1 else "Normal")

    # 3) Visualize anomalies for each variable (filtered series) in one window
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].scatter(features_df["timestamp"], features_df["temperature"], c=features_df["anomaly"], cmap="coolwarm", s=10)
    axes[0].set_title("Temperature (filtered) Anomaly Detection - Isolation Forest")
    axes[0].set_ylabel("Temperature (°C)")

    axes[1].scatter(features_df["timestamp"], features_df["pressure"], c=features_df["anomaly"], cmap="coolwarm", s=10)
    axes[1].set_title("Pressure (filtered) Anomaly Detection - Isolation Forest")
    axes[1].set_ylabel("Pressure (hPa)")

    axes[2].scatter(features_df["timestamp"], features_df["humidity"], c=features_df["anomaly"], cmap="coolwarm", s=10)
    axes[2].set_title("Humidity (filtered) Anomaly Detection - Isolation Forest")
    axes[2].set_xlabel("Timestamp")
    axes[2].set_ylabel("Humidity (%RH)")

    fig.tight_layout()
    plt.show()

    # 4) Print anomaly summary
    outliers = features_df[features_df["anomaly"] == -1]
    print(f"\nTotal anomalies detected: {len(outliers)}")
    print(outliers.head())


if __name__ == "__main__":
    main()