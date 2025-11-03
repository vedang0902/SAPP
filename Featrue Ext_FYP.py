#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-03T17:14:59.571Z
"""

import pandas as pd
import numpy as np

def extract_features(df, window=5):
    """
    Extracts temporal and statistical features for atmospheric data.
    Inputs:
        df: DataFrame with columns ['timestamp', 'temperature', 'humidity', 'pressure']
        window: rolling window size for statistics (default = 5)
    Returns:
        DataFrame with added feature columns
    """

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Rolling mean, std, min, max, skew, kurtosis
    for col in ['temperature', 'humidity', 'pressure']:
        df[f'{col}_mean'] = df[col].rolling(window).mean()
        df[f'{col}_std'] = df[col].rolling(window).std()
        df[f'{col}_min'] = df[col].rolling(window).min()
        df[f'{col}_max'] = df[col].rolling(window).max()
        df[f'{col}_skew'] = df[col].rolling(window).skew()
        df[f'{col}_kurt'] = df[col].rolling(window).kurt()

        # Rate of change (first derivative)
        df[f'{col}_delta'] = df[col].diff()

        # Gradient over the window (trend estimation)
        df[f'{col}_gradient'] = df[col].diff(window)

        # Rolling energy (for detecting turbulent fluctuations)
        df[f'{col}_energy'] = df[f'{col}_delta'] ** 2
        df[f'{col}_energy'] = df[f'{col}_energy'].rolling(window).mean()

    # Composite features
    df['temp_rh_ratio'] = df['temperature'] / (df['humidity'] + 1e-6)
    df['pressure_temp_product'] = df['pressure'] * df['temperature']

    # Drop initial NaNs created by rolling operations
    df = df.dropna().reset_index(drop=True)

    return df

# Example usage:
if __name__ == "__main__":
    # Example sensor data
    data = {
        'timestamp': pd.date_range(start='2025-10-30', periods=100, freq='S'),
        'temperature': np.random.normal(25, 0.5, 100),
        'humidity': np.random.normal(60, 1.5, 100),
        'pressure': np.random.normal(1013, 0.8, 100)
    }
    df = pd.DataFrame(data)

    features_df = extract_features(df)
    print(features_df.head(10))

    # End example usage