# ==============================================================
# Smart Atmospheric Profiling Payload (SAPP)
# Median + Kalman Filter for Sensor Data (Temperature / Humidity / Pressure)
# Author: Vedang Kane
# --------------------------------------------------------------
# Median filter removes sudden spikes and impulse noise.
# Kalman filter smooths the data to estimate the true sensor value.
# ==============================================================

import statistics
import random
import time

# --------------------------------------------------------------
# 1. Median Filter Implementation
# --------------------------------------------------------------
def median_filter(data_window, new_value, window_size):
    """
    Applies a median filter to the incoming sensor data stream.

    Parameters:
    -----------
    data_window (list): Rolling window of recent sensor readings
    new_value (float): Latest sensor reading
    window_size (int): Size of the median filter window (e.g., 3, 5)

    Returns:
    --------
    float: Median-filtered sensor value
    """
    # Add the new reading to the window
    data_window.append(new_value)

    # Keep only the last 'window_size' readings
    if len(data_window) > window_size:
        data_window.pop(0)

    # Compute and return the median
    return statistics.median(data_window)


# --------------------------------------------------------------
# 2. Kalman Filter Implementation
# --------------------------------------------------------------
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, estimated_error, initial_value):
        """
        Initializes the Kalman Filter with given parameters.

        Parameters:
        -----------
        process_variance (float): Q - Process noise variance
        measurement_variance (float): R - Measurement noise variance
        estimated_error (float): P - Initial estimation uncertainty
        initial_value (float): x - Initial estimated state
        """
        self.Q = process_variance
        self.R = measurement_variance
        self.P = estimated_error
        self.x = initial_value

    def update(self, measurement):
        """
        Updates the Kalman Filter with a new measurement.

        Parameters:
        -----------
        measurement (float): The current sensor reading

        Returns:
        --------
        float: Updated (filtered) sensor estimate
        """
        # 1. Prediction step: estimate uncertainty
        self.P = self.P + self.Q

        # 2. Kalman gain
        K = self.P / (self.P + self.R)

        # 3. Update estimate
        self.x = self.x + K * (measurement - self.x)

        # 4. Update estimation uncertainty
        self.P = (1 - K) * self.P

        return self.x


# --------------------------------------------------------------
# 3. Example Simulation
# --------------------------------------------------------------
if __name__ == "__main__":
    # Create a Kalman filter instance (for temperature)
    kf = KalmanFilter(
        process_variance=0.01,       # Q - small value for slow-changing temperature
        measurement_variance=0.5,    # R - noise variance from sensor specs
        estimated_error=1.0,         # P - initial estimation uncertainty
        initial_value=25.0           # x - starting value
    )

    # Initialize median filter window
    median_window = []
    window_size = 5  # 5-sample window median filter

    # Simulate noisy temperature data
    true_temperature = 25.0
    print("Time | Raw (°C) | Median (°C) | Kalman Filtered (°C)")
    print("----------------------------------------------------")

    for t in range(1, 21):
        # Simulate random noise and occasional spike
        noisy_value = true_temperature + random.uniform(-1.0, 1.0)
        if t == 7 or t == 13:
            noisy_value += random.uniform(3.0, 5.0)  # simulate sudden spike

        # Apply Median filter
        median_value = median_filter(median_window, noisy_value, window_size)

        # Apply Kalman filter on median-filtered output
        kalman_value = kf.update(median_value)

        # Print results
        print(f"{t:02d}   | {noisy_value:8.2f} | {median_value:10.2f} | {kalman_value:19.2f}")

        time.sleep(0.5)
