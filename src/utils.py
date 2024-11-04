import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Generate a timestamp range for data collection
timestamps = pd.date_range("2024-01-01", periods=5000, freq="H")

# Simulate wearable data with anomalies
np.random.seed(42)
heart_rate = np.random.normal(75, 5, len(timestamps))  # Simulate heart rate around 75 bpm
blood_glucose = np.random.normal(100, 15, len(timestamps))  # mg/dL
spo2 = np.random.normal(95, 2, len(timestamps))  # oxygen saturation in %
activity = np.random.normal(50, 20, len(timestamps))  # arbitrary activity level
temperature = np.random.normal(37, 0.5, len(timestamps))  # body temperature in °C

# Inject some anomalies
heart_rate[1000:1010] = 150  # Simulate a tachycardia event
blood_glucose[2000:2010] = 50  # Low glucose event
spo2[3000:3010] = 85  # Low SpO2 event
temperature[4000:4010] = 39  # Fever event

# Create a DataFrame
data = pd.DataFrame({
    "timestamp": timestamps,
    "heart_rate": heart_rate,
    "blood_glucose": blood_glucose,
    "spo2": spo2,
    "activity": activity,
    "temperature": temperature
})

# Save to CSV
data.to_csv("data/simulated_wearable_data.csv", index=False)
print("Dataset saved as simulated_wearable_data.csv")
