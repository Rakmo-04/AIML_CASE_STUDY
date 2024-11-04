import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = tf.keras.models.load_model("remote_patient_model.keras")  # Updated to use the new .keras format

# Function to load and preprocess data
def load_and_preprocess_data(filepath, seq_length=50):
    data = pd.read_csv(filepath)
    
    # Scale features to the range [0, 1]
    scaler = MinMaxScaler()
    data[['heart_rate', 'blood_glucose', 'spo2', 'activity', 'temperature']] = scaler.fit_transform(
        data[['heart_rate', 'blood_glucose', 'spo2', 'activity', 'temperature']]
    )

    # Create sequences for LSTM input
    def create_sequences(data, seq_length=50):
        sequences = []
        for i in range(seq_length, len(data)):
            sequences.append(data[i - seq_length:i].values)
        return np.array(sequences)

    sequences = create_sequences(data[['heart_rate', 'blood_glucose', 'spo2', 'activity', 'temperature']], seq_length)
    return sequences, scaler

# Function to detect anomalies
def detect_anomalies(data, model, threshold):
    predictions = model.predict(data)
    mse = np.mean(np.power(data - predictions, 2), axis=(1, 2))
    anomalies = mse > threshold
    return mse, anomalies

# Load training data and set the anomaly threshold
X_train, scaler = load_and_preprocess_data("data/simulated_wearable_data.csv")
mse, _ = detect_anomalies(X_train, model, threshold=0)  # Use a placeholder threshold of 0 for initial detection

# Set the actual threshold using the 95th percentile of the MSE
threshold = np.percentile(mse, 95)  # 95th percentile for threshold

# Now detect anomalies with the calculated threshold
mse, anomalies = detect_anomalies(X_train, model, threshold=threshold)

# Plot reconstruction error with the threshold line
plt.figure(figsize=(15, 6))
plt.plot(mse, label="Reconstruction Error")
plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
plt.title("Reconstruction Error for Anomaly Detection")
plt.xlabel("Data Points")
plt.ylabel("MSE")
plt.legend()
plt.show()

# Anomaly timestamps
timestamps = pd.read_csv("data/simulated_wearable_data.csv")["timestamp"].values[50:]
anomaly_timestamps = timestamps[anomalies]
print("Anomaly detected at timestamps:", anomaly_timestamps)
