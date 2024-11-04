import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data function
def load_and_preprocess_data(filepath, seq_length=50):
    data = pd.read_csv(filepath)

    # Scale features to the range [0, 1]
    scaler = MinMaxScaler()
    data[['heart_rate', 'blood_glucose', 'spo2', 'activity', 'temperature']] = scaler.fit_transform(
        data[['heart_rate', 'blood_glucose', 'spo2', 'activity', 'temperature']]
    )

    # Function to create sequences for LSTM
    def create_sequences(data, seq_length=50):
        sequences = []
        for i in range(seq_length, len(data)):
            sequences.append(data[i-seq_length:i].values)
        return np.array(sequences)

    # Generate sequences from the data
    sequences = create_sequences(data[['heart_rate', 'blood_glucose', 'spo2', 'activity', 'temperature']], seq_length)
    return sequences, scaler

# Build and train model function
def build_and_train_model(X_train, epochs=10, batch_size=32):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation="relu", return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dense(X_train.shape[2])  # Adjust output units to match feature count
    ])
    
    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    
    # Train the model with validation split
    model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    return model

# Load data and preprocess
X_train, scaler = load_and_preprocess_data("data/simulated_wearable_data.csv")

# Build, train, and save the model
model = build_and_train_model(X_train)
model.save("remote_patient_model.keras")
print("Model trained and saved as remote_patient_model.keras")
