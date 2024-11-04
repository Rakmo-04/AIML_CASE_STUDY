import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_and_preprocess_data(filepath, seq_length=50):
    # Load dataset
    data = pd.read_csv(filepath)
    scaler = MinMaxScaler()
    data[['heart_rate', 'blood_glucose', 'spo2', 'activity', 'temperature']] = scaler.fit_transform(
        data[['heart_rate', 'blood_glucose', 'spo2', 'activity', 'temperature']]
    )
    
    # Prepare sequences for LSTM
    def create_sequences(data, seq_length=50):
        sequences = []
        for i in range(seq_length, len(data)):
            sequences.append(data[i-seq_length:i].values)
        return np.array(sequences)

    sequences = create_sequences(data[['heart_rate', 'blood_glucose', 'spo2', 'activity', 'temperature']])
    return sequences, scaler

sequences, scaler = load_and_preprocess_data("data/simulated_wearable_data.csv")
print("Data preprocessed with sequences shape:", sequences.shape)
