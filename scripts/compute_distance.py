import wfdb
import numpy as np
import pandas as pd
from scipy.signal import decimate
from neuralforecast import NeuralForecast
from scipy.signal import savgol_filter
from pyts.metrics import dtw
from forecasting import recursive_forecast

record = wfdb.rdrecord('../test_data/s10_sit')
signal = decimate(record.p_signal[:, 0], q=5)
signal = savgol_filter(signal, 15, 3)
signal = (signal - np.mean(signal)) / np.std(signal)
distances = []

nf = NeuralForecast.load('../models/v3')  

for i in range(500, len(signal), 500):
    current_segment = signal[i:i + 500]
    past_segment = signal[i-500:i]
    segment_df = pd.DataFrame({
        'unique_id': "past_segment",
        'ds': pd.date_range(start='2025-01-01', periods=len(past_segment), freq='20ms'),
        'y': past_segment
    })

    forecast_segment_df = recursive_forecast(nf, segment_df, repeats=5)
    forecast_segment = forecast_segment_df['NBEATS'].values

    alignment = dtw(forecast_segment, current_segment)
    distances.append(alignment)

max_distance = np.max(distances)
avg_distance = np.mean(distances)
print(f'Average DTW Distance: {avg_distance}')
print(f'Max DTW Distance: {max_distance}')