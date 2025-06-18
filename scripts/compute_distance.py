import wfdb
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import decimate
from neuralforecast import NeuralForecast
from scipy.signal import savgol_filter
from pyts.metrics import dtw


def recursive_forecast(nf, start_df, repeats=10):
    forecasts = []
    current_df = start_df.copy()

    for i in range(repeats):
        forecast_df = nf.predict(df=current_df)

        forecast_df_clean = forecast_df.dropna(subset=['NBEATS']).copy()

        if forecast_df_clean.empty:
            print(f"[!] Прогноз пуст на итерации {i}")
            break

        forecasts.append(forecast_df_clean)

        new_block = forecast_df_clean.copy()
        new_block['y'] = new_block['NBEATS']
        current_df = pd.concat([current_df, new_block[['unique_id', 'ds', 'y']]]).reset_index(drop=True)

    full_forecast = pd.concat(forecasts).reset_index(drop=True)
    return full_forecast

record = wfdb.rdrecord('../test_data/s10_sit')
signal = decimate(record.p_signal[:, 0], q=5)
signal = savgol_filter(signal, 15, 3)
signal = (signal - np.mean(signal)) / np.std(signal)
distances = []

# df = pd.DataFrame({
#     'unique_id': "signal",
#     'ds': pd.date_range(start='2025-01-01', periods=len(signal), freq='20ms'),
#     'y': signal
# })

nf = NeuralForecast.load('../models/v3')  
# forecast_df = recursive_forecast(nf, df, repeats=5)

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

avg_distance = np.mean(distances)
print(f'Average DTW Distance: {avg_distance}')