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

signal_to_compare = decimate(record.p_signal[513:1013, 0], q=5)
signal_to_compare = savgol_filter(signal_to_compare, 15, 3)
signal_to_compare = (signal_to_compare - np.mean(signal_to_compare)) / np.std(signal_to_compare)

compare_df = pd.DataFrame({
    'unique_id': "signal_to_compare",
    'ds': pd.date_range(start='2025-01-01', periods=len(signal_to_compare), freq='20ms'),
    'y': signal_to_compare
})

df = pd.DataFrame({
    'unique_id': "signal",
    'ds': pd.date_range(start='2025-01-01', periods=len(signal), freq='20ms'),
    'y': signal
})

nf = NeuralForecast.load('../models/v1')  
forecast_df = recursive_forecast(nf, df, repeats=5)


forecast_to_compare = forecast_df['NBEATS']
alignment = dtw(forecast_to_compare, signal_to_compare)
print(f'DTW Distance: {alignment}')












