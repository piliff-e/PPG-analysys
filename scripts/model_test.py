import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wfdb
from scipy.signal import decimate
from neuralforecast import NeuralForecast
from scipy.signal import savgol_filter

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


filename = 's1_walk'
record = wfdb.rdrecord(f'../test_data/{filename}')
full_signal = decimate(record.p_signal[:, 0], q=5)
signal = decimate(record.p_signal[:512, 0], q=5)
signal = savgol_filter(signal, 15, 3)
signal = (signal - np.mean(signal)) / np.std(signal)
full_signal = savgol_filter(full_signal, 15, 3)
full_signal = (full_signal - np.mean(full_signal)) / np.std(full_signal)
full_df = pd.DataFrame({
    'unique_id': 'full_signal',
    'ds': pd.date_range(start='2025-01-01', periods=len(full_signal), freq='20ms'),
    'y': full_signal
})

df = pd.DataFrame({
    'unique_id': filename,
    'ds': pd.date_range(start='2025-01-01', periods=len(signal), freq='20ms'),
    'y': signal
})

nf = NeuralForecast.load('../models/v1')  
forecast_df = recursive_forecast(nf, df, repeats=50)

plt.figure(figsize=(20, 10))
plt.plot(full_df['ds'], full_df['y'], label='Оригинал', color='blue')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.title('Прогноз PPG с использованием StatsForecast')
# Прогноз
plt.figure(figsize=(20, 10))
plt.plot(full_df['ds'], full_df['y'], label='Оригинал', color='blue')
plt.plot(forecast_df['ds'], forecast_df['NBEATS'], label='Прогноз', color='red', linestyle='--')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.title('Прогноз PPG с использованием StatsForecast')
plt.show()