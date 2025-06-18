import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wfdb
from scipy.signal import decimate
from neuralforecast import NeuralForecast
from scipy.signal import savgol_filter
from forecasting import recursive_forecast

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