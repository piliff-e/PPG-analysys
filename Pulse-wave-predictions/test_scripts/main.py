import wfdb
import numpy as np
import matplotlib.pyplot as plt
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

record = wfdb.rdrecord('../test_data/s1_walk')
signal = decimate(record.p_signal[:, 0], q=5)
signal = savgol_filter(signal, 15, 3)
signal = (signal - np.mean(signal)) / np.std(signal)

backup_signal = signal.copy()

full_df = pd.DataFrame({
        'unique_id': "segment",
        'ds': pd.date_range(start='2025-01-01', periods=len(backup_signal), freq='20ms'),
        'y': backup_signal
    })


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
    if len(current_segment) < 500:
        forecast_segment = forecast_segment[:len(current_segment)]
        
    alignment = dtw(forecast_segment, current_segment)
    if alignment > 5:
        signal[i:i + 500] = forecast_segment
        print(f'[!] Обнаружено отклонение на сегменте {i} - {i}+500, DTW Distance: {alignment}')
    else:
        print(f'[+] Сегмент {i} - {i}+500 в норме, DTW Distance: {alignment}')

final_df = pd.DataFrame({
    'unique_id': 'full_signal',
    'ds': pd.date_range(start='2025-01-01', periods=len(signal), freq='20ms'),
    'y': signal
})

plt.figure(figsize=(20, 10))
plt.plot(full_df['ds'], full_df['y'], label='Оригинал', color='blue')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.title('Изначальный сигнал PPG')


plt.figure(figsize=(20, 10))
plt.plot(final_df['ds'], final_df['y'], label='Оригинал', color='blue')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.title('Исправленный сигнал PPG с использованием StatsForecast')

plt.show()