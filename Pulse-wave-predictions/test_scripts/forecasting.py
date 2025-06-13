import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heartpy as hp
import wfdb
from scipy.signal import decimate
from neuralforecast.models import RNN, NBEATS
from neuralforecast import NeuralForecast
from scipy.signal import savgol_filter

def recursive_forecast(nf, start_df, repeats=10):
    forecasts = []
    current_df = start_df.copy()

    for i in range(repeats):
        nf.fit(current_df)

        forecast_df = nf.predict()

        # Удалим строки, где прогноз содержит NaN
        forecast_df_clean = forecast_df.dropna(subset=['NBEATS']).copy()

        if forecast_df_clean.empty:
            print(f"[!] Прогноз пуст на итерации {i}")
            break

        forecasts.append(forecast_df_clean)

        # Добавляем предсказанные значения как новые наблюдения
        new_block = forecast_df_clean.copy()
        new_block['y'] = new_block['NBEATS']
        current_df = pd.concat([current_df, new_block[['unique_id', 'ds', 'y']]]).reset_index(drop=True)

    full_forecast = pd.concat(forecasts).reset_index(drop=True)
    return full_forecast

# record = wfdb.rdrecord('../data/s1_walk')  # Чтение данных
# ppg_ac = decimate(record.p_signal[:, 0],q=5)  # AC-компонента (пульсовая волна)
# time = np.arange(len(ppg_ac))  # Временные метки для вашего массива
# wd, m = hp.process(ppg_ac, sample_rate=50)  # sample_rate влияет на "BPM" на графике -- измени, чтобы проверить



all_dfs = []
for filename in ['s1_walk', 's2_walk', 's3_walk']:
    record = wfdb.rdrecord(f'../test_data/{filename}')
    signal = decimate(record.p_signal[:, 0], q=5)
    signal = savgol_filter(signal, 15, 3)
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    segment_df = pd.DataFrame({
        'unique_id': filename,
        'ds': pd.date_range(start='2025-01-01', periods=len(signal), freq='20ms'),
        'y': signal
    })
    all_dfs.append(segment_df)

df = pd.concat(all_dfs).reset_index(drop=True)


nf = NeuralForecast(
    models=[NBEATS(input_size=512, h=100, max_steps=300, learning_rate=1e-3)],  # 128 прошлых точек → 50 шагов вперёд
    freq='20ms'
)

s1_df = df[df['unique_id'] == 's1_walk'].copy()

# Запускаем многократное прогнозирование
forecast_df = recursive_forecast(nf, s1_df, repeats=10)

# nf.fit(df)
# forecast_df = nf.predict(step_size=100, num_windows=20)
# metrics_df = nf.evaluate(df, metrics=['mae', 'mse'])
# print(metrics_df)
# model = RNN(
#     input_size=24,
#     # hidden_size=64,
#     # context_length=24,
#     h=12,
#     # horizon=12,
#     max_steps=3,              
#     # learning_rate=1e-3
#     enable_progress_bar=True
# )

# Визуализация
# hp.plotter(wd, m)

plt.figure(figsize=(20, 10))
plt.plot(s1_df['ds'], s1_df['y'], label='Оригинал', color='blue')
# Прогноз
plt.plot(forecast_df['ds'], forecast_df['NBEATS'], label='Прогноз', color='red', linestyle='--')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.title('Прогноз PPG с использованием StatsForecast')
plt.show()