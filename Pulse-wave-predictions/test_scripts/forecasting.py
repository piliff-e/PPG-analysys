import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heartpy as hp
import wfdb
from scipy.signal import decimate
from neuralforecast.models import NBEATS
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

# record = wfdb.rdrecord('../data/s1_walk')  # Чтение данных
# ppg_ac = decimate(record.p_signal[:, 0],q=5)  # AC-компонента (пульсовая волна)
# time = np.arange(len(ppg_ac))  # Временные метки для вашего массива
# wd, m = hp.process(ppg_ac, sample_rate=50)  # sample_rate влияет на "BPM" на графике -- измени, чтобы проверить



all_dfs = []
filenames = ['s10_sit', 's11_sit', 's12_sit']
for filename in filenames:
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

test_record = wfdb.rdrecord('../test_data/s1_walk')
test_signal = decimate(test_record.p_signal[:, 0], q=5)
test_signal = savgol_filter(test_signal, 15, 3)
test_signal = (test_signal - np.mean(test_signal)) / np.std(test_signal)
raw_df = pd.DataFrame({
    'unique_id': 's1_walk',
    'ds': pd.date_range(start='2025-01-01', periods=len(test_signal), freq='20ms'),
    'y': test_signal
})

# raw_dfs = []
# for filename in ['s1_walk', 's2_walk', 's3_walk']:
#     record = wfdb.rdrecord(f'../test_data/{filename}')
#     signal = decimate(record.p_signal[:, 0], q=5)
#     signal = (signal - np.mean(signal)) / np.std(signal)
    
#     segment_df = pd.DataFrame({
#         'unique_id': filename,
#         'ds': pd.date_range(start='2025-01-01', periods=len(signal), freq='20ms'),
#         'y': signal
#     })
#     raw_dfs.append(segment_df)

# raw_df = pd.concat(raw_dfs).reset_index(drop=True)
# raw_nf = NeuralForecast(
#     models=[NBEATS(input_size=512, h=100, max_steps=500, learning_rate=1e-3)],  # 128 прошлых точек → 50 шагов вперёд
#     freq='20ms'
# ) 
nf = NeuralForecast.load('../models')  
# nf = NeuralForecast(models=[model], freq='20ms')
# nf = NeuralForecast(
#     models=[NBEATS(input_size=512, h=100, max_steps=1, learning_rate=1e-3,batch_size=8)],  # 128 прошлых точек → 50 шагов вперёд
#     freq='20ms'
# )
# nf.fit(df)

#UNCOMMENT TO TRAIN
# path_to_save = '../models/version_1.pth'
# nf.models[0].save(path_to_save)

#UNCOMMENT TO PREDICT
# s1_df = df[df['unique_id'] == 's11_sit'].copy()
forecast_df = recursive_forecast(nf, raw_df, repeats=10)

# raw_s1_df = raw_df[raw_df['unique_id'] == 's2_walk'].copy()
# raw_forecast_df = recursive_forecast(raw_nf, raw_s1_df, repeats=10)


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
plt.plot(raw_df['ds'], raw_df['y'], label='Оригинал', color='blue')
# Прогноз
plt.plot(forecast_df['ds'], forecast_df['NBEATS'], label='Прогноз', color='red', linestyle='--')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.title('Прогноз PPG с использованием StatsForecast')


# plt.figure(figsize=(20, 10))
# plt.plot(raw_s1_df['ds'], raw_s1_df['y'], label='Оригинал', color='blue')
# plt.plot(raw_forecast_df['ds'], raw_forecast_df['NBEATS'], label='Прогноз', color='red', linestyle='--')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.title('Прогноз PPG с использованием StatsForecast без предобработки')
plt.show()