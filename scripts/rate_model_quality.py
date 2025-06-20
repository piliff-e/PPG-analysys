import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import wfdb
from forecasting import recursive_forecast
from neuralforecast import NeuralForecast
import pandas as pd
from scipy.signal import savgol_filter, decimate


def evaluate_forecast(true_y: np.ndarray, pred_y: np.ndarray, timestamps=None, label='Модель'):
    assert len(true_y) == len(pred_y), f"Размеры не совпадают: true_y={len(true_y)}, pred_y={len(pred_y)}"
    
    # Метрики для модели
    mae = np.mean(np.abs(pred_y - true_y))
    mse = np.mean((pred_y - true_y) ** 2)
    corr, _ = pearsonr(true_y, pred_y)

    with open('quality_report.txt', 'a') as f:
        f.write(f'📊 Оценка качества прогноза ({label}):\n')
        f.write(f'   ➤ MAE:  {mae:.5f}\n')
        f.write(f'   ➤ MSE:  {mse:.5f}\n')
        f.write(f'   ➤ Corr: {corr:.5f}\n')

    print(f'📊 Оценка качества прогноза ({label}):')
    print(f'   ➤ MAE:  {mae:.5f}')
    print(f'   ➤ MSE:  {mse:.5f}')
    print(f'   ➤ Corr: {corr:.5f}')

    # График
    plt.figure(figsize=(14, 5))
    x = timestamps if timestamps is not None else np.arange(len(true_y))
    plt.plot(x, true_y, label='Истинное значение', linewidth=1.2)
    plt.plot(x, pred_y, label=f'Прогноз ({label})', linewidth=1.2)
    plt.title('Сравнение предсказания с реальностью')
    plt.xlabel('Время')
    plt.ylabel('Сигнал')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === Загрузка и предобработка ===
filename = 's10_sit'
record = wfdb.rdrecord(f'../test_data/{filename}')
signal_raw = record.p_signal[:, 0]
signal = decimate(signal_raw, q=5)
signal = savgol_filter(signal, 15, 3)

# Сохраняем параметры для обратной нормализации
mean, std = np.mean(signal), np.std(signal)
signal = (signal - mean) / std

nf = NeuralForecast.load('../models/v4')

forecasted_signal = []
start_time = pd.to_datetime('2025-01-01')

# Прогноз
for i in range(500, len(signal), 500):
    past_segment = signal[i-500:i]
    segment_start = start_time + pd.Timedelta(milliseconds=20 * (i - 500))

    segment_df = pd.DataFrame({
        'unique_id': 'past_segment',
        'ds': pd.date_range(start=segment_start, periods=500, freq='20ms'),
        'y': past_segment
    })

    forecast_segment_df = recursive_forecast(nf, segment_df, repeats=5)

    forecast_segment = forecast_segment_df['NBEATS'].values
    if len(forecast_segment) > len(signal) - i:
        forecast_segment = forecast_segment[:len(signal) - i]

    forecasted_signal.append(forecast_segment)

# Склеиваем результат
pred_y = np.concatenate(forecasted_signal)
true_y = signal[500:500 + len(pred_y)]  
timestamps = np.arange(len(true_y)) * 20  


print(f"Длина прогноза: {len(pred_y)}, Длина истинных значений: {len(true_y)}")
evaluate_forecast(true_y, pred_y, timestamps=timestamps, label='NBEATS')