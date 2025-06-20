import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import wfdb
from forecasting import recursive_forecast
from neuralforecast import NeuralForecast
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import decimate

def evaluate_forecast(true_y: np.ndarray, pred_y: np.ndarray, timestamps=None, label='Модель'):
    """
    Оценка качества прогноза по массивам:
    - true_y: массив истинных значений
    - pred_y: массив предсказаний модели
    - timestamps: необязательные временные метки (массив того же размера)
    """
    assert len(true_y) == len(pred_y), f"Размеры не совпадают: true_y={len(true_y)}, pred_y={len(pred_y)}"
    
    # Метрики
    mae = np.mean(np.abs(pred_y - true_y))
    mse = np.mean((pred_y - true_y) ** 2)
    corr, _ = pearsonr(true_y, pred_y)

    # Вывод
    with open ('quality_report.txt', 'a') as f:
        f.write(f'Что значат эти метрики:\n')
        f.write(f'   ➤ MAE (Mean Absolute Error): средняя абсолютная ошибка между предсказанными и истинными значениями.\n')
        f.write(f'   ➤ MSE (Mean Squared Error): средняя квадратичная ошибка, показывает, насколько сильно предсказания отклоняются от истинных значений.\n')
        f.write(f'   ➤ Corr (Correlation): коэффициент корреляции Пирсона, показывает степень линейной зависимости между предсказанными и истинными значениями.\n')
        f.write(f'\n')
        f.write(f'📊 Оценка качества прогноза ({label}):\n')
        f.write(f'   ➤ MAE:  {mae:.5f}\n')
        f.write(f'   ➤ MSE:  {mse:.5f}\n')
        f.write(f'   ➤ Corr: {corr:.5f}\n')
    print(f'📊 Оценка качества прогноза ({label}):')
    print(f'   ➤ MAE:  {mae:.5f}')
    print(f'   ➤ MSE:  {mse:.5f}')
    print(f'   ➤ Corr: {corr:.5f}')

    # График
    plt.figure(figsize=(12, 4))
    x = timestamps if timestamps is not None else np.arange(len(true_y))
    plt.plot(x, true_y, label='Истинное значение')
    plt.plot(x, pred_y, label=f'Прогноз ({label})')
    plt.title('Сравнение предсказания с реальностью')
    plt.xlabel('Время')
    plt.ylabel('Нормализованный сигнал')
    plt.legend()
    plt.grid(True)
    plt.show()

filename = 's10_sit'
record = wfdb.rdrecord(f'../test_data/{filename}')
signal = decimate(record.p_signal[:, 0], q=5)
signal = savgol_filter(signal, 15, 3)
signal = (signal - np.mean(signal)) / np.std(signal)

df = pd.DataFrame({
    'unique_id': filename,
    'ds': pd.date_range(start='2025-01-01', periods=len(signal), freq='20ms'),
    'y': signal
})
nf = NeuralForecast.load('../models/v3') 

forecasted_signal = []

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
        
    forecasted_signal.append(forecast_segment)
    

true_y = signal[500:]  
pred_y = np.concatenate(forecasted_signal)  
print(f"Длина прогноза: {len(pred_y)}, Длина истинных значений: {len(true_y)}")
evaluate_forecast(true_y, pred_y, label='NBEATS')