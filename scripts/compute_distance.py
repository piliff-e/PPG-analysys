import numpy as np
import pandas as pd
import wfdb
from forecasting import recursive_forecast
from neuralforecast import NeuralForecast
from pyts.metrics import dtw
from scipy.signal import decimate, savgol_filter

record = wfdb.rdrecord("../test_data/s10_sit")
signal = decimate(record.p_signal[:, 0], q=5)
signal = savgol_filter(signal, 15, 3)
signal = (signal - np.mean(signal)) / np.std(signal)
distances = []

nf = NeuralForecast.load("../models/v3")

for i in range(500, len(signal), 500):
    current_segment = signal[i : i + 500]
    past_segment = signal[i - 500 : i]
    segment_df = pd.DataFrame(
        {
            "unique_id": "past_segment",
            "ds": pd.date_range(
                start="2025-01-01", periods=len(past_segment), freq="20ms"
            ),
            "y": past_segment,
        }
    )

    forecast_segment_df = recursive_forecast(nf, segment_df, repeats=5)
    forecast_segment = forecast_segment_df["NBEATS"].values

    alignment = dtw(forecast_segment, current_segment)
    distances.append(alignment)

max_distance = np.max(distances)
avg_distance = np.mean(distances)
print(f"Average DTW Distance: {avg_distance}")
print(f"Max DTW Distance: {max_distance}")


# import numpy as np
# import pandas as pd
# from pyts.metrics import dtw
# from neuralforecast import NeuralForecast
# from scipy.signal import decimate, savgol_filter
# import wfdb
#
# from preprocessing import load_and_preprocess
# from forecasting import recursive_forecast, create_history_df
#
# def compute_dtw_threshold(record_name: str,
#                           data_path: str,
#                           model_path: str,
#                           window_size: int = 500,
#                           repeats: int = 5,
#                           decimate_q: int = 5,
#                           sg_window: int = 15,
#                           sg_poly: int = 3,
#                           freq: str = "20ms") -> None:
#     """
#     Загружает сигнал record_name из data_path, предобрабатывает,
#     загружает модель NeuralForecast из model_path,
#     для каждого сегмента длины window_size считает DTW между прогнозом и реальным сегментом,
#     выводит среднее и максимум расстояний.
#     """
#     # 1. Загрузка и предобработка сигнала
#     print(f"[compute_distance] Загрузка и предобработка сигнала '{record_name}'")
#     signal, fs = load_and_preprocess(record_name, data_path,
#                                      decimate_q=decimate_q,
#                                      sg_window=sg_window,
#                                      sg_poly=sg_poly)
#     print(f"[compute_distance] Сигнал длины {len(signal)}, fs={fs} Hz")
#
#     # 2. Загрузка модели
#     try:
#         nf = NeuralForecast.load(model_path)
#         print(f"[compute_distance] Модель загружена из '{model_path}'")
#     except Exception as e:
#         print(f"[compute_distance] Ошибка загрузки модели: {e}")
#         return
#
#     distances = []
#
#     length = len(signal)
#     # Для каждого сегмента начинаем с index = window_size, шаг = window_size
#     for start in range(window_size, length, window_size):
#         # Текущий сегмент (реальный), который хотим сравнить
#         end = min(start + window_size, length)
#         current_segment = signal[start:end]
#         # Предыдущие window_size точек для истории
#         hist_start = start - window_size
#         history = signal[hist_start:start]
#
#         # Подготовка DataFrame истории
#         # unique_id можно взять фиксированным, например record_name или "past_segment"
#         # Если модель обучалась на конкретных unique_id, можно здесь указать record_name, но
#         # в compute_distance главное оценить характер расстояний.
#         df_hist = create_history_df(history, record_name, freq=freq)
#
#         # Прогноз рекурсивно repeats раз
#         forecast_df = recursive_forecast(nf, df_hist, repeats=repeats)
#         if forecast_df.empty or "NBEATS" not in forecast_df.columns:
#             print(f"[compute_distance] Прогноз пуст или нет 'NBEATS' для сегмента {hist_start}-{start}")
#             continue
#
#         pred = forecast_df["NBEATS"].values
#         # Обрезаем прогноз до длины current_segment, если нужно
#         L = min(len(pred), len(current_segment))
#         if L <= 0:
#             continue
#         pred_seg = pred[:L]
#         curr_seg = current_segment[:L]
#
#         # Вычисление DTW
#         try:
#             alignment = dtw(pred_seg, curr_seg)
#             distances.append(alignment)
#             print(f"[compute_distance] Segment {start}-{start+L}, DTW={alignment:.2f}")
#         except Exception as e:
#             print(f"[compute_distance] Ошибка DTW на сегменте {start}-{start+L}: {e}")
#             continue
#
#     if distances:
#         distances = np.array(distances)
#         avg_distance = np.mean(distances)
#         max_distance = np.max(distances)
#         std_distance = np.std(distances)
#         print(f"[compute_distance] Среднее DTW Distance: {avg_distance:.2f}")
#         print(f"[compute_distance] Max DTW Distance: {max_distance:.2f}")
#         print(f"[compute_distance] STD DTW Distance: {std_distance:.2f}")
#     else:
#         print("[compute_distance] Нет измеренных расстояний.")
#
# if __name__ == "__main__":
#     # Настройки: при необходимости изменить record_name, пути, параметры
#     record_name = "s10_sit"
#     data_path = "../test_data"
#     model_path = "../models/v3"
#     window_size = 500
#     repeats = 5
#     decimate_q = 5
#     sg_window = 15
#     sg_poly = 3
#     freq = "20ms"
#
#     compute_dtw_threshold(record_name=record_name,
#                           data_path=data_path,
#                           model_path=model_path,
#                           window_size=window_size,
#                           repeats=repeats,
#                           decimate_q=decimate_q,
#                           sg_window=sg_window,
#                           sg_poly=sg_poly,
#                           freq=freq)
