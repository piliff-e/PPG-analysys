import numpy as np
import pandas as pd
import wfdb
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from scipy.signal import decimate, savgol_filter

all_dfs = []
filenames = ["s10_sit", "s11_sit", "s12_sit"]
for filename in filenames:
    record = wfdb.rdrecord(f"../test_data/{filename}")
    signal = decimate(record.p_signal[:, 0], q=5)
    signal = savgol_filter(signal, 15, 3)
    signal = (signal - np.mean(signal)) / np.std(signal)

    segment_df = pd.DataFrame(
        {
            "unique_id": filename,
            "ds": pd.date_range(start="2025-01-01", periods=len(signal), freq="20ms"),
            "y": signal,
        }
    )
    all_dfs.append(segment_df)

df = pd.concat(all_dfs).reset_index(drop=True)

nf = NeuralForecast(
    models=[
        NBEATS(input_size=500, h=100, max_steps=500, learning_rate=1e-3, batch_size=16)
    ],
    freq="20ms",
)

# ЧТОБЫ ДООБУЧИТЬ РАССКОММЕНТИРУЙ И УКАЖИ ПУТЬ К НУЖНОЙ МОДЕЛИ
# nf = NeuralForecast.load('../models/v1')

for i in range(10):
    nf.fit(df)

nf.save(path="../models/v3", model_index=[0], overwrite=True)


# import pandas as pd
# from forecasting import create_history_df
# from neuralforecast import NeuralForecast
# from neuralforecast.models import NBEATS
# from preprocessing import load_and_preprocess
#
#
# def train_model(
#     filenames: list[str],
#     data_path: str,
#     model_save_path: str,
#     decimate_q: int = 5,
#     sg_window: int = 15,
#     sg_poly: int = 3,
#     freq: str = "20ms",
#     input_size: int = 500,
#     horizon: int = 100,
#     max_steps: int = 500,
#     learning_rate: float = 1e-3,
#     batch_size: int = 16,
#     epochs: int = 10,
# ) -> None:
#     """
#     Тренирует NeuralForecast модель NBEATS на списке записей filenames из data_path.
#     Сохраняет обученную модель в model_save_path.
#     Параметры:
#       - filenames: список имён записей без расширения, например ['s10_sit','s11_sit','s12_sit']
#       - data_path: путь к папке с WFDB-файлами
#       - model_save_path: путь к директории, куда сохранять модель (например '../models/v3')
#       - decimate_q, sg_window, sg_poly: параметры предобработки
#       - freq: строка частоты, должна совпадать с выборкой ds, напр. "20ms"
#       - input_size, horizon: параметры модели NBEATS
#       - max_steps, learning_rate, batch_size: гиперпараметры обучения NBEATS
#       - epochs: число проходов по всем данным (вызов nf.fit df несколько раз)
#     """
#     all_dfs = []
#
#     # 1. Сбор обучающего DataFrame
#     for filename in filenames:
#         print(f"[train_basic_model] Загрузка и предобработка '{filename}'")
#         signal, fs = load_and_preprocess(
#             filename,
#             data_path,
#             decimate_q=decimate_q,
#             sg_window=sg_window,
#             sg_poly=sg_poly,
#         )
#         # Проверка: fs и freq должны соответствовать. Здесь не проверяем автоматически, предполагаем согласовано.
#         N = len(signal)
#         # Строим DataFrame: unique_id = filename, ds = pd.date_range, y = signal
#         df_segment = create_history_df(
#             signal, filename, freq=freq, start_time="2025-01-01"
#         )
#         all_dfs.append(df_segment)
#
#     if not all_dfs:
#         print("[train_basic_model] Нет данных для обучения.")
#         return
#
#     df = pd.concat(all_dfs).reset_index(drop=True)
#     print(f"[train_basic_model] Сформирован DataFrame для обучения, размер {df.shape}")
#
#     # 2. Инициализация модели NeuralForecast
#     print("[train_basic_model] Инициализация модели NeuralForecast NBEATS")
#     nf = NeuralForecast(
#         models=[
#             NBEATS(
#                 input_size=input_size,
#                 h=horizon,
#                 max_steps=max_steps,
#                 learning_rate=learning_rate,
#                 batch_size=batch_size,
#             )
#         ],
#         freq=freq,
#     )
#
#     # 3. Обучение: несколько эпох (несколько вызовов fit)
#     print(f"[train_basic_model] Начало обучения, epochs={epochs}")
#     for epoch in range(1, epochs + 1):
#         nf.fit(df)
#         print(f"[train_basic_model] Эпоха {epoch}/{epochs} пройдена")
#
#     # 4. Сохранение модели
#     nf.save(path=model_save_path, model_index=[0], overwrite=True)
#     print(f"[train_basic_model] Модель сохранена в '{model_save_path}'")
#
#
# if __name__ == "__main__":
#     # Параметры: при необходимости изменить
#     filenames = ["s10_sit", "s11_sit", "s12_sit"]
#     data_path = "../test_data"
#     model_save_path = "../models/v3"
#     decimate_q = 5
#     sg_window = 15
#     sg_poly = 3
#     freq = "20ms"
#     input_size = 500
#     horizon = 100
#     max_steps = 500
#     learning_rate = 1e-3
#     batch_size = 16
#     epochs = 10
#
#     train_model(
#         filenames=filenames,
#         data_path=data_path,
#         model_save_path=model_save_path,
#         decimate_q=decimate_q,
#         sg_window=sg_window,
#         sg_poly=sg_poly,
#         freq=freq,
#         input_size=input_size,
#         horizon=horizon,
#         max_steps=max_steps,
#         learning_rate=learning_rate,
#         batch_size=batch_size,
#         epochs=epochs,
#     )
