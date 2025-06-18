"""
import matplotlib.pyplot as plt
import numpy as np

# Импорт DTW-функций:
from bad_segments_detection.sliding_window.sliding_window_with_dtw import (
    find_bad_by_dtw,
    sliding_window_dtw,
)
from forecasting.building_good_segments import (
    build_good_segments_with_model,
)  # или соответствующий импорт
from neuralforecast import NeuralForecast
from replacement.replace import analyze_with_heartpy, plot_signals, replace_segments
from shared import load_ppg


def main():
    # Параметры
    record_name = "s13_sit"  # пример
    data_path = "test_data"
    model_path = "models"  # папка с закешированной моделью
    N = 512
    K = 100
    step = 50
    threshold_factor = 2.0

    # 1) Загрузка сигнала
    signal, fs = load_ppg(f"{data_path}/{record_name}")
    print(f"Загружен сигнал {record_name}, длина {len(signal)}, fs={fs}")

    # 2) Загрузка модели
    try:
        nf = NeuralForecast.load(model_path)
    except Exception as e:
        print(f"[Error] Не удалось загрузить модель из {model_path}: {e}")
        return
    print("Модель загружена")

    # 3) Скользящий расчёт DTW-ошибки
    print("Вычисляем средние DTW-расстояния по скользящему окну...")
    avg_dtw = sliding_window_dtw(signal, nf, record_name, fs, N=N, K=K, step=step)
    # Визуализируем avg_dtw
    t = np.arange(len(signal)) / fs
    plt.figure(figsize=(10, 3))
    plt.plot(t, avg_dtw, label="avg DTW")
    # пороговая линия
    nonzero = avg_dtw[avg_dtw > 0]
    if nonzero.size > 0:
        thr = threshold_factor * np.std(nonzero)
        plt.axhline(
            thr, color="red", linestyle="--", label=f"порог {threshold_factor}·std"
        )
    plt.xlabel("Время (с)")
    plt.title("Среднее DTW-расстояние прогноза к оригиналу")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) Поиск плохих сегментов по DTW
    bad_segments = find_bad_by_dtw(avg_dtw, threshold_factor=threshold_factor)
    print("Найдены плохие сегменты по DTW:", bad_segments)

    # 5) Построение «хороших» сегментов через прогнозную модель
    # Здесь можно использовать build_good_segments_with_model или рекурсивную версию
    good_info = build_good_segments_with_model(
        signal, bad_segments, nf, record_name, fs, N=N, K=K
    )
    print(
        "Собраны хорошие сегменты (start, end, len):",
        [(s, e, len(arr)) for s, e, arr in good_info],
    )

    # 6) Подготовка списков для замены
    filtered_bad = []
    good_arrays = []
    for s, e, arr in good_info:
        filtered_bad.append((s, e))
        good_arrays.append(arr)
    if not filtered_bad:
        print("Нет сегментов с достаточной историей или прогноз не получился.")
        return

    # 7) Замена сегментов в сигнале
    replaced = replace_segments(signal, filtered_bad, good_arrays)

    # 8) Визуализация оригинал vs заменённый
    plot_signals(signal, replaced, fs, filtered_bad)

    # 9) При желании: HeartPy-анализ
    analyze_with_heartpy(signal, fs, replaced)


if __name__ == "__main__":
    main()
"""

import matplotlib.pyplot as plt
import numpy as np
from bad_segments_detection.sliding_window.sliding_window_with_dtw import (
    find_bad_by_dtw,
    sliding_window_dtw,
)
from forecasting.building_good_segments import build_good_segments_with_model
from neuralforecast import NeuralForecast
from replacement.replace import analyze_with_heartpy, plot_signals, replace_segments
from shared import load_ppg


def main():
    record_name = "s13_sit"
    data_path = "test_data"
    model_path = "models"
    N = 512
    K = 100
    step = 50
    threshold_factor = 2.0

    # 1) Загрузка сигнала
    signal, fs = load_ppg(f"{data_path}/{record_name}")
    print(f"Загружен сигнал {record_name}, длина {len(signal)}, fs={fs}")

    # 2) Загрузка модели
    try:
        nf = NeuralForecast.load(model_path)
    except Exception as e:
        print(f"[Error] Не удалось загрузить модель из {model_path}: {e}")
        return
    print("Модель загружена")

    # ПРОВЕРКА
    # 1) Подготовка одного окна для теста
    history = signal[:N]  # первые N точек
    import pandas as pd

    # Формируем DataFrame как в sliding_window_dtw
    base_time = pd.Timestamp("2025-01-01")
    freq_ms = int(round(1000.0 / fs)) if fs and fs > 0 else 20
    ds = pd.date_range(start=base_time, periods=N, freq=f"{freq_ms}ms")
    df_win = pd.DataFrame({"unique_id": record_name, "ds": ds, "y": history})

    # 2) Вызываем predict
    try:
        forecast_df = nf.predict(df=df_win)
        print("forecast_df.head():\n", forecast_df.head())
        print("Колонки:", forecast_df.columns)
        print("Размер:", len(forecast_df))
        # если есть колонка 'NBEATS':
        if "NBEATS" in forecast_df.columns:
            pred = forecast_df["NBEATS"].values
            print("pred первые 10:", pred[:10])
            print("pred длина:", len(pred))
        else:
            print("Нет колонки 'NBEATS' в forecast_df")
    except Exception as e:
        print("Ошибка nf.predict на тестовом окне:", e)

    # 3) Скользящий расчёт DTW-ошибки
    print("Вычисляем средние DTW-расстояния по скользящему окну...")
    avg_dtw = sliding_window_dtw(signal, nf, record_name, fs, N=N, K=K, step=step)

    # 4) Визуализация avg_dtw
    t = np.arange(len(signal)) / fs
    plt.figure(figsize=(10, 3))
    plt.plot(t, avg_dtw, label="avg DTW")
    nonzero = avg_dtw[avg_dtw > 0]
    if nonzero.size > 0:
        thr = threshold_factor * np.std(nonzero)
        plt.axhline(
            thr, color="red", linestyle="--", label=f"порог {threshold_factor}·std"
        )
    plt.xlabel("Время (с)")
    plt.title("Среднее DTW-расстояние прогноза к оригиналу")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5) Поиск плохих сегментов по DTW
    bad_segments = find_bad_by_dtw(avg_dtw, threshold_factor=threshold_factor)
    print("Найдены плохие сегменты по DTW:", bad_segments)

    # 6) Построение «хороших» сегментов
    good_info = build_good_segments_with_model(
        signal, bad_segments, nf, record_name, fs, N=N, K=K
    )
    print(
        "Собраны хорошие сегменты (start, end, len):",
        [(s, e, len(arr)) for s, e, arr in good_info],
    )

    # 7) Подготовка для замены
    filtered_bad = []
    good_arrays = []
    for s, e, arr in good_info:
        filtered_bad.append((s, e))
        good_arrays.append(arr)
    if not filtered_bad:
        print("Нет сегментов с достаточной историей или прогноз не получился.")
        return

    # 8) Замена сегментов
    replaced = replace_segments(signal, filtered_bad, good_arrays)

    # 9) Визуализация оригинал vs заменённый
    plot_signals(signal, replaced, fs, filtered_bad)

    # 10) HeartPy-анализ
    analyze_with_heartpy(signal, fs, replaced)


if __name__ == "__main__":
    main()
