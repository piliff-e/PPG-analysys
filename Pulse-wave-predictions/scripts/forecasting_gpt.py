import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from scipy.signal import decimate, savgol_filter


def load_and_preprocess(record_name, decimate_q=5, savgol_window=15, savgol_poly=3):
    """
    Загружает PPG-сигнал из WFDB и выполняет предобработку:
    - децимация
    - сглаживание Savitzky-Golay
    - нормализация (центровка и масштаб)
    Возвращает numpy array.
    """
    record = wfdb.rdrecord(record_name)
    raw = record.p_signal[:, 0]
    # Децимация
    sig = decimate(raw, q=decimate_q)
    # Сглаживание
    sig = savgol_filter(sig, savgol_window, savgol_poly)
    # Нормализация
    sig = (sig - np.mean(sig)) / np.std(sig)
    return sig


def build_dataframe(signals_dict, start_date="2025-01-01", freq="20ms"):
    """
    Создаёт единый DataFrame для NeuralForecast из нескольких сигналов.
    signals_dict: dict уникальный_id -> numpy array сигнала.
    Возвращает DataFrame с колонками ['unique_id', 'ds', 'y'].
    """
    df_list = []
    for uid, sig in signals_dict.items():
        df = pd.DataFrame(
            {
                "unique_id": uid,
                "ds": pd.date_range(start=start_date, periods=len(sig), freq=freq),
                "y": sig,
            }
        )
        df_list.append(df)
    if not df_list:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])
    combined = pd.concat(df_list).reset_index(drop=True)
    return combined


def create_model(input_size=512, horizon=100, max_steps=300, lr=1e-3, freq="20ms"):
    """
    Создаёт и возвращает объект NeuralForecast с моделью NBEATS.
    input_size соответствует исторической длине, horizon - прогнозная.
    """
    nf = NeuralForecast(
        models=[
            NBEATS(
                input_size=input_size, h=horizon, max_steps=max_steps, learning_rate=lr
            )
        ],
        freq=freq,
    )
    return nf


def recursive_forecast(nf, start_df, repeats=10):
    """
    Многократный рекурсивный прогноз:
    на каждой итерации дообучаем nf на current_df и получаем прогноз,
    добавляем предсказанное в current_df для следующей итерации.
    Возвращает DataFrame со всеми прогнозами.
    """
    forecasts = []
    current_df = start_df.copy()
    for i in range(repeats):
        print(f"[Forecast] Iteration {i + 1}/{repeats}, data length: {len(current_df)}")
        nf.fit(current_df)
        forecast_df = nf.predict()
        # Удаляем NaN-прогнозы
        forecast_df_clean = forecast_df.dropna(subset=["NBEATS"]).copy()
        if forecast_df_clean.empty:
            print(f"[!] Прогноз пуст на итерации {i}")
            break
        forecasts.append(forecast_df_clean)
        # Добавляем предсказанное как новые наблюдения
        new_block = forecast_df_clean.copy()
        new_block["y"] = new_block["NBEATS"]
        current_df = pd.concat(
            [current_df, new_block[["unique_id", "ds", "y"]]]
        ).reset_index(drop=True)
    if forecasts:
        full = pd.concat(forecasts).reset_index(drop=True)
    else:
        full = pd.DataFrame(columns=["unique_id", "ds", "NBEATS"])
    return full


def forecast_signal(filenames, data_path="../test_data", repeats=10):
    """
    Основная функция: принимает список имён WFDB-записей, строит общий DataFrame,
    создаёт модель, выполняет прогноз для первой записи в списке (или для всех по выбору).
    Возвращает словарь: имя -> DataFrame прогнозов.
    Для совместимости с изначальной логикой, прогнозируем только для s1_walk,
    но можно расширить.
    """
    # Загружаем и предобрабатываем сигналы
    signals = {}
    for name in filenames:
        path = f"{data_path}/{name}"
        try:
            sig = load_and_preprocess(path)
            signals[name] = sig
        except Exception as e:
            print(f"Ошибка загрузки предобработки для {name}: {e}")
    if not signals:
        raise RuntimeError("Нет загруженных сигналов для прогноза")
    # Собираем DataFrame для обучения
    df_all = build_dataframe(signals)
    # Создаём модель
    nf = create_model()
    # Прогноз для каждой записи по отдельности, используя рекурсивно только её данные
    forecast_results = {}
    for name in filenames:
        start_df = df_all[df_all["unique_id"] == name].copy()
        if start_df.empty:
            print(f"Нет данных для {name}, пропускаем")
            continue
        print(f"Запуск recursive_forecast для записи {name}")
        forecast_df = recursive_forecast(nf, start_df, repeats=repeats)
        forecast_results[name] = forecast_df
    return forecast_results


def plot_forecast(original_signal, forecast_df, fs=50, title="Прогноз PPG"):
    """
    Визуализация оригинального сигнала и прогноза.
    original_signal: numpy array
    forecast_df: DataFrame с колонками 'ds' и 'NBEATS'
    fs: частота семплирования оригинала (после децимации)
    """
    plt.figure(figsize=(20, 10))
    # Временные оси: индекс или ds?
    # Для простоты: по индексам
    x_orig = np.arange(len(original_signal))
    plt.plot(x_orig, original_signal, label="Оригинал", color="blue")
    # Прогноз: от len до len+... или по ds? Отображаем в отнесении к оригиналу
    if not forecast_df.empty:
        # Предполагаем, что forecast_df['ds'] продолжает timeline после оригинала
        # Для упрощения: ставим точки после конца оригинала
        x_start = len(original_signal)
        y_pred = forecast_df["NBEATS"].values
        x_pred = np.arange(x_start, x_start + len(y_pred))
        plt.plot(x_pred, y_pred, label="Прогноз", color="red", linestyle="--")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Если нужно взять load_and_preprocess в forecasting, импортируем его:
# from forecasting import load_and_preprocess  # чтобы load_and_preprocess был виден

if __name__ == "__main__":
    # Пример использования
    filenames = ["s1_walk", "s2_walk", "s3_walk"]
    forecasts = forecast_signal(filenames, data_path="../test_data", repeats=10)
    # Визуализируем для первой записи
    first = filenames[0]
    if first in forecasts:
        # Загружаем оригинал снова для визуализации
        orig, fs = load_and_preprocess(f"../test_data/{first}"), None
        # fs неизвестна после децимации, но x-axis по индексам
        plot_forecast(orig, forecasts[first], title=f"Прогноз для {first}")
