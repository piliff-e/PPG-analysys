# Модуль для детекции плохих сегментов скользящим окном и замены их предсказаниями из закэшированной модели

import numpy as np
import pandas as pd
from forecasting import (
    NeuralForecast,
)  # recursive_forecast может не использоваться, но NeuralForecast.load
from replace import plot_signals
from shared import load_ppg


def load_forecast_model(model_path):
    """
    Загружает закэшированную модель NeuralForecast из указанной папки.
    Возвращает объект nf или выбрасывает исключение при ошибке.
    """
    # Предполагаем, что модель сохранялась через nf.save(path)
    nf = NeuralForecast.load(model_path)
    print(f"[Model] Загружена модель из {model_path}")
    return nf


def sliding_window_error(signal, nf, name, fs, N=512, K=100, step=50):
    """
    Вычисляет усреднённую ошибку прогноза скользящим окном для всего сигнала.
    Возвращает avg_errors (массив той же длины, где 0 для первых N и последних <K).
    """
    M = len(signal)
    errors = np.zeros(M)
    counts = np.zeros(M, dtype=int)
    base_time = pd.Timestamp("2025-01-01")
    # Вычисляем интервал в ms по fs
    if fs and fs > 0:
        freq_ms = int(round(1000.0 / fs))
    else:
        freq_ms = 20
    for t in range(N, M - K + 1, step):
        history = signal[t - N : t]
        # Формируем ds: от base_time + (t-N)*freq_ms до base_time+(t-1)*freq_ms
        start_time = base_time + pd.to_timedelta((t - N) * freq_ms, unit="ms")
        ds = pd.date_range(start=start_time, periods=N, freq=f"{freq_ms}ms")
        df_win = pd.DataFrame({"unique_id": name, "ds": ds, "y": history})
        try:
            forecast_df = nf.predict(df=df_win)
        except Exception as e:
            print(f"[Warn] Ошибка прогноза на окне t={t}: {e}")
            continue
        if "NBEATS" not in forecast_df.columns:
            continue
        pred = forecast_df["NBEATS"].values
        L = min(K, len(pred))
        for i in range(L):
            idx = t + i
            if idx >= M:
                break
            err = abs(pred[i] - signal[idx])
            errors[idx] += err
            counts[idx] += 1
    avg_errors = np.zeros(M)
    mask = counts > 0
    avg_errors[mask] = errors[mask] / counts[mask]
    return avg_errors


def find_bad_segments_from_errors(avg_errors, threshold_factor=2.0):
    """
    Находит интервалы (start, end) по avg_errors: где avg_errors > threshold_factor * std(avg_errors>0).
    """
    mask_nonzero = avg_errors > 0
    if not np.any(mask_nonzero):
        return []
    std = np.std(avg_errors[mask_nonzero])
    if std == 0:
        std = 1e-6
    bad_mask = avg_errors > threshold_factor * std
    segments = []
    in_seg = False
    for i, flag in enumerate(bad_mask):
        if flag and not in_seg:
            start = i
            in_seg = True
        elif not flag and in_seg:
            end = i
            segments.append((start, end))
            in_seg = False
    if in_seg:
        segments.append((start, len(avg_errors)))
    return segments


def replace_bad_with_forecast(signal, bad_segments, nf, name, fs, N=512, K=100):
    """
    Заменяет плохие сегменты в копии сигнала предсказаниями модели nf.
    Для каждого сегмента (b0,b1) идём: pos = b0; пока pos < b1:
      - если pos < N: пропускаем сегмент (нельзя прогнозировать)
      - history = signal[pos-N:pos] (с учётом уже заменённых предыдущих участков)
      - predict horizon K; вставляем min(K, b1-pos) значений в новую копию
      - pos += K
    Возвращает новый массив length M.
    """
    M = len(signal)
    replaced = signal.copy().astype(float)
    base_time = pd.Timestamp("2025-01-01")
    if fs and fs > 0:
        freq_ms = int(round(1000.0 / fs))
    else:
        freq_ms = 20
    for b0, b1 in bad_segments:
        pos = b0
        while pos < b1:
            if pos < N:
                print(
                    f"[Replace] Нехватка истории для сегмента {b0}-{b1}, позиция {pos}, пропуск."
                )
                break
            history = replaced[pos - N : pos]
            start_time = base_time  # или base_time + offset, не критично
            ds = pd.date_range(start=start_time, periods=N, freq=f"{freq_ms}ms")
            df_win = pd.DataFrame({"unique_id": name, "ds": ds, "y": history})
            try:
                forecast_df = nf.predict(df=df_win)
            except Exception as e:
                print(f"[Replace] Ошибка прогноза при pos={pos}: {e}")
                break
            if "NBEATS" not in forecast_df.columns:
                print(f"[Replace] Нет колонки NBEATS в прогнозе при pos={pos}")
                break
            pred = forecast_df["NBEATS"].values
            L = min(K, b1 - pos, len(pred))
            if L <= 0:
                break
            replaced[pos : pos + L] = pred[:L]
            pos += L
    return replaced


# Пример использования в main:
if __name__ == "__main__":
    # Параметры
    record_name = "s11_sit"  # тестовая запись
    data_path = "../test_data"
    model_path = "../models"  # путь, где сохранена модель
    N = 512
    K = 100
    step = 50
    threshold_factor = 2.0

    # 1) Загрузка сигнала
    sig, fs = load_ppg(f"{data_path}/{record_name}")

    # 2) Загрузка модели
    nf = load_forecast_model(model_path)

    # 3) Детекция ошибок скользящим окном
    avg_errors = sliding_window_error(sig, nf, record_name, fs, N=N, K=K, step=step)
    bad_segments = find_bad_segments_from_errors(
        avg_errors, threshold_factor=threshold_factor
    )
    print("Найденные плохие сегменты:", bad_segments)

    # 4) Визуализация avg_errors (опционально)
    import matplotlib.pyplot as plt

    t = np.arange(len(sig)) / fs
    plt.figure(figsize=(10, 3))
    plt.plot(t, avg_errors, label="avg_errors")
    plt.axhline(
        threshold_factor * np.std(avg_errors[avg_errors > 0]),
        color="red",
        linestyle="--",
        label="порог",
    )
    plt.xlabel("Время, с")
    plt.title("Средняя ошибка прогноза скользящим окном")
    plt.legend()
    plt.show()

    # 5) Замена плохих сегментов предсказаниями
    replaced = replace_bad_with_forecast(
        sig, bad_segments, nf, record_name, fs, N=N, K=K
    )

    # 6) Визуализация замены
    plot_signals(sig, replaced, fs, bad_segments)

    # 7) HeartPy-анализ (если нужно)
    # from replace import analyze_with_heartpy
    # analyze_with_heartpy(sig, fs, replaced)
