"""
import numpy as np
from dtw import dtw


def dtw_distance(a: np.ndarray, b: np.ndarray):
    dist, _, _, _ = dtw(a, b, dist=lambda x, y: abs(x - y))
    return dist


def sliding_window_dtw(signal, nf, name, fs, N=512, K=100, step=50):
    M = len(signal)
    avg_dtw = np.zeros(M)
    counts = np.zeros(M, dtype=int)
    for t in range(N, M - K + 1, step):
        history = signal[t - N : t]
        ds = ...  # как делали ранее
        forecast_df = nf.predict(df=...)
        pred = forecast_df["NBEATS"].values[:K]
        for i in range(len(pred)):
            orig_seg = signal[t + i : t + i + len(pred)]
            approx = pred  # можно взять прогноз окна
            if len(orig_seg) >= len(pred):
                d = dtw_distance(orig_seg[: len(pred)], approx)
                idx = t + i
                avg_dtw[idx] += d
                counts[idx] += 1
    mask = counts > 0
    avg_dtw[mask] = avg_dtw[mask] / counts[mask]
    return avg_dtw


def find_bad_by_dtw(avg_dtw, thr):
    bad = []
    in_seg = False
    for i, v in enumerate(avg_dtw):
        if v > thr and not in_seg:
            start = i
            in_seg = True
        elif v <= thr and in_seg:
            bad.append((start, i))
            in_seg = False
    if in_seg:
        bad.append((start, len(avg_dtw)))
    return bad
"""
# sliding_window_with_dtw.py

import numpy as np
import pandas as pd
from dtw import dtw  # или из tslearn.metrics импорт dtw, dtw_path


def dtw_distance(a: np.ndarray, b: np.ndarray):
    """
    Вычисляет DTW-расстояние между одномерными рядами a и b.
    """
    dist, _, _, _ = dtw(a, b, dist=lambda x, y: abs(x - y))
    return dist


def sliding_window_dtw(signal, nf, name, fs, N=512, K=100, step=50):
    """
    Скользящее окно: для каждой позиции t от N до len(signal)-K прогнозируем K точек,
    вычисляем DTW-расстояние между прогнозом и соответствующим участком оригинала,
    аккумулируем и усредняем эти расстояния для каждого индекса.
    Возвращает avg_dtw — массив длины len(signal), где ненулевые значения в областях,
    покрытых прогнозами.
    """
    M = len(signal)
    avg_dtw = np.zeros(M)
    counts = np.zeros(M, dtype=int)
    base_time = pd.Timestamp("2025-01-01")
    # интервал времени между отсчетами в миллисекундах
    if fs and fs > 0:
        freq_ms = int(round(1000.0 / fs))
    else:
        freq_ms = 20

    for t in range(N, M - K + 1, step):
        history = signal[t - N : t]
        # формируем DataFrame для прогноза на этот фрагмент
        ds = pd.date_range(start=base_time, periods=N, freq=f"{freq_ms}ms")
        df_win = pd.DataFrame({"unique_id": name, "ds": ds, "y": history})
        try:
            forecast_df = nf.predict(df=df_win)  # здесь используем df_win
        except Exception as e:
            print(f"[DTW] Ошибка прогноза на окне t={t}: {e}")
            continue
        if "NBEATS" not in forecast_df.columns:
            # если модель возвращает другое имя колонки, поправьте здесь
            continue
        pred = forecast_df["NBEATS"].values
        L = min(K, len(pred), M - t)
        if L <= 0:
            continue
        # Вычисляем DTW между прогнозом pred[:L] и соответствующим участком оригинала signal[t:t+L]
        orig_segment = signal[t : t + L]
        try:
            d = dtw_distance(pred[:L], orig_segment)
        except Exception as e:
            print(f"[DTW] Ошибка DTW на окне t={t}: {e}")
            continue
        # Накопим это расстояние для всех точек окна (t .. t+L-1)
        for j in range(L):
            idx = t + j
            avg_dtw[idx] += d
            counts[idx] += 1

    # Усреднение
    mask = counts > 0
    avg_dtw[mask] = avg_dtw[mask] / counts[mask]
    return avg_dtw


def find_bad_by_dtw(avg_dtw, threshold_factor=2.0):
    """
    Находит интервалы, где avg_dtw превышает threshold_factor * std ненулевых значений avg_dtw.
    """
    mask_nonzero = avg_dtw > 0
    if not np.any(mask_nonzero):
        return []
    std = np.std(avg_dtw[mask_nonzero])
    if std == 0:
        std = 1e-6
    bad_mask = avg_dtw > threshold_factor * std
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
        segments.append((start, len(avg_dtw)))
    return segments
