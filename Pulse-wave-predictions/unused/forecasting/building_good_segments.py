import numpy as np
import pandas as pd

# или если у вас forecasting.py экспортирует NeuralForecast: from forecasting import NeuralForecast
# Но обычно: from neuralforecast import NeuralForecast


def build_good_segments_with_model(
    signal, bad_segments, nf, record_name, fs, N=512, K=100
):
    """
    Для каждого сегмента (b0,b1) из bad_segments пытаемся предсказать "хорошие" значения
    опираясь на N предыдущих точек сигнала.
    Возвращает список good_segments того же порядка, что bad_segments, но сегменты,
    для которых нет истории или прогноз не получился, пропускаются (не добавляются).
    Параметры:
      - signal: numpy array исходного сигнала (float).
      - bad_segments: список (b0,b1).
      - nf: объект NeuralForecast, загруженный через NeuralForecast.load или аналог.
      - record_name: строка unique_id для DataFrame, можно передать любое, но желательно то же,
        что использовалось при обучении (если модель чувствительна к unique_id).
      - fs: частота дискретизации (Гц).
      - N: длина истории (input_size).
      - K: длина прогноза (horizon).
    """
    good_segments = []
    # Базовое время для столбца 'ds'. Оно не влияет на форму прогноза, лишь на метки времени.
    base_time = pd.Timestamp("2025-01-01")
    # Частота (интервал) в милисекундах
    freq_ms = None
    if fs and fs > 0:
        freq_ms = int(round(1000.0 / fs))
    else:
        freq_ms = 20  # дефолт
    for b0, b1 in bad_segments:
        # Проверка истории
        if b0 < N:
            print(
                f"[build_good] Пропускаем сегмент {b0}-{b1}: недостаточно истории (нужно {N}, есть {b0})"
            )
            continue
        # Берём историю из уже, возможно, частично изменённого сигнала? Здесь signal неизменён.
        history = signal[b0 - N : b0].copy()
        # Формируем DataFrame
        # 'ds' от base_time (может быть одинаков для всех окон, главное, регулярный freq)
        ds = pd.date_range(start=base_time, periods=N, freq=f"{freq_ms}ms")
        df_win = pd.DataFrame({"unique_id": record_name, "ds": ds, "y": history})
        # Делаем прогноз
        try:
            forecast_df = nf.predict(df=df_win)
        except Exception as e:
            print(f"[build_good] Ошибка прогноза для сегмента {b0}-{b1}: {e}")
            continue
        if "NBEATS" not in forecast_df.columns:
            print(
                f"[build_good] В прогнозе нет колонки 'NBEATS' для сегмента {b0}-{b1}"
            )
            continue
        pred_vals = forecast_df["NBEATS"].values
        # Берём первые L точек, где L = min(K, b1-b0, len(pred_vals))
        L = min(K, b1 - b0, len(pred_vals))
        if L <= 0:
            continue
        good_seg = pred_vals[:L]
        good_segments.append((b0, b0 + L, good_seg))
        # Если сегмент длиннее K, и вы хотите рекурсивно достраивать оставшуюся часть,
        # можно здесь использовать цикл:
        # pos = b0
        # while pos < b1:
        #     if pos < N: break
        #     history = replaced[pos-N:pos]  # если вы вставляете замену в сигнальный буфер
        #     ... прогноз ...
        #     pos += L
        #     L = min(K, b1-pos)
        #     ...
        # Но для упрощения сейчас делаем только однократный прогноз.
    return good_segments


def build_good_segments_recursively(
    signal, bad_segments, nf, record_name, fs, N=512, K=100
):
    """
    Рекурсивно достраивает для каждого плохого сегмента. Возвращает список (b0, b1_final, array_total).
    array_total длины (b1_final - b0): собранный прогноз за несколько итераций.
    """
    good_info = []
    base_time = pd.Timestamp("2025-01-01")
    freq_ms = int(round(1000.0 / fs)) if fs and fs > 0 else 20

    # Копия сигнала, в которую можем вставлять прогнозы, чтобы при рекурсивном прогнозе брать уже заменённые значения
    replaced_signal = signal.copy().astype(float)

    for b0, b1 in bad_segments:
        if b0 < N:
            print(f"[build_rec] Пропускаем сегмент {b0}-{b1}: недостаточно истории")
            continue
        parts = []
        pos = b0
        # Для каждого сегмента заводим локальную копию replaced_signal, чтобы использовать апдейты при рекурсии
        while pos < b1:
            if pos < N:
                break
            history = replaced_signal[pos - N : pos]
            ds = pd.date_range(start=base_time, periods=N, freq=f"{freq_ms}ms")
            df_win = pd.DataFrame({"unique_id": record_name, "ds": ds, "y": history})
            try:
                forecast_df = nf.predict(df=df_win)
            except Exception as e:
                print(f"[build_rec] Ошибка прогноза при pos={pos}: {e}")
                break
            if "NBEATS" not in forecast_df.columns:
                print(f"[build_rec] Нет колонки NBEATS при pos={pos}")
                break
            pred_vals = forecast_df["NBEATS"].values
            L = min(K, b1 - pos, len(pred_vals))
            if L <= 0:
                break
            # вставляем в replaced_signal, чтобы следующая итерация могла брать историю из прогноза
            replaced_signal[pos : pos + L] = pred_vals[:L]
            parts.append(pred_vals[:L])
            pos += L
        if parts:
            # объединяем части в один массив
            arr_total = np.concatenate(parts)
            good_info.append((b0, b0 + len(arr_total), arr_total))
    return good_info
