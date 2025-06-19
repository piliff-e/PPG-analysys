import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from pyts.metrics import dtw


def recursive_forecast(
    nf: NeuralForecast, start_df: pd.DataFrame, repeats: int = 10
) -> pd.DataFrame:
    """
    Рекурсивный прогноз: на вход DataFrame с колонками ['unique_id','ds','y'] длины N,
    возвращает конкатенацию прогнозов: nf.predict + добавление прогноза в историю, повторить repeats раз.
    """
    forecasts = []
    current_df = start_df.copy()

    for i in range(repeats):
        forecast_df = nf.predict(df=current_df)

        # Отфильтровать NaN
        forecast_df_clean = forecast_df.dropna(subset=["NBEATS"]).copy()
        if forecast_df_clean.empty:
            print(f"[forecast_utils] Прогноз пуст на итерации {i}")
            break
        forecasts.append(forecast_df_clean)

        # Добавляем прогноз в текущую историю
        new_block = forecast_df_clean.copy()
        new_block["y"] = new_block["NBEATS"]
        current_df = pd.concat(
            [current_df, new_block[["unique_id", "ds", "y"]]]
        ).reset_index(drop=True)
    if forecasts:
        full_forecast = pd.concat(forecasts).reset_index(drop=True)
        return full_forecast
    else:
        return pd.DataFrame()  # пустой


def create_history_df(
    history: np.ndarray,
    record_name: str,
    freq: str = "20ms",
    start_time: str = "2025-01-01",
) -> pd.DataFrame:
    """
    Строит DataFrame для прогноза:
    - history: numpy-массив длины N
    - record_name: unique_id, используемый в прогнозе
    - freq: строка частоты, например "20ms", должна совпадать с тем, что использовалось при обучении
    - start_time: "2025-01-01" или любая другая базовая метка
    Возвращает DataFrame с колонками ['unique_id','ds','y'] длины len(history).
    """
    N = len(history)
    ds = pd.date_range(start=start_time, periods=N, freq=freq)
    df = pd.DataFrame({"unique_id": record_name, "ds": ds, "y": history})
    return df


def detect_and_replace_segments(
    signal: np.ndarray,
    nf: NeuralForecast,
    record_name: str,
    window_size: int,
    repeats: int,
    threshold: float,
    freq: str = "20ms",
) -> tuple[np.ndarray, list]:
    """
    Идёт по сигналу с шагом = window_size:
      для каждой позиции start = window_size, 2*window_size, ..., строит прогноз на предыдущих window_size точках,
      вычисляет DTW между прогнозом и реальным сегментом такой же длины,
      если DTW > threshold — заменяет сегмент прогнозом и сохраняет (start, start+L) в bad_segments.
    Параметры:
      - signal: исходный (уже нормализованный) numpy-массив
      - nf: загруженный NeuralForecast
      - record_name: unique_id, совпадающий с тем, на чём обучалась модель
      - window_size: N (и одновременно шаг)
      - repeats: число итераций recursive_forecast при прогнозировании сегмента
      - threshold: DTW-порог для замены
      - freq: строка частоты для DataFrame, например "20ms"
    Возвращает:
      - replaced_signal: numpy-массив с подставленными прогнозами
      - bad_segments: список (start, end) заменённых участков
    """
    replaced_signal = signal.copy()
    bad_segments: list[tuple[int, int]] = []
    length = len(signal)

    for start in range(window_size, length, window_size):
        end = min(start + window_size, length)
        hist_start = start - window_size
        history = replaced_signal[hist_start:start]

        # Построение DataFrame для прогноза
        df_hist = create_history_df(history, record_name, freq=freq)

        # Прогноз: рекурсивно repeats раз
        forecast_df = recursive_forecast(nf, df_hist, repeats=repeats)
        if forecast_df.empty or "NBEATS" not in forecast_df.columns:
            print(
                f"[!] Прогноз пуст или нет 'NBEATS' для сегмента {start - window_size}-{start}"
            )
            continue
        pred = forecast_df["NBEATS"].values

        # Сравнение с текущим сегментом
        curr = replaced_signal[start:end]
        L = min(len(pred), len(curr))
        if L <= 0:
            continue
        pred_seg = pred[:L]
        curr_seg = curr[:L]

        # Вычисление DTW
        try:
            alignment = dtw(pred_seg, curr_seg)
        except Exception as e:
            print(f"[!] Ошибка DTW на сегменте {start}-{start + L}: {e}")
            continue
        if alignment > threshold:
            # Замена
            replaced_signal[start : start + L] = pred_seg
            bad_segments.append((start, start + L))
            print(
                f"[!] Обнаружено отклонение на сегменте {start}-{start + L}, DTW={alignment:.2f}"
            )
        else:
            print(f"[+] Сегмент {start}-{start + L} в норме, DTW={alignment:.2f}")

    return replaced_signal, bad_segments
