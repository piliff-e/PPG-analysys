import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from neuralforecast import NeuralForecast
from pyts.metrics import dtw
from scipy.signal import decimate, savgol_filter
from forecasting import recursive_forecast


def plot_mismatches(
    orig_signal: np.ndarray, replaced_signal: np.ndarray, fs: float, bad_segments: list
):
    """
    Рисует оригинальный сигнал и на участках bad_segments подкрашивает форму заменённого сигнала.
    orig_signal, replaced_signal — numpy array одной длины.
    bad_segments — список (start, end) индексов, где сигнал был заменён.
    fs — частота дискретизации (Гц).
    """
    t = np.arange(len(orig_signal)) / fs

    # Рисуем исходный сигнал нейтральным цветом (синий или серый)
    plt.plot(t, orig_signal, linewidth=1, label="Оригинал")

    first_replacement = (
        True  # Флаг, чтобы пометить только первую заменённую линию в легенде
    )

    for s, e in bad_segments:
        # ограничиваем в пределах длины массива
        s0 = max(0, s)
        e0 = min(len(orig_signal), e)
        if s0 >= e0:
            continue

        # Подсветка фоном (не влияет на легенду)
        plt.axvspan(s0 / fs, e0 / fs, color="red", alpha=0.2)

        # Рисуем на этом участке исправленную форму
        if first_replacement:
            # первая замена: даём явную метку для легенды
            plt.plot(
                t[s0:e0],
                replaced_signal[s0:e0],
                color="red",
                linewidth=1.5,
                label="Исправлено",
            )
            first_replacement = False
        else:
            # последующие: без метки, чтобы не плодить записи в легенде
            plt.plot(
                t[s0:e0],
                replaced_signal[s0:e0],
                color="red",
                linewidth=1.5,
                label="_nolegend_",
            )

    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда PPG")
    plt.title("PPG: оригинал (синий) и заменённые сегменты (красный)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


record = wfdb.rdrecord("../test_data/s13_sit")
signal = decimate(record.p_signal[:, 0], q=5)
signal = savgol_filter(signal, 15, 3)
signal = (signal - np.mean(signal)) / np.std(signal)
fs = record.fs

backup_signal = signal.copy()

full_df = pd.DataFrame(
    {
        "unique_id": "segment",
        "ds": pd.date_range(
            start="2025-01-01", periods=len(backup_signal), freq="20ms"
        ),
        "y": backup_signal,
    }
)


nf = NeuralForecast.load("../models/v3")

bad_segments = []  # список кортежей (start_idx, end_idx) в исходном сигнале

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
    if len(current_segment) < 500:
        forecast_segment = forecast_segment[: len(current_segment)]

    alignment = dtw(forecast_segment, current_segment)
    if alignment > 16:
        signal[i : i + 500] = forecast_segment
        bad_segments.append((i, min(i + 500, len(signal))))
        print(
            f"[!] Обнаружено отклонение на сегменте {i} - {i}+500, DTW Distance: {alignment}"
        )
    else:
        print(f"[+] Сегмент {i} - {i}+500 в норме, DTW Distance: {alignment}")

final_df = pd.DataFrame(
    {
        "unique_id": "full_signal",
        "ds": pd.date_range(start="2025-01-01", periods=len(signal), freq="20ms"),
        "y": signal,
    }
)

orig_signal = backup_signal  # или сохранённая копия до замен
replaced_signal = signal  # текущий массив после замен
# bad_segments собран в процессе

plot_mismatches(orig_signal, replaced_signal, fs, bad_segments)
