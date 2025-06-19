import matplotlib.pyplot as plt
import numpy as np


def plot_mismatches(
    orig_signal: np.ndarray, replaced_signal: np.ndarray, fs: float, bad_segments: list
):
    """
    Рисует оригинальный сигнал (синий) и поверх заменённые участки (красный).
    """
    t = np.arange(len(orig_signal)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, orig_signal, color="blue", linewidth=1, label="Оригинал")

    first_replacement = True
    for s, e in bad_segments:
        s0 = max(0, s)
        e0 = min(len(orig_signal), e)
        if s0 >= e0:
            continue
        # Фоновая подсветка
        plt.axvspan(
            s0 / fs,
            e0 / fs,
            color="red",
            alpha=0.2,
            label="Плохой сегмент" if first_replacement else "_nolegend_",
        )
        # Линия прогноза
        plt.plot(
            t[s0:e0],
            replaced_signal[s0:e0],
            color="red",
            linewidth=1.5,
            label="Исправлено" if first_replacement else "_nolegend_",
        )
        first_replacement = False

    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда PPG")
    plt.title("PPG: оригинал и заменённые сегменты")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
