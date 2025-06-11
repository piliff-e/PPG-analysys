import wfdb
import numpy as np
import heartpy as hp
import matplotlib.pyplot as plt


def load_ppg(record_name):
    """
    Загрузка PPG-сигнала из WFDB-файлов (.hea/.dat/.atr).
    """
    record = wfdb.rdrecord(record_name)
    signal = record.p_signal[:, 0]  # одномерный список float'ов
    fs = record.fs  # частота семплирования
    return signal, fs


def replace_segments(signal, bad_segments, good_segments):
    """
    Замена "плохих" сегментов bad_segments на "хорошие" good_segments.
     - bad_segments: список пар (b0, b1), где b0 - начало "плохого", а b1 - конец
     - good_segments: в данном случае список синусоид на отрезках [b0, b1]
    """
    out = signal.copy()
    for (b0, b1), good in zip(bad_segments, good_segments):
        if len(good) != (b1 - b0):
            good = np.interp(
                np.linspace(0, len(good), b1 - b0, endpoint=False),
                np.arange(len(good)), good)
        out[b0:b1] = good
    return out


def manual_segments():
    """
    Создание вручную "плохих" и "хороших" сегментов для теста.
    """
    bad = [(500, 700), (900, 1500), (2000, 2150), (3000, 3500), (5000, 5400)]
    good = []
    for b0, b1 in bad:
        x = np.linspace(b0, b1)
        good.append((b1 - b0) * np.sin(x))
    return bad, good


def analyze_with_heartpy(signal, fs, replaced):
    """
    Базовый анализ BPM до и после замены "плохих" сегментов на "хорошие".
    """
    try:
        wd_orig, m_orig = hp.process(signal, sample_rate=fs)
        wd_rep, m_rep = hp.process(replaced, sample_rate=fs)
        print('Original BPM:', m_orig['bpm'], '→ Replaced BPM:', m_rep['bpm'])
    except Exception as e:
        print('Анализ при помощи HeartPy невозможен:', e)


def plot_signals(orig, replaced, fs, bad_segments):
    """
    Визуализация сигнала вместе с заменами "плохих" сегментов на "хорошие".
    """
    t = np.arange(len(orig)) / fs
    plt.plot(t, orig, label='Оригинал')
    plt.plot(t, replaced, alpha=0.7, label='После замены')
    for b0, b1 in bad_segments:
        plt.axvspan(b0 / fs, b1 / fs, color='red', alpha=0.2)
    plt.legend()
    plt.xlabel('Время (с)')
    plt.title('PPG: замена плохих сегментов')
    plt.tight_layout()
    plt.show()


def main():
    # Загрузка данных
    signal, fs = load_ppg('../test_data/s1_walk')

    # Создание "плохих" и "хороших" сегментов
    bad_segments, good_segments = manual_segments()

    # Замена "плохих" сегментов на "хорошие"
    replaced = replace_segments(signal, bad_segments, good_segments)

    # Визуализация
    plot_signals(signal, replaced, fs, bad_segments)

    # Анализ при помощи HeartPy
    analyze_with_heartpy(signal, fs, replaced)


if __name__ == '__main__':
    main()
