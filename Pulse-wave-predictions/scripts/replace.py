import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
from build_good_segments import build_good_segments_with_model
from neuralforecast import NeuralForecast
from shared import load_ppg
from temporary import generate_random_bad_segments


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
                np.arange(len(good)),
                good,
            )
        out[b0:b1] = good
    return out


def analyze_with_heartpy(signal, fs, replaced):
    """
    Базовый анализ BPM до и после замены "плохих" сегментов на "хорошие".
    """
    try:
        wd_orig, m_orig = hp.process(signal, sample_rate=fs)
        wd_rep, m_rep = hp.process(replaced, sample_rate=fs)
        print("Original BPM:", m_orig["bpm"], "→ Replaced BPM:", m_rep["bpm"])
    except Exception as e:
        print("Анализ при помощи HeartPy невозможен:", e)


def plot_signals(orig, replaced, fs, bad_segments):
    """
    Визуализация сигнала вместе с заменами "плохих" сегментов на "хорошие".
    """
    t = np.arange(len(orig)) / fs
    plt.plot(t, orig, label="Оригинал")
    plt.plot(t, replaced, alpha=0.7, label="После замены")
    for b0, b1 in bad_segments:
        plt.axvspan(b0 / fs, b1 / fs, color="red", alpha=0.2)
    plt.legend()
    plt.xlabel("Время (с)")
    plt.title("PPG: замена плохих сегментов")
    plt.tight_layout()
    plt.show()


"""
def main():
    # Загрузка данных
    signal, fs = load_ppg("../test_data/s13_sit")

    # Создание "плохих" и "хороших" сегментов
    # bad_segments, good_segments = manual_segments()
    bad_segments = generate_random_bad_segments(len(signal))
    good_segments = build_good_segments_with_model(signal, bad_segments, nf, "s13_sit", fs) 

    # Для replace_segments:
    filtered_bad = []
    good_arrays = []
    for (b0,b1) in bad_segments:
        # Найдём в good_info тот, что имеет тот же b0
        for (s,e,arr) in good_segments:
            if s == b0:
                filtered_bad.append((s,e))
                good_arrays.append(arr)
                break
    # Теперь:
    replaced_signal = replace_segments(signal, filtered_bad, good_arrays)
    # Замена "плохих" сегментов на "хорошие"
    # replaced = replace_segments(signal, bad_segments, good_segments)

    # Визуализация
    plot_signals(signal, replaced_signal, fs, bad_segments)

    # Анализ при помощи HeartPy
    analyze_with_heartpy(signal, fs, replaced_signal)
"""


def main():
    # 1) Параметры
    record_name = "s13_sit"  # например
    data_path = "../test_data"
    model_path = "../models"  # папка с сохранённой моделью NeuralForecast
    # Параметры истории и прогноза (должны совпадать с обученной моделью)
    N = 512
    K = 100
    # Генерация:
    num_rand_segs = 5
    min_len = 200
    max_len = 1000
    seed = 123

    # 2) Загрузка исходного сигнала
    signal, fs = load_ppg(f"{data_path}/{record_name}")
    print(f"Загружен сигнал {record_name}, длина {len(signal)}, fs={fs}")

    # 3) Загрузка модели
    nf = NeuralForecast.load(model_path)
    print("Модель загружена")

    # 4) Генерация случайных «плохих» сегментов
    bad_segments = generate_random_bad_segments(
        len(signal),
        num_segments=num_rand_segs,
        min_len=min_len,
        max_len=max_len,
        seed=seed,
    )
    print("Случайные сегменты:", bad_segments)

    # 5) Построение «хороших» сегментов через модель
    good_info = build_good_segments_with_model(
        signal, bad_segments, nf, record_name, fs, N=N, K=K
    )
    print(
        "Построены хорошие сегменты (start, end, len):",
        [(s, e, len(arr)) for (s, e, arr) in good_info],
    )

    # 6) Подготовка списков для replace_segments
    filtered_bad = []
    good_arrays = []
    for s, e, arr in good_info:
        filtered_bad.append((s, e))
        good_arrays.append(arr)
    # Если ни для одного сегмента история не подошла, filtered_bad будет пустой
    if not filtered_bad:
        print(
            "Ни для одного сегмента не хватило истории или не удалось сделать прогноз."
        )
        return

    # 7) Заменяем в сигнале
    replaced = replace_segments(signal, filtered_bad, good_arrays)

    # 8) Визуализация
    plot_signals(signal, replaced, fs, filtered_bad)

    # Опционально: анализ HeartPy:
    # from replace import analyze_with_heartpy
    # analyze_with_heartpy(signal, fs, replaced)


if __name__ == "__main__":
    main()
