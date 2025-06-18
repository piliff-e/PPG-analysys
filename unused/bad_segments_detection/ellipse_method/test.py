# Пример использования bad_segment_detector.py:
import matplotlib.pyplot as plt
import numpy as np
from bad_segment_detector import (
    detect_bad_cycles,
    extract_cycle_features,
    peak_detection,
    train_envelope_model,
    troughs_detection,
)
from shared import load_ppg

# signal, fs = load_ppg("../test_data/s6_walk")  # или предварительно предобработать
# Собрать обучающие признаки из нескольких чистых сигналов:
feature_list = []
interval_list = []
for name in ["s1_walk", "s2_walk", "s3_walk"]:
    sig, fs = load_ppg(f"../test_data/{name}")
    feats, intervals = extract_cycle_features(sig, fs)
    if feats.size > 0:
        feature_list.append(feats)
    # if intervals.size > 0:
    #     interval_list.append(intervals)
if feature_list:
    X_train = np.vstack(feature_list)
    model, mean, std = train_envelope_model(X_train, contamination=0.1)
    # Для тестовой записи:
    sig_test, fs_test = load_ppg("../test_data/s6_walk")
    bad_intervals = detect_bad_cycles(sig_test, fs_test, model, mean, std)
    print("Найдены плохие интервалы:", bad_intervals)
    # Далее замена циклов с помощью прогноза


good = []
for b0, b1 in bad_intervals:
    x = np.linspace(b0, b1)
    good.append((b1 - b0) * np.sin(x))
"""
replaced_signal = replace_segments(sig_test, bad, good)
plot_signals(sig_test, replaced_signal, fs_test, bad)
"""


"""
# 1) Извлечём все циклы: признаки + интервалы
features, intervals = extract_cycle_features(sig_test, fs_test)
# 2) Детекция «плохих» циклов
bad_intervals = detect_bad_cycles(sig_test, fs_test, model, mean, std)
"""

# 3) График всего сигнала с границами циклов
t = np.arange(len(sig_test)) / fs_test
plt.figure(figsize=(12, 4))
plt.plot(t, sig_test, label="PPG сигнал")
# Для каждого цикла нарисовать вертикальную линию начала
for s, e in intervals:
    plt.axvline(s / fs_test, color="gray", alpha=0.2, label="Циклы")
# Подсветить плохие циклы
for s, e in bad_intervals:
    plt.axvspan(s / fs_test, e / fs_test, color="red", alpha=0.3, label="Плохие циклы")
plt.xlabel("Время, с")
plt.title("Сигнал PPG с границами циклов и подсветкой плохих")
plt.legend()
plt.tight_layout()
plt.show()


# Выбрать несколько временных окон, например [0:10с], [190:210с], [270:290с]
for t0 in [0, 190, 270]:
    start_idx = int(t0 * fs_test)
    end_idx = int((t0 + 10) * fs_test)
    seg = sig_test[start_idx:end_idx]
    peaks = peak_detection(seg, fs_test)
    troughs = troughs_detection(seg, peaks)
    t = (np.arange(len(seg)) + start_idx) / fs_test
    plt.figure(figsize=(8, 3))
    plt.plot(t, seg, label="PPG")
    plt.scatter((peaks + start_idx) / fs_test, seg[peaks], color="r", label="peaks")
    plt.scatter(
        (troughs + start_idx) / fs_test, seg[troughs], color="g", label="troughs"
    )
    plt.title(f"Разбиение на циклы в окне {t0}-{t0 + 10} с")
    plt.legend()
    plt.show()


"""
# row — строка df_test для цикла около 200 с
s0, e0 = int(row['start']), int(row['end'])
cycle = sig_test[s0:e0]
t_cycle = np.arange(len(cycle)) / fs_test + row['time_mid_s'] - (e0-s0)/(2*fs_test)
plt.figure()
plt.plot(t_cycle, cycle)
plt.title("Цикл около 200 с")
plt.show()
"""

# Вычислите скользящее среднее и скользящее std по сигналу, чтобы увидеть, как меняется уровень сигнала и шум.
window = int(fs_test * 5)  # 5-секундное окно
mov_std = np.array(
    [np.std(sig_test[i : i + window]) for i in range(0, len(sig_test) - window, window)]
)
plt.plot(mov_std)
plt.title("Скользящий STD (5с) сигнала")
plt.show()
