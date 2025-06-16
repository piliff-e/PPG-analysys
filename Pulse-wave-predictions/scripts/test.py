# Пример использования bad_segment_detector.py:
import numpy as np
from bad_segment_detector import (
    detect_bad_cycles,
    extract_cycle_features,
    train_envelope_model,
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

import matplotlib.pyplot as plt

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
