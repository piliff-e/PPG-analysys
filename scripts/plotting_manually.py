import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
# import heartpy as hp


""" Получение данных """
# Чтение данных
record = wfdb.rdrecord('../data/s1_walk')
annotations = wfdb.rdann('../data/s1_walk', 'atr')

# Извлечение сигналов
ppg_ac = record.p_signal[:, 0]  # AC-компонента (пульсовая волна)
ppg_dc = record.p_signal[:, 1]  # DC-компонента (качество сигнала)
time = np.arange(len(ppg_ac)) / record.fs  # Временная ось в секундах

# Фильтрация AC-компоненты (улучшенные параметры)
# (Это было предложением DeepSeek'а, такая фильтрация явно неправильная (слишком сильная),
# но дальше на графике наглядно видно, как примерно у нас всё должно выглядеть, когда сделаем forecasting.)
b, a = butter(3, [0.5, 3], btype='bandpass', fs=record.fs)  # Верхний предел 3 Гц (для ЧСС до 180 уд/мин)
filtered_ac = filtfilt(b, a, ppg_ac)  # Фазово-независимая фильтрация

# Поиск артефактов DC (оптимизированный метод)
# (Это тоже предложение DeepSeek'а, и от этого всего зависит то,
# как будут отображаться артефакты DC, и будут ли вообще (сейчас не отображаются).)
diff_dc = ppg_dc[2:] - ppg_dc[:-2]
mad = np.median(np.abs(diff_dc - np.median(diff_dc)))
threshold = max(3 * mad, 30)  # Комбинированный порог
artifacts_dc = np.where(np.abs(diff_dc) > threshold)[0] + 1


""" Визуализация """
plt.figure(figsize=(16, 9))

# (Это рисует с помощью heartpy, обрезанная версия в соответствующем файле.)
# wd, m = hp.process(ppg_ac, sample_rate=50)  # sample_rate = 50 Гц
# hp.plotter(wd, m)
#
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# plt.savefig('s1_walk_heartpy.png')


# AC-компонента
plt.plot(
    time, ppg_ac,
    color='#FF9999',  # Бледно-красная ломаная (изначальная, фактическая пульсовая волна)
    linewidth=0.8,
    alpha=0.5,
    label='AC PPG (Raw)'
)
plt.plot(
    time, filtered_ac,
    color='#CC0000',  # Ярко-красная ломаная ("отфильтрованная" DeepSeek'ом)
    linewidth=1.2,
    label='AC PPG (Filtered)'
)

# Аннотации (N)
plt.scatter(
    annotations.sample / record.fs,
    ppg_ac[annotations.sample],
    color='#FF9900',  # Оранжевые крестики
    marker='x',
    s=50,
    label='Annotations (N)'
)

# DC-компонента
plt.plot(
    time, ppg_dc,
    color='#0066CC',  # Синяя ломаная (качество сигнала, видимо...)
    linewidth=0.8,
    label='DC PPG'
)
plt.scatter(
    time[artifacts_dc],
    ppg_dc[artifacts_dc],
    color='#6600CC',  # Фиолетовые крестики (сейчас их нет)
    marker='x',
    s=50,
    label='DC Artifacts'
)

# Настройки графика
plt.title('PPG Signal Analysis (s1_walk)', fontsize=14)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True, alpha=0.6)
plt.legend(loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.show()

# (Это какие-то проверки, предложенные DeepSeek'ом, когда я спросил, почему артефакты DC сранно рисуются.)
# print('Уникальные метки аннотаций:', np.unique(annotations.symbol))
# total_time = len(ppg_dc) / record.fs
# artifact_time = len(artifacts_dc) / record.fs
# print(f'Артефакты занимают {artifact_time / total_time * 100:.1f}% времени')
# print(f"MAD: {mad:.2f}, Порог: {threshold:.2f}")
