import wfdb
import matplotlib.pyplot as plt
import heartpy as hp


""" Извлечение сигналов """
record = wfdb.rdrecord('../data/s1_walk')  # Чтение данных
ppg_ac = record.p_signal[:, 0]  # AC-компонента (пульсовая волна)


""" Визуализация """
plt.figure(figsize=(16, 9))

wd, m = hp.process(ppg_ac, sample_rate=50)  # sample_rate влияет на "BPM" на графике -- измени, чтобы проверить
hp.plotter(wd, m)

plt.grid(True)
plt.tight_layout()
plt.show()
# plt.savefig('s1_walk_heartpy.png')  # Этой функцией можно сохранить полученную картинку
