"""
# Предполагая, что df_test уже есть как выше:
# 1) Найдём цикл, наиболее близкий к 200 с:
target_idx = (df_test['time_mid_s'] - 200).abs().argmin()
row = df_test.iloc[target_idx]
print("Цикл вокруг 200 с:", row)
# Выведем start, end, признаки, label, score.

# 2) Визуализируем сам этот участок сигнала:
s0, e0 = int(row['start']), int(row['end'])
import matplotlib.pyplot as plt
t = np.arange(len(sig_test)) / fs_test
plt.figure(figsize=(8,3))
plt.plot(t[s0- int(1*fs_test): e0 + int(1*fs_test)], sig_test[s0- int(1*fs_test): e0 + int(1*fs_test)], label='Около цикла')
plt.axvspan(s0/fs_test, e0/fs_test, color='red', alpha=0.3, label='Граница цикла')
plt.xlabel('Время, с')
plt.legend()
plt.title('Участок вокруг явного выброса на 200 с')
plt.tight_layout()
plt.show()
"""
