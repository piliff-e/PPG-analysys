import numpy as np
import pandas as pd
from bad_segment_detector import extract_cycle_features, train_envelope_model
from shared import load_ppg

# 1) Собираем DataFrame признаков обучающей выборки
rows = []
for name in ["s1_walk", "s2_walk", "s3_walk"]:
    sig, fs = load_ppg(f"../test_data/{name}")
    feats, intervals = extract_cycle_features(sig, fs)
    for (s, e), feat in zip(intervals, feats):
        rows.append(
            {
                "record": name,
                "start": s,
                "end": e,
                "skew": feat[0],
                "kurt": feat[1],
                "ap_entropy": feat[2],
                "shan_entropy": feat[3],
                "spec_entropy": feat[4],
            }
        )
df_train = pd.DataFrame(rows)
print("Train cycles:", len(df_train))

# 2) Обучение EllipticEnvelope
X_train = df_train[
    ["skew", "kurt", "ap_entropy", "shan_entropy", "spec_entropy"]
].values
model, mean, std = train_envelope_model(X_train, contamination=0.1)

# 3) Сбор признаков для тестовой записи
sig_test, fs_test = load_ppg("../test_data/s6_walk")
feats_test, intervals_test = extract_cycle_features(sig_test, fs_test)
rows_test = []
for (s, e), feat in zip(intervals_test, feats_test):
    rows_test.append(
        {
            "record": "s6_walk",
            "start": s,
            "end": e,
            "skew": feat[0],
            "kurt": feat[1],
            "ap_entropy": feat[2],
            "shan_entropy": feat[3],
            "spec_entropy": feat[4],
        }
    )
df_test = pd.DataFrame(rows_test)
print("Test cycles:", len(df_test))

# 4) Распределения признаков
import matplotlib.pyplot as plt

for col in ["skew", "kurt", "ap_entropy", "shan_entropy", "spec_entropy"]:
    plt.figure(figsize=(6, 3))
    plt.hist(df_train[col].dropna(), bins=30, alpha=0.5, label="train")
    plt.hist(df_test[col].dropna(), bins=30, alpha=0.5, label="test")
    plt.title(f"Распределение {col}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 5) Решение модели для тестовых циклов
Xn_test = (
    df_test[["skew", "kurt", "ap_entropy", "shan_entropy", "spec_entropy"]].values
    - mean
) / std
# Предикт (-1 / 1) и decision_function или Mahalanobis distance, если доступно
labels = model.predict(Xn_test)  # -1 или 1
# Если есть метод decision_function:
try:
    df_dec = model.decision_function(Xn_test)  # чем ниже, тем более аномально
except Exception:
    df_dec = None

df_test["label"] = labels
if df_dec is not None:
    df_test["score"] = df_dec

print(df_test.head())
print("Число помеченных аномальных циклов:", np.sum(labels == -1))

# Дополнительно: если знаете время (в секундах), можно добавить колонку time_mid = (start+end)/(2*fs)
df_test["time_mid_s"] = (df_test["start"] + df_test["end"]) / (2 * fs_test)
print(df_test[["time_mid_s", "label", "score"]].head(10))


# Предполагая, что df_test уже есть как выше:
# 1) Найдём цикл, наиболее близкий к 200 с:
target_idx = (df_test["time_mid_s"] - 200).abs().argmin()
row = df_test.iloc[target_idx]
print("Цикл вокруг 200 с:", row)
# Выведем start, end, признаки, label, score.

# 2) Визуализируем сам этот участок сигнала:
s0, e0 = int(row["start"]), int(row["end"])
import matplotlib.pyplot as plt

t = np.arange(len(sig_test)) / fs_test
plt.figure(figsize=(8, 3))
plt.plot(
    t[s0 - int(1 * fs_test) : e0 + int(1 * fs_test)],
    sig_test[s0 - int(1 * fs_test) : e0 + int(1 * fs_test)],
    label="Около цикла",
)
plt.axvspan(s0 / fs_test, e0 / fs_test, color="red", alpha=0.3, label="Граница цикла")
plt.xlabel("Время, с")
plt.legend()
plt.title("Участок вокруг явного выброса на 200 с")
plt.tight_layout()
plt.show()
